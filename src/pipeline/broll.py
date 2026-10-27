"""M7: b-roll — план вставок от vision-модели и добыча картинок.

Порядок источников задаёт `broll_source`:

- `generate` (по умолчанию) — картинку РИСУЮТ по описанию кадра, сток остаётся
  запасным путём. Так вставка попадает в смысл фразы и в палитру ролика, а не в
  ближайшее к слову изображение с чужого стола. Ровно этим эталонный ролик и
  отличается от нашего: там графика заказная (`STYLE_PLAN.md §5`).
- `stock` — прежний порядок: клип Pexels → фото Pexels → генерация.

Сток при этом можно выключить целиком, не трогая код: без `PEXELS_API_KEY`
`_fetch_stock` сразу возвращает `None`, и единственным источником остаётся
генерация. Не нарисовалось — вставка отменяется молча, как и раньше при полном
провале. Так оно и настроено сейчас.

Результат кешируется по хешу запроса: один и тот же «vintage film camera» не
должен стоить дважды, а генерация, в отличие от стока, стоит денег.

⚠ Моделям запрещено рисовать текст на картинках: кириллицу они врут (§5 M7).
Надписи накладывает слой субтитров.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import httpx

from src.core.config import Settings, settings as default_settings
from src.core.edl import EDL, Broll, BrollAnim, Timeline
from src.core.proc import ProcError, ffmpeg, run
from src.llm import images as imagegen
from src.llm.client import LLMBadOutput, LLMClient, LLMUnavailable
from src.llm.schemas import BROLL_SCHEMA, IMAGE_CHECK_SCHEMA
from src.pipeline.director import words_table
from src.pipeline.probe import MediaInfo, probe

log = logging.getLogger(__name__)

from src.core.edl import BROLL_MAX, BROLL_WINDOW  # потолок один на весь проект
FRAMES_FOR_VISION = 4
PEXELS_URL = "https://api.pexels.com/v1/search"

# Какую нарезку брать у Pexels. Параметр `orientation=portrait` в поиске
# отбирает вертикальные СНИМКИ, но не влияет на то, какую версию файла отдадут.
#
# Замерено на живом ключе: `large` приходит зажатым по высоте 650 — реальный
# файл 433x650. Для кадра шириной 1080 это мыло. `portrait` отдаёт 800x1200,
# то есть вдвое больше и ровно в нужной пропорции.
PEXELS_RENDITION = "portrait"
PEXELS_FALLBACK_RENDITIONS = ("large2x", "original", "large")

# Дописка к любому промпту генерации.
#
# ⚠ Проверено на реальной генерации: одного «no text» мало — модель всё равно
# рисует на предметах фальшивые надписи и перевёрнутые буквы (§5 M7 ⚠).
# Поэтому просим не только убрать текст, но и выбрать ракурс, где надписей
# просто не бывает: поверхность без брендов, никаких экранов и упаковок.
#
# Стилевых слов здесь больше нет. Раньше сюда была вшита «photographic,
# natural light, shallow depth of field» — то есть у каждой вставки любого
# ролика был один и тот же навязанный вид. Стиль теперь приезжает из самого
# ролика (`ReelStyle`), и зашитая фотографичность ему прямо мешала: эталонная
# вставка — не фотография, а нарисованный кадр.
NO_TEXT_SUFFIX = (
    "clean unbranded surfaces, no screens, no packaging, "
    "no text, no letters, no numbers, no words, no captions, no signage, "
    "no watermark, no logo, no brand name, no user interface, no typography"
)

# Чем дописывается промпт, когда памятки нет: модель не вернула `style` или
# её вовсе не спрашивали. Нейтрально и коротко — лучше, чем пустота, и
# заведомо хуже, чем снятое с кадров ролика.
FALLBACK_STYLE_HINT = "cinematic, restrained colours, soft directional light"

# Композиция кадра. Едет отдельной строкой, а не внутри промпта модели: это
# наша раскладка, а не её замысел (§0). Вертикаль и запас по краям — потому
# что сверху и снизу лягут субтитры.
COMPOSITION_HINT = (
    "vertical 9:16 composition, single subject centred, "
    "generous empty margin at the top and bottom of the frame"
)

# --- подача картинки (Ф7) ---------------------------------------------------
#
# По умолчанию вставка полноэкранная. Карточка в углу поверх говорящей головы
# читается как новостная плашка, а не как иллюстрация, поэтому угловой якорь
# оставлен ровно одному сценарию — `fullscreen_to_inset`, где угол это конечная
# точка перелёта, а не способ подачи.
INSET_ANIM = BrollAnim.FULLSCREEN_TO_INSET
# Углы, и только они: `center` — это ровно то место, где лицо говорящего, а
# свободной полосы между лицом и субтитром в кадре нет (`broll_free_band`).
INSET_ANCHORS = {"top_left", "top_right", "bottom_left", "bottom_right"}
DEFAULT_ANCHOR = "fullscreen"
DEFAULT_INSET_ANCHOR = "top_right"

# Константы перелёта продублированы из render/src/layers/Broll.tsx (HOLD_MS,
# HOLD_RATIO, TRAVEL_MS, FADE_MS) и render/src/style.ts (MIN_ANIM_FRAMES).
# Дублировать приходится: порог нужен на этапе плана, а fps здесь ещё неизвестен.
# Числа обязаны совпадать с TSX — это сверяет
# `test_landed_card_stands_still_long_enough_to_be_seen` в tests/test_broll_fetch.py.
HOLD_MS, HOLD_RATIO, TRAVEL_MS, FADE_MS, MIN_ANIM_FRAMES = 900.0, 0.35, 520.0, 220.0, 8
# Сколько карточка обязана просто СТОЯТЬ в углу после посадки. Прежний минимум
# 1.6 с оставлял на покой 5-7 кадров (0.21 с при 24 fps): картинка прилетала и
# сразу начинала таять, то есть дефект AUDIT §8.3 был закрыт только формально.
REST_SECONDS = 0.5
# Частоты, на которых ролик может рендериться. Порог берётся по худшей из них:
# на 24 fps пол в 8 кадров длиннее в миллисекундах, чем заказанное затухание.
FPS_RANGE = (24, 25, 30, 50, 60)


def inset_floor_seconds(fps: int) -> float:
    """Минимальная длительность перелёта в угол: удержание + полёт + уход + покой.

    Считается ровно теми же формулами, что и анимация в Broll.tsx: `animFrames`
    с полом в 8 кадров и `holdFrames = min(animFrames(HOLD_MS), span·HOLD_RATIO)`.
    """
    def frames(ms: float) -> int:
        return max(MIN_ANIM_FRAMES, round(ms / 1000 * fps))

    tail = frames(TRAVEL_MS) + frames(FADE_MS) + REST_SECONDS * fps
    span = tail / (1 - HOLD_RATIO)
    if span * HOLD_RATIO > frames(HOLD_MS):  # удержание упёрлось в свой потолок
        span = tail + frames(HOLD_MS)
    return span / fps


# Длительность вставки: (минимум, максимум) в секундах, по типу подачи.
# Полный кадр меньше 1.5 с не успевает прочитаться, дольше 2.5 — зритель теряет
# говорящего.
FULLSCREEN_SECONDS = (1.5, 2.5)
INSET_SECONDS = (math.ceil(max(inset_floor_seconds(f) for f in FPS_RANGE) * 10) / 10, 3.0)

# Слова, которые нечем проиллюстрировать. Правило «только конкретные референты»
# стоит в промпте, но промпт — пожелание: модель регулярно просит «success» и
# «motivation», и сток честно отдаёт по ним рукопожатие в офисе или рассвет.
# Такая вставка не добавляет смысла, она добавляет шум, поэтому фильтр стоит
# после модели и выбрасывает запрос, в котором нет ни одного конкретного слова.
# Список — данные рядом с кодом: правится без трогания промпта.
#
# В списке только то, что сток иллюстрирует штампом. Снимаемые существительные
# сюда не попадают: по «nature» приходит настоящий пейзаж, а не рукопожатие.
ABSTRACT_WORDS = frozenset(
    """
    abstract achievement ambition attention balance belief business career
    challenge change choice
    commitment communication community concept confidence consciousness content
    control
    creativity culture decision dedication desire determination development
    discipline dream education effort emotion energy engagement enthusiasm
    excellence experience failure faith fear feeling focus freedom friendship
    future goal growth habit happiness harmony health history hope idea identity
    imagination impact importance improvement information innovation inspiration
    intelligence intention interest joy justice knowledge leadership learning
    life love luck marketing meaning memory mind mindset moment motivation
    mystery
    opportunity optimism option passion patience peace perception performance
    persistence personality perspective philosophy plan pleasure potential power
    practice principle priority problem process productivity progress purpose
    quality reality reason relationship respect responsibility result risk safety
    satisfaction secret self sense service simplicity situation skill society
    solution strategy strength stress success support system talent
    teamwork thinking thought time trust truth understanding value vision wisdom
    work
    """.split()
)

# Служебные слова: сами по себе ничего не иллюстрируют, но и абстракцией запрос
# не делают — при подсчёте их просто не видно.
GLUE_WORDS = frozenset(
    "a an the of in on at to for with without and or from by as into over under about".split()
)


class BrollError(RuntimeError):
    pass


# --------------------------------------------------------------------------- стиль


@dataclass(slots=True)
class ReelStyle:
    """Как выглядит ЭТОТ ролик: палитра, свет, настроение.

    Заполняется один раз на ролик тем же vision-вызовом, который планирует
    вставки, — он и так смотрит на кадры, поэтому отдельного запроса и
    отдельных денег памятка не стоит.

    Смысл её в том, что промпт вставки описывает ПРЕДМЕТ, а памятка — мир, в
    котором этот предмет снят. Без неё две вставки одного ролика приходят из
    двух разных миров, и обе — не из того, куда их вставляют.
    """

    palette: str = ""
    light: str = ""
    mood: str = ""

    @classmethod
    def from_raw(cls, raw: Any) -> "ReelStyle":
        if not isinstance(raw, dict):
            return cls()
        def text(name: str) -> str:
            value = raw.get(name)
            return value.strip()[:120] if isinstance(value, str) else ""
        return cls(palette=text("palette"), light=text("light"), mood=text("mood"))

    def hint(self) -> str:
        parts = [p for p in (self.mood, self.palette, self.light) if p]
        return ", ".join(parts) if parts else FALLBACK_STYLE_HINT

    def __bool__(self) -> bool:
        return bool(self.palette or self.light or self.mood)


def build_image_prompt(subject: str, style: ReelStyle | None, *, fix: str = "") -> str:
    """Полный промпт генерации: предмет → мир ролика → раскладка → запрет текста.

    Порядок не случаен. Первым идёт предмет, потому что именно он обязан
    выжить, если модель обрежет длинный промпт; запрет надписей идёт последним,
    потому что на него модели реагируют тем хуже, чем раньше он упомянут
    (проверено на живых генерациях, §5 M7 ⚠).
    """
    hint = (style or ReelStyle()).hint()
    parts = [subject.strip().rstrip(".")]
    if fix:
        parts.append(fix.strip().rstrip("."))
    # Памятка помечена словами «цвет и свет»: без пометки модель читает её как
    # продолжение описания сцены и пытается мирить несовместимое — «горный
    # хребет на закате» плюс «уютный вечерний интерьер» из памятки. Пометка
    # переводит её в разряд обработки, а место остаётся за промптом вставки.
    parts += [f"Colour and light: {hint}", COMPOSITION_HINT, NO_TEXT_SUFFIX]
    return ". ".join(p for p in parts if p)


@dataclass(slots=True)
class BrollPlan:
    """Что vision-вызов решил про вставки: сами вставки, стиль и кадры ролика.

    Кадры возвращаются вместе с планом, а не перечитываются с диска: они уже
    прочитаны для самого вызова, и именно они идут референсом в генерацию.
    """

    items: list[Broll]
    style: ReelStyle
    frames: list[bytes]


# --------------------------------------------------------------------------- кадры


def extract_frames(
    src: str | Path, dst_dir: Path, *, count: int = FRAMES_FOR_VISION, info: MediaInfo | None = None
) -> list[Path]:
    """Равномерные кадры для vision-вызова. JPEG, узкая сторона до 768 px."""
    src = Path(src)
    info = info or probe(src)
    dst_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    span = max(0.5, info.duration)
    for i in range(count):
        # Не с самого начала и не с самого конца: там обычно заставка и затемнение.
        t = span * (i + 0.5) / count
        path = dst_dir / f"frame{i}.jpg"
        ffmpeg(
            ["-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
             "-vf", "scale='min(768,iw)':-2", "-q:v", "4", str(path)],
            timeout=120,
        )
        if path.exists():
            out.append(path)
    return out


# --------------------------------------------------------------------------- план


def _cap(edl: EDL) -> int:
    return min(BROLL_MAX, max(0, math.ceil(edl.out_duration / BROLL_WINDOW)))


def _is_abstract_token(token: str) -> bool:
    # Множественное число режется наивно: списку хватает единственного, а
    # «ideas» и «goals» модель просит не реже, чем «idea» и «goal».
    return token in ABSTRACT_WORDS or (token.endswith("s") and token[:-1] in ABSTRACT_WORDS)


def is_abstract_query(query: str) -> bool:
    """Нечего снимать: в запросе нет ни одного конкретного слова.

    Числа и незнакомые слова считаются конкретными — стоп-лист перечисляет то,
    что заведомо не иллюстрируется, а не то, что разрешено.
    """
    tokens = [t for t in re.findall(r"\w+", query.lower()) if t not in GLUE_WORDS]
    if not tokens:
        return True
    return all(_is_abstract_token(t) for t in tokens)


def _delivery(anim: BrollAnim, anchor: Any) -> tuple[str, float, float]:
    """Куда ставим картинку и в какие границы зажимаем её длительность.

    Решение о подаче — раскладка кадра, а не смысл, поэтому его принимает Python,
    а не модель (§0): модель выбирает, ЧТО показать и КОГДА, а полноэкранная это
    вставка или карточка в углу — следствие выбранной анимации.
    """
    if anim is INSET_ANIM:
        rest = anchor if anchor in INSET_ANCHORS else DEFAULT_INSET_ANCHOR
        return rest, *INSET_SECONDS
    return DEFAULT_ANCHOR, *FULLSCREEN_SECONDS


def sanitize_items(edl: EDL, items: list[dict[str, Any]]) -> list[Broll]:
    """Обрезает план до того, что валидатор пропустит: потолок, плотность, границы.

    Модель почти всегда просит больше картинок, чем можно. Решение «сколько
    влезет» — арифметика, значит наша, а не её (§0).
    """
    known = {w.idx for w in edl.words}
    tl = edl.timeline()
    by_idx = {w.idx: w for w in edl.words}

    scored: list[tuple[float, Broll]] = []
    for raw in items:
        idx = raw.get("word_idx")
        if not isinstance(idx, int) or idx not in known:
            continue
        word = by_idx[idx]
        window = tl.word_window(word)
        if window is None:  # слово вырезано монтажом
            continue
        query = (raw.get("query") or "").strip()
        if not query:
            continue
        if is_abstract_query(query):
            # INFO, а не DEBUG: отброшенная вставка меняет ролик, и её обязано
            # быть видно в логе прогона, а не только при отладке.
            log.info("вставка отброшена как абстрактная: %r", query)
            continue
        try:
            anim = BrollAnim(raw.get("anim") or "pop_in")
            anchor, lo, hi = _delivery(anim, raw.get("anchor"))
            item = Broll(
                word_idx=idx,
                duration=min(hi, max(lo, float(raw.get("duration") or lo))),
                query=query,
                prompt=(raw.get("prompt") or None),
                anim=anim,
                anchor=anchor,
            )
        except (ValueError, TypeError) as e:
            log.debug("вставка отброшена (%s): %r", e, raw)
            continue
        scored.append((window[0], item))

    scored.sort(key=lambda p: p[0])
    kept: list[Broll] = []
    last_at = -1e9
    for at, item in scored:
        if at - last_at < BROLL_WINDOW:
            continue
        kept.append(item)
        last_at = at
        if len(kept) >= _cap(edl):
            break
    # Направление наезда чередуется по позиции в ролике: две вставки подряд,
    # едущие в одну сторону, читаются как один длинный проезд. Из индекса, а не
    # из `random`, — иначе перерендер того же листа даёт другое видео (М3).
    for i, item in enumerate(kept):
        item.kb_dir = 1 if i % 2 else -1
    return kept


async def plan_broll(
    edl: EDL,
    src: str | Path,
    work: Path,
    *,
    llm: LLMClient,
    info: MediaInfo | None = None,
    cfg: Settings | None = None,
) -> BrollPlan:
    """Vision-вызов: кадры + транскрипт → план вставок и стиль ролика."""
    cfg = cfg or default_settings
    cap = _cap(edl)
    if cap <= 0:
        return BrollPlan([], ReelStyle(), [])

    frames = extract_frames(src, work / "frames", info=info)
    images = [p.read_bytes() for p in frames]

    variables = {
        "language": cfg.whisper_language or "auto",
        "duration": f"{edl.out_duration:.1f}",
        "max_items": str(cap),
        "words_table": words_table(edl.words),
    }
    try:
        result = await llm.complete(
            "plan_broll",
            variables,
            BROLL_SCHEMA,
            "vision",
            images=images,
            # У reasoning-моделей размышления считаются в max_tokens: с тесным
            # бюджетом ответ обрывается на полуслове (stop=length) и b-roll
            # молча отваливается. Проверено — 4000 не хватает.
            max_tokens=16_000
        )
    except (LLMUnavailable, LLMBadOutput) as e:
        log.warning("план b-roll не получен: %s", e)
        return BrollPlan([], ReelStyle(), images)

    items = sanitize_items(edl, result.data.get("items") or [])
    style = ReelStyle.from_raw(result.data.get("style"))
    log.info("b-roll: модель предложила %d, оставлено %d (потолок %d)",
             len(result.data.get("items") or []), len(items), cap)
    if style:
        log.info("стиль ролика: %s", style.hint())
    return BrollPlan(items, style, images)


# --------------------------------------------------------------------------- картинки


def cache_key(query: str, prompt: str | None, size: str, *, model: str = "", style: str = "") -> str:
    # Нарезка входит в ключ намеренно: сменили её — старые файлы другого
    # разрешения перестают подходить, и кеш обязан это заметить сам.
    #
    # Модель и стиль — по той же причине. Один и тот же «vintage film camera»,
    # нарисованный другой моделью или под другую палитру, — это ДРУГАЯ
    # картинка, и отдавать вместо неё старую значит молча проигнорировать
    # смену настроек. Оба поля пустые у стока, поэтому его ключи не съехали.
    raw = f"{query}|{prompt or ''}|{size}|{PEXELS_RENDITION}|{model}|{style}".lower().strip()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


async def fetch_pexels(client: httpx.AsyncClient, query: str, api_key: str) -> bytes | None:
    """Сток: дешевле генерации и выглядит достовернее."""
    try:
        r = await client.get(
            PEXELS_URL,
            params={"query": query, "orientation": "portrait", "per_page": 3, "size": "medium"},
            headers={"Authorization": api_key},
            timeout=30,
        )
        r.raise_for_status()
        photos = r.json().get("photos") or []
        if not photos:
            return None
        src = photos[0].get("src") or {}
        url = next(
            (src[k] for k in (PEXELS_RENDITION, *PEXELS_FALLBACK_RENDITIONS) if src.get(k)),
            None,
        )
        if not url:
            log.warning("Pexels не отдал ни одной нарезки по %r", query)
            return None
        img = await client.get(url, timeout=60)
        img.raise_for_status()
        return img.content
    except (httpx.HTTPError, KeyError, ValueError) as e:
        log.warning("Pexels не отдал картинку по %r: %s", query, e)
        return None


# --- видео вместо фото (М4) -------------------------------------------------
#
# Двухсекундный клип в кадре читается как монтаж мгновенно, и никакой Ken Burns
# на стоп-кадре этого не заменит. Платим за это трижды: скачиванием (клип 1080p
# на 3 секунды — единицы мегабайт против 200 КБ у фото), рендером (второй
# видеопоток на композитинге) и релевантностью — видео на Pexels в среднем хуже
# по попаданию в запрос, чем фото.
#
# Поэтому порядок такой: сначала видео, при неудаче молча фото. «Неудача» здесь
# — это жёсткие фильтры по метаданным, а не оценка содержимого; чего именно тут
# НЕ делается и почему, написано у `fetch_pexels_video`.
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"

# Во сколько раз нарезка может быть шире кадра. 1.5, а не 2.0, и это ровно та
# граница, что отсекает 4K: лестница Pexels идёт 1080 / 1440 / 2160, и при
# потолке 2.0 нарезка 2160 в кадр 1080 проходила (2160 <= 2160). Она
# декодируется поверх основного видео на каждом кадре рендера, а видно от неё
# ровно то же, что от 1440.
VIDEO_MAX_OVERSIZE = 1.5
# Клип обязан быть ДЛИННЕЕ вставки с запасом: ролик, обрезанный ровно в свой
# конец, на последнем кадре показывает чёрное — файл кончился раньше, чем
# композиция. Полсекунды хватает и на округление кадров, и на неточность
# длительности в метаданных стока.
VIDEO_MARGIN_S = 0.5
# Потолок веса файла. Не оптимизация, а предохранитель: за ним стоит время
# скачивания в общем таймауте этапа и место в кеше.
VIDEO_MAX_BYTES = 32 * 1024 * 1024


def pick_video_file(files: list[dict[str, Any]], *, frame_width: int) -> dict[str, Any] | None:
    """Из нарезок Pexels — самая маленькая, которая ещё покрывает кадр по ШИРИНЕ.

    Именно по ширине, и это не придирка. Первая версия фильтровала по высоте
    («не ниже 720»), и на живом стоке она выбрала 540x960: высота 960 в коридор
    попадала, а ширина 540 — это двукратный апскейл в кадр 1080, то есть мыло.
    Вертикальные нарезки Pexels идут 360x640 / 540x960 / 720x1280 / 1080x1920 /
    1440x2560 / 2160x3840, и по высоте эти пары не различить без пропорции.

    Не самая большая: 4K декодируется поверх основного видео на каждом кадре
    рендера, а в кадре 1080 шириной от него не остаётся ничего.

    Ничего не покрывает кадр — возвращаем None и уходим на фото. Растянутый
    вдвое клип хуже резкой фотографии, даже если он и «монтажнее».
    """
    ceiling = frame_width * VIDEO_MAX_OVERSIZE
    ok = [
        f
        for f in files
        if f.get("link")
        and str(f.get("file_type") or "").endswith("mp4")
        and frame_width <= int(f.get("width") or 0) <= ceiling
    ]
    if not ok:
        return None
    return min(ok, key=lambda f: int(f["width"]))


def video_start(duration: float, need: float) -> float:
    """С какой секунды играть клип: серединой окна, а не с начала.

    У стоковых видео первые кадры самые статичные — оператор ещё не начал
    движение. Берём кусок из середины: он же и самый устойчивый по экспозиции.
    """
    slack = duration - need
    return max(0.0, slack / 2.0)


async def fetch_pexels_video(
    client: httpx.AsyncClient,
    query: str,
    api_key: str,
    need: float,
    *,
    frame_width: int,
) -> tuple[bytes, dict[str, Any]] | None:
    """Сток, видео. `need` — сколько секунд вставка реально займёт в ролике.

    Отбор идёт по МЕТАДАННЫМ: вертикальная ориентация, длительность с запасом,
    высота в коридоре, вес в пределах потолка. Содержимое клипа здесь никто не
    смотрит.

    Это сознательный отказ от того, что просит MOTION_PLAN («оценку релевантности
    делает тот же vision-вызов»). Вызов, который планирует b-roll, идёт ДО
    поиска: у него на руках кадры ролика и транскрипт, а кандидата ещё нет.
    Оценить кандидата — это отдельный второй vision-вызов на скачанный клип,
    то есть плюс запрос и плюс задержка на каждую вставку, при том что фолбэк
    на фото и без него срабатывает молча. Ставка на порядок выдачи стока
    сделана осознанно и отмечена здесь, а не выдана за реализованную оценку.
    """
    try:
        r = await client.get(
            PEXELS_VIDEO_URL,
            params={"query": query, "orientation": "portrait", "per_page": 5},
            headers={"Authorization": api_key},
            timeout=30,
        )
        r.raise_for_status()
        videos = r.json().get("videos") or []
    except (httpx.HTTPError, KeyError, ValueError) as e:
        log.warning("Pexels не отдал видео по %r: %s", query, e)
        return None

    # Из подходящих берём САМЫЙ КОРОТКИЙ. Сток отдаёт файл целиком, обрезать его
    # по дороге нечем, а из клипа берётся две секунды: тринадцатисекундный 1080p
    # это 23.7 МБ на живом ключе против плановых «3-8». Порядок выдачи Pexels при
    # равной длительности сохраняется — сортировка устойчивая, релевантность не
    # ломается.
    for v in sorted(videos, key=lambda v: float(v.get("duration") or 0)):
        duration = float(v.get("duration") or 0)
        if duration < need + VIDEO_MARGIN_S:
            continue
        picked = pick_video_file(v.get("video_files") or [], frame_width=frame_width)
        if picked is None:
            continue
        try:
            got = await client.get(picked["link"], timeout=120)
            got.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("клип не скачался (%s): %s", e, picked["link"])
            continue
        if len(got.content) > VIDEO_MAX_BYTES:
            log.info(
                "клип по %r весит %.1f МБ при потолке %.0f — пропускаю",
                query, len(got.content) / 1024 / 1024, VIDEO_MAX_BYTES / 1024 / 1024,
            )
            continue
        meta = {
            "duration": duration,
            "width": int(picked.get("width") or 0),
            "height": int(picked.get("height") or 0),
            "start": video_start(duration, need + VIDEO_MARGIN_S),
        }
        log.info(
            "b-roll видео по %r: %dx%d, %.1f с, вход с %.1f с (%.1f МБ)",
            query, meta["width"], meta["height"], duration, meta["start"],
            len(got.content) / 1024 / 1024,
        )
        return got.content, meta

    log.info("подходящего видео по %r нет — беру фото", query)
    return None


# --- генерация (вместо поиска) ----------------------------------------------


async def check_image(
    llm: LLMClient | None, data: bytes, *, prompt: str, subject: str
) -> tuple[bool, str]:
    """Приёмка картинки: `(годится, что поправить)`.

    Спрашиваем у vision-модели две вещи — то ли нарисовано и нет ли букв.
    Второе важнее: фальшивая надпись видна в кадре сразу, а заметить её в
    готовом ролике уже поздно.

    Любая осечка самой проверки — это «годится». Приёмка не должна отменять
    вставку, которую, возможно, не за что отменять: цена ошибки во вторую
    сторону (лишний кадр в ролике) ниже, чем пустота на месте картинки.
    """
    if llm is None or not llm.ready:
        return True, ""
    try:
        result = await llm.complete(
            "check_image",
            {"prompt": prompt, "subject": subject},
            IMAGE_CHECK_SCHEMA,
            "vision",
            images=[data],
            max_tokens=2_000,
        )
    except (LLMUnavailable, LLMBadOutput) as e:
        log.debug("приёмка картинки не сработала: %s", e)
        return True, ""
    shows = bool(result.data.get("shows_subject", True))
    has_text = bool(result.data.get("has_text", False))
    fix = result.data.get("fix")
    return (shows and not has_text), (fix if isinstance(fix, str) else "")


async def generate_image(
    client: httpx.AsyncClient,
    item: Broll,
    cfg: Settings,
    *,
    style: ReelStyle | None = None,
    refs: Sequence[bytes] = (),
    llm: LLMClient | None = None,
) -> bytes | None:
    """Нарисовать вставку: промпт + референсы, при надобности вторая попытка.

    Повторов ровно один. Второй провал означает, что модель не понимает этот
    предмет, и дальнейшие попытки только жгут деньги и время: уходим на сток,
    который для того и остался.
    """
    if not imagegen.ready(cfg):
        return None
    subject = item.prompt or item.query
    prompt = build_image_prompt(subject, style)
    data = await imagegen.generate(client, prompt, refs=refs, cfg=cfg)
    if data is None:
        return None
    if not cfg.broll_verify:
        return data

    ok, fix = await check_image(llm, data, prompt=subject, subject=item.query)
    if ok:
        return data
    log.info("картинка %r не принята (%s) — вторая попытка", item.query, fix or "без пояснения")
    retry = await imagegen.generate(
        client, build_image_prompt(subject, style, fix=fix), refs=refs, cfg=cfg
    )
    if retry is None:
        return None
    ok, _ = await check_image(llm, retry, prompt=subject, subject=item.query)
    if ok:
        return retry
    log.info("картинка %r не принята и со второй попытки — иду на сток", item.query)
    return None


@dataclass(slots=True)
class Asset:
    """Добытый файл вставки: чем он оказался и как его играть."""

    path: Path
    kind: str = "photo"
    # Секунда внутри клипа, с которой начинать. У фото всегда 0.
    start: float = 0.0


# Как долго кешу верить, что записанный рядом с клипом вход в него — про этот
# самый файл. Никак: вход кладётся в имя, а не в отдельный индекс, чтобы кеш не
# мог рассинхронизироваться со своим содержимым.
_VIDEO_START_SEP = "@"


def _cached_video(cache_dir: Path, key: str) -> Asset | None:
    """Клип из кеша вместе с точкой входа: она зашита в имя файла."""
    for path in cache_dir.glob(f"{key}{_VIDEO_START_SEP}*.mp4"):
        if path.stat().st_size <= 1024:
            continue
        try:
            start = float(path.stem.split(_VIDEO_START_SEP, 1)[1])
        except (IndexError, ValueError):
            continue
        return Asset(path, "video", start)
    return None


async def _fetch_stock(
    client: httpx.AsyncClient, item: Broll, cache_dir: Path, cfg: Settings, key: str
) -> Asset | None:
    """Сток: клип → фото. Клип впереди (М4) — он читается как монтаж сразу.

    Видео есть не на всякий запрос, поэтому откат на фото обязан быть
    молчаливым: для вызывающего кода разницы нет, он получает файл.
    """
    if cfg.broll_video and cfg.pexels_api_key:
        hit = _cached_video(cache_dir, key)
        if hit is not None:
            log.debug("b-roll видео из кеша: %s", item.query)
            return hit
        got = await fetch_pexels_video(
            client, item.query, cfg.pexels_api_key, item.duration,
            frame_width=cfg.render_width,
        )
        if got is not None:
            data, meta = got
            start = round(float(meta["start"]), 2)
            path = cache_dir / f"{key}{_VIDEO_START_SEP}{start}.mp4"
            path.write_bytes(data)
            return Asset(path, "video", start)

    if not cfg.pexels_api_key:
        return None
    data = await fetch_pexels(client, item.query, cfg.pexels_api_key)
    if not data:
        return None
    cached = cache_dir / f"{key}.jpg"
    cached.write_bytes(data)
    return Asset(cached)


async def fetch_asset(
    client: httpx.AsyncClient,
    item: Broll,
    cache_dir: Path,
    cfg: Settings,
    *,
    style: ReelStyle | None = None,
    refs: Sequence[bytes] = (),
    llm: LLMClient | None = None,
) -> Asset | None:
    """Достать картинку для вставки. Порядок источников решает `broll_source`.

    `generate` — рисуем, а не ищем: сток отдаёт чужой предмет в чужом кадре, и
    на переходе это видно. Не нарисовалось — уходим на сток, он для того и
    остался. `stock` возвращает прежний порядок целиком.

    Что бы ни вышло, результат один: файл на диске или `None`. Вызывающий код
    про источник не знает и знать не должен.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    generating = cfg.broll_source == "generate" and imagegen.ready(cfg)
    # Ключ генерации помнит модель и стиль, ключ стока — нет: сток по тому же
    # запросу отдаёт то же самое независимо от наших настроек.
    gen_key = cache_key(
        item.query,
        item.prompt,
        imagegen.size_tag(cfg),
        model=imagegen.image_model(cfg),
        style=(style or ReelStyle()).hint(),
    )
    stock_key = cache_key(item.query, item.prompt, cfg.wavespeed_size)

    if generating:
        cached = cache_dir / f"{gen_key}.jpg"
        if cached.exists() and cached.stat().st_size > 1024:
            log.debug("b-roll из кеша (генерация): %s", item.query)
            return Asset(cached)
        data = await generate_image(client, item, cfg, style=style, refs=refs, llm=llm)
        if data:
            cached.write_bytes(data)
            return Asset(cached)
        log.info("нарисовать %r не вышло — пробую сток", item.query)

    hit_dir = cache_dir / f"{stock_key}.jpg"
    if hit_dir.exists() and hit_dir.stat().st_size > 1024:
        log.debug("b-roll из кеша: %s", item.query)
        return Asset(hit_dir)
    asset = await _fetch_stock(client, item, cache_dir, cfg, stock_key)
    if asset is not None:
        return asset

    # Сток пуст, а генерацию либо не звали, либо она не была первой. Пробуем
    # её последней — это прежнее поведение `stock`, и оно же последний шанс.
    if not generating and imagegen.ready(cfg):
        data = await generate_image(client, item, cfg, style=style, refs=refs, llm=llm)
        if data:
            cached = cache_dir / f"{gen_key}.jpg"
            cached.write_bytes(data)
            return Asset(cached)
    return None


def asset_aspect(path: str | Path) -> float | None:
    """Пропорция скачанного файла (ширина/высота) или None, если не прочитался.

    Нужна геометрии: карточка наследует пропорцию источника, иначе `cover` режет
    (AUDIT §8.3). Сток отдаёт 2:3, генерация — заказанную вертикаль 9:16, клип —
    свою третью; числа разные, поэтому меряется файл, а не источник.
    """
    try:
        out = run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-print_format", "json", str(path)],
            timeout=20,
        )
        stream = (json.loads(out).get("streams") or [{}])[0]
        w, h = int(stream.get("width") or 0), int(stream.get("height") or 0)
    except (ProcError, ValueError, TypeError, IndexError, OSError) as e:
        log.debug("не удалось измерить %s: %s", path, e)
        return None
    return w / h if w > 0 and h > 0 else None


def _ref_frame(frames: Sequence[bytes], at: float, duration: float) -> list[bytes]:
    """Кадр ролика, ближайший к моменту вставки.

    Кадры сняты равномерно (`extract_frames`: t = span·(i+0.5)/count), поэтому
    номер восстанавливается арифметикой, а не поиском. Берём именно ближний, а
    не первый: свет в ролике меняется, и вставка обязана попасть в тот кусок,
    рядом с которым она встанет.
    """
    if not frames or duration <= 0:
        return list(frames[:1])
    i = int(round(at / duration * len(frames) - 0.5))
    return [frames[max(0, min(len(frames) - 1, i))]]


async def _generate_series(
    client: httpx.AsyncClient,
    items: list[Broll],
    cache_dir: Path,
    cfg: Settings,
    *,
    plan: BrollPlan,
    llm: LLMClient | None,
    tl: Timeline,
    by_idx: dict[int, Any],
    duration: float,
) -> list[Asset | None]:
    """Вставки рисуются по очереди, и каждая следующая видит предыдущую.

    Очередь здесь не от бедности, а ради связности: вторая картинка ролика,
    сгенерированная с первой перед глазами, выходит из того же комплекта, а не
    из соседнего. Потолок вставок — две (`BROLL_MAX`), так что очередь стоит
    одного лишнего ожидания на ролик, и на фоне рендера это не видно.
    """
    out: list[Asset | None] = []
    made: list[bytes] = []
    for item in items:
        word = by_idx.get(item.word_idx)
        window = tl.word_window(word) if word is not None else None
        refs = _ref_frame(plan.frames, window[0] if window else 0.0, duration)
        # Предыдущая вставка идёт ВТОРЫМ референсом: первым обязан остаться
        # кадр ролика, иначе комплект начнёт уезжать от самого видео.
        refs = refs + made[-1:]
        asset = await fetch_asset(
            client, item, cache_dir, cfg, style=plan.style, refs=refs, llm=llm
        )
        out.append(asset)
        if asset is not None and asset.kind == "photo":
            try:
                made.append(asset.path.read_bytes())
            except OSError:
                pass
    return out


async def plan_and_fetch(
    edl: EDL,
    src: str | Path,
    work: Path,
    *,
    llm: LLMClient,
    info: MediaInfo | None = None,
    cfg: Settings | None = None,
) -> int:
    """Полный шаг M7. Проставляет `edl.broll` и возвращает число вставок."""
    cfg = cfg or default_settings
    work = Path(work)
    plan = await plan_broll(edl, src, work, llm=llm, info=info, cfg=cfg)
    items = plan.items
    if not items:
        edl.broll = []
        return 0

    cache_dir = Path(cfg.work_dir) / "broll-cache"
    tl = edl.timeline()
    by_idx = {w.idx: w for w in edl.words}
    generating = cfg.broll_source == "generate" and imagegen.ready(cfg)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # Очередь — только ради референсов: вторая картинка ролика рисуется с
        # первой перед глазами и выходит из того же комплекта. Провайдеру,
        # который референсы выбрасывает (WaveSpeed), очередь не даёт ничего и
        # стоит лишнего ожидания на каждую вставку. Сток — тем более: это
        # чистый ввод-вывод.
        if generating and imagegen.uses_refs(cfg):
            assets = await _generate_series(
                client, items, cache_dir, cfg,
                plan=plan, llm=llm, tl=tl, by_idx=by_idx, duration=edl.out_duration,
            )
        else:
            style = plan.style if generating else None
            assets = list(await asyncio.gather(
                *(
                    fetch_asset(client, item, cache_dir, cfg, style=style, llm=llm)
                    for item in items
                )
            ))

    resolved: list[Broll] = []
    for item, asset in zip(items, assets):
        if asset is None:
            log.info("картинка не найдена для %r — вставка отменена", item.query)
            continue
        item.asset = str(asset.path)
        item.kind = asset.kind
        item.asset_start = asset.start
        # Пропорция меряется у ФАЙЛА, а не берётся у источника: сток отдаёт 2:3,
        # генерация — заказанную вертикаль, а клип приходит своей третьей.
        # Карточка наследует её, иначе `cover` режет источник (AUDIT §8.3);
        # дальше число едет в `broll_rect` через план.
        item.asset_aspect = asset_aspect(asset.path)
        if item.asset_aspect:
            log.info(
                "вставка %r: %s, пропорция источника %.3f",
                item.query, asset.kind, item.asset_aspect,
            )
        resolved.append(item)

    edl.broll = resolved
    videos = sum(1 for b in resolved if b.kind == "video")
    log.info(
        "b-roll: получено %d из %d запланированных (видео %d, фото %d)",
        len(resolved), len(items), videos, len(resolved) - videos,
    )
    return len(resolved)


if __name__ == "__main__":  # проверка добычи одной картинки на реальном ключе
    import sys

    from src.core.logging import setup_logging

    setup_logging("INFO")

    async def _main() -> None:
        # python -m src.pipeline.broll "vintage film camera" [кадр.jpg]
        #
        # Второй аргумент — кадр ролика: с ним видно главное, ради чего затеяна
        # генерация, — попала ли картинка в палитру и свет исходника.
        query = sys.argv[1] if len(sys.argv) > 1 else "vintage film camera on a desk"
        refs = [Path(sys.argv[2]).read_bytes()] if len(sys.argv) > 2 else []
        item = Broll(
            word_idx=0, duration=1.5, query=query, anim=BrollAnim.POP_IN, anchor="top_right"
        )
        cache = Path(default_settings.work_dir) / "broll-cache"
        async with httpx.AsyncClient(follow_redirects=True) as client:
            got = await fetch_asset(client, item, cache, default_settings, refs=refs)
        print(
            json.dumps(
                {
                    "query": query,
                    "source": default_settings.broll_source,
                    "model": imagegen.image_model(default_settings),
                    "refs": len(refs),
                    "kind": got.kind if got else None,
                    "asset": str(got.path) if got else None,
                    "start": got.start if got else None,
                    "aspect": asset_aspect(got.path) if got else None,
                },
                ensure_ascii=False,
            )
        )

    asyncio.run(_main())
