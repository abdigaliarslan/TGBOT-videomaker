"""M7: b-roll — план вставок от vision-модели и генерация картинок.

Где и что показать, решает vision-модель по кадрам и транскрипту; картинку
РИСУЕТ генератор по описанию кадра. Так вставка попадает в смысл фразы и в
палитру ролика, а не в ближайшее к слову изображение с чужого стола: ровно этим
эталонный ролик и отличался от нашего, там графика заказная (`STYLE_PLAN.md §5`).

Стока больше нет. Не нарисовалось — вставка отменяется молча: пустое место
лучше чужой фотографии. Генератор не настроен — план вставок даже не
запрашивается, vision-вызов без картинок стоил бы денег впустую.

Результат кешируется по хешу запроса: один и тот же «vintage film camera» не
должен стоить дважды.

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

# Соль ключа кеша. Когда-то здесь стояла нарезка стока, и она входила в ключ
# всех картинок, нарисованных в том числе. Сток снят, а слово оставлено: убрать
# его — значит сдвинуть ключи всего кеша и заново заплатить за каждую уже
# нарисованную картинку.
_CACHE_SALT = "portrait"

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

# Запрет людей и то, что его снимает: если человек — сам предмет вставки
# («a barista pouring latte art»), запрещать его нельзя.
NO_PEOPLE_SUFFIX = "no people, no person, no figures, empty of humans"
PEOPLE_RE = re.compile(
    r"\b(people|person|persons|man|men|woman|women|girl|boy|child|children|kid|kids|"
    r"crowd|player|athlete|skater|barista|chef|worker|hand|hands|face|portrait|"
    r"tourist|tourists|family|couple|friends?)\b",
    re.IGNORECASE,
)

# Чем дописывается промпт, когда памятки нет: модель не вернула `style` или
# её вовсе не спрашивали. Нейтрально и коротко — лучше, чем пустота, и
# заведомо хуже, чем снятое с кадров ролика.
FALLBACK_STYLE_HINT = "cinematic, restrained colours, soft directional light"

# Композиция кадра. Едет отдельной строкой, а не внутри промпта модели: это
# наша раскладка, а не её замысел (§0). Портрет 4:5 — пропорция карточки, запас
# по краям — потому что углы у неё скруглены.
COMPOSITION_HINT = (
    "portrait 4:5 composition, single subject centred and fully in frame, "
    "small even margin around it"
)

# --- подача картинки (Ф7) ---------------------------------------------------
#
# По умолчанию вставка — крупная карточка по центру поверх размытого и
# притемнённого кадра (`card`), как в эталоне `style-showcase.mp4`: говорящий
# остаётся в кадре, и вставка читается как часть ролика, а не как подмена
# кадра. Под карточкой лицо не страшно — оно размыто вместе со всем фоном.
# Карточка в углу поверх НЕразмытой головы читается как новостная плашка,
# поэтому угловой якорь оставлен ровно одному сценарию — `fullscreen_to_inset`,
# где угол это конечная точка перелёта, а не способ подачи.
INSET_ANIM = BrollAnim.FULLSCREEN_TO_INSET
# Углы, и только они: `center` — это ровно то место, где лицо говорящего, а
# свободной полосы между лицом и субтитром в кадре нет (`broll_free_band`).
INSET_ANCHORS = {"top_left", "top_right", "bottom_left", "bottom_right"}
DEFAULT_ANCHOR = "card"
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
# «motivation», и генератор честно рисует по ним рукопожатие в офисе или
# рассвет. Такая вставка не добавляет смысла, она добавляет шум, поэтому фильтр
# стоит после модели и выбрасывает запрос, в котором нет ни одного конкретного
# слова. Список — данные рядом с кодом: правится без трогания промпта.
#
# В списке только то, что иллюстрируется штампом. Изобразимые существительные
# сюда не попадают: по «nature» выходит настоящий пейзаж, а не рукопожатие.
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
    parts += [f"Colour and light: {hint}", COMPOSITION_HINT]
    # Людей в кадре не бывает, если о них не просили. Памятка снята с кадров
    # говорящего, и модель охотно рисует по ней его самого: на тестовом ролике
    # «white t-shirt, beige skin tones» превратились в незнакомца в белой
    # футболке посреди катка Медеу.
    if not PEOPLE_RE.search(subject):
        parts.append(NO_PEOPLE_SUFFIX)
    parts.append(NO_TEXT_SUFFIX)
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
    # Размер, модель и стиль входят в ключ намеренно. Один и тот же «vintage
    # film camera», нарисованный другой моделью, другим размером или под другую
    # палитру, — это ДРУГАЯ картинка, и отдавать вместо неё старую значит молча
    # проигнорировать смену настроек.
    raw = f"{query}|{prompt or ''}|{size}|{_CACHE_SALT}|{model}|{style}".lower().strip()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# --- генерация -------------------------------------------------------------


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
    предмет, и дальнейшие попытки только жгут деньги и время: вставка
    отменяется.
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
    log.info("картинка %r не принята и со второй попытки — вставки не будет", item.query)
    return None


@dataclass(slots=True)
class Asset:
    """Нарисованный файл вставки."""

    path: Path


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
    """Картинка для вставки: из кеша, иначе нарисовать. Не вышло — `None`.

    Ключ помнит модель, размер и стиль: та же вставка под другую палитру — это
    другая картинка, и старая из кеша её не заменяет.
    """
    if not imagegen.ready(cfg):
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(
        item.query,
        item.prompt,
        imagegen.size_tag(cfg),
        model=imagegen.image_model(cfg),
        style=(style or ReelStyle()).hint(),
    )
    cached = cache_dir / f"{key}.jpg"
    if cached.exists() and cached.stat().st_size > 1024:
        log.debug("b-roll из кеша: %s", item.query)
        return Asset(cached)
    data = await generate_image(client, item, cfg, style=style, refs=refs, llm=llm)
    if not data:
        return None
    cached.write_bytes(data)
    return Asset(cached)


def asset_aspect(path: str | Path) -> float | None:
    """Пропорция нарисованного файла (ширина/высота) или None, если не прочитался.

    Нужна геометрии: карточка наследует пропорцию источника, иначе `cover` режет
    (AUDIT §8.3). Заказываем вертикаль 9:16, но провайдеры отдают её на пиксель
    мимо (1024×1820), поэтому меряется файл, а не заказ.
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
        if asset is not None:
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
    if not imagegen.ready(cfg):
        # Рисовать нечем — и спрашивать, где рисовать, незачем: vision-вызов
        # плана стоит денег, а его результат всё равно некуда деть.
        log.info("b-roll пропущен: генератор картинок не настроен (%s)", cfg.image_provider)
        edl.broll = []
        return 0
    plan = await plan_broll(edl, src, work, llm=llm, info=info, cfg=cfg)
    items = plan.items
    if not items:
        edl.broll = []
        return 0

    cache_dir = Path(cfg.work_dir) / "broll-cache"
    tl = edl.timeline()
    by_idx = {w.idx: w for w in edl.words}
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # Очередь — только ради референсов: вторая картинка ролика рисуется с
        # первой перед глазами и выходит из того же комплекта. Провайдеру,
        # который референсы выбрасывает (WaveSpeed, fal), очередь не даёт
        # ничего и стоит лишнего ожидания на каждую вставку.
        if imagegen.uses_refs(cfg):
            assets = await _generate_series(
                client, items, cache_dir, cfg,
                plan=plan, llm=llm, tl=tl, by_idx=by_idx, duration=edl.out_duration,
            )
        else:
            assets = list(await asyncio.gather(
                *(
                    fetch_asset(client, item, cache_dir, cfg, style=plan.style, llm=llm)
                    for item in items
                )
            ))

    resolved: list[Broll] = []
    for item, asset in zip(items, assets):
        if asset is None:
            log.info("картинка не нарисовалась для %r — вставка отменена", item.query)
            continue
        item.asset = str(asset.path)
        # Пропорция меряется у ФАЙЛА: карточка наследует её, иначе `cover` режет
        # источник (AUDIT §8.3); дальше число едет в `broll_rect` через план.
        item.asset_aspect = asset_aspect(asset.path)
        if item.asset_aspect:
            log.info("вставка %r: пропорция %.3f", item.query, item.asset_aspect)
        resolved.append(item)

    edl.broll = resolved
    log.info("b-roll: нарисовано %d из %d запланированных", len(resolved), len(items))
    return len(resolved)


if __name__ == "__main__":  # проверка генерации одной картинки на реальном ключе
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
                    "model": imagegen.image_model(default_settings),
                    "refs": len(refs),
                    "asset": str(got.path) if got else None,
                    "aspect": asset_aspect(got.path) if got else None,
                },
                ensure_ascii=False,
            )
        )

    asyncio.run(_main())
