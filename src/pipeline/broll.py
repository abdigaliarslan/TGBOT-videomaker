"""M7: b-roll — план вставок от vision-модели и добыча картинок.

Порядок источников: сток (Pexels) дешевле генерации, генерация (WaveSpeed) —
запасной путь, когда стока нет. Результат кешируется по хешу запроса: один и
тот же «vintage film camera» не должен стоить дважды.

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
from typing import Any

import httpx

from src.core.config import Settings, settings as default_settings
from src.core.edl import EDL, Broll, BrollAnim, Timeline
from src.core.proc import ProcError, ffmpeg, run
from src.llm.client import LLMBadOutput, LLMClient, LLMUnavailable
from src.llm.schemas import BROLL_SCHEMA
from src.pipeline.director import words_table
from src.pipeline.probe import MediaInfo, probe

log = logging.getLogger(__name__)

from src.core.edl import BROLL_MAX, BROLL_WINDOW  # потолок один на весь проект
FRAMES_FOR_VISION = 4
PEXELS_URL = "https://api.pexels.com/v1/search"
WAVESPEED_URL = "https://api.wavespeed.ai/api/v3"

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
# просто не бывает: крупный план, поверхность без брендов.
NO_TEXT_SUFFIX = (
    "photographic, natural light, shallow depth of field, clean unbranded surfaces, "
    "close-up angle that avoids any labels, "
    "no text, no letters, no numbers, no words, no captions, no signage, "
    "no watermark, no logo, no brand name, no user interface, no typography"
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
) -> list[Broll]:
    """Vision-вызов: кадры + транскрипт → план вставок."""
    cfg = cfg or default_settings
    cap = _cap(edl)
    if cap <= 0:
        return []

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
        return []

    items = sanitize_items(edl, result.data.get("items") or [])
    log.info("b-roll: модель предложила %d, оставлено %d (потолок %d)",
             len(result.data.get("items") or []), len(items), cap)
    return items


# --------------------------------------------------------------------------- картинки


def cache_key(query: str, prompt: str | None, size: str) -> str:
    # Нарезка входит в ключ намеренно: сменили её — старые файлы другого
    # разрешения перестают подходить, и кеш обязан это заметить сам.
    raw = f"{query}|{prompt or ''}|{size}|{PEXELS_RENDITION}".lower().strip()
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

# Нижняя граница высоты нарезки. Кадр рендерится в 1080x1920, и клип ниже 720
# после вписывания в него — мыло; выше 1080 брать незачем, декодировать дороже,
# а видно то же самое.
VIDEO_MIN_HEIGHT = 720
VIDEO_MAX_HEIGHT = 1080
# Клип обязан быть ДЛИННЕЕ вставки с запасом: ролик, обрезанный ровно в свой
# конец, на последнем кадре показывает чёрное — файл кончился раньше, чем
# композиция. Полсекунды хватает и на округление кадров, и на неточность
# длительности в метаданных стока.
VIDEO_MARGIN_S = 0.5
# Потолок веса файла. Не оптимизация, а предохранитель: за ним стоит время
# скачивания в общем таймауте этапа и место в кеше.
VIDEO_MAX_BYTES = 32 * 1024 * 1024


def pick_video_file(files: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Из нарезок Pexels — самая маленькая, которая ещё не мыло.

    Не самая большая: 4K-исходник декодируется поверх основного видео на каждом
    кадре рендера, а в кадре 1080 шириной от него не остаётся ничего.
    """
    ok = [
        f
        for f in files
        if f.get("link")
        and str(f.get("file_type") or "").endswith("mp4")
        and VIDEO_MIN_HEIGHT <= int(f.get("height") or 0) <= VIDEO_MAX_HEIGHT
    ]
    if not ok:
        return None
    return min(ok, key=lambda f: int(f["height"]))


def video_start(duration: float, need: float) -> float:
    """С какой секунды играть клип: серединой окна, а не с начала.

    У стоковых видео первые кадры самые статичные — оператор ещё не начал
    движение. Берём кусок из середины: он же и самый устойчивый по экспозиции.
    """
    slack = duration - need
    return max(0.0, slack / 2.0)


async def fetch_pexels_video(
    client: httpx.AsyncClient, query: str, api_key: str, need: float
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

    for v in videos:
        duration = float(v.get("duration") or 0)
        if duration < need + VIDEO_MARGIN_S:
            continue
        picked = pick_video_file(v.get("video_files") or [])
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


async def generate_wavespeed(
    client: httpx.AsyncClient, prompt: str, cfg: Settings
) -> bytes | None:
    """Диффузия как фолбэк, когда стока нет."""
    if not cfg.wavespeed_api_key:
        return None
    headers = {"Authorization": f"Bearer {cfg.wavespeed_api_key}", "Content-Type": "application/json"}
    body = {
        "prompt": f"{prompt}. {NO_TEXT_SUFFIX}",
        "size": cfg.wavespeed_size,
        "enable_base64_output": False,
    }
    try:
        r = await client.post(
            f"{WAVESPEED_URL}/{cfg.wavespeed_model}", json=body, headers=headers, timeout=60
        )
        r.raise_for_status()
        data = r.json().get("data") or {}
        poll_url = (data.get("urls") or {}).get("get")
        job_id = data.get("id")
        if not poll_url and job_id:
            poll_url = f"{WAVESPEED_URL}/predictions/{job_id}/result"
        if not poll_url:
            log.warning("WaveSpeed не вернул ссылку на результат: %s", r.text[:200])
            return None

        for _ in range(60):  # до ~60 с
            await asyncio.sleep(1.0)
            got = await client.get(poll_url, headers=headers, timeout=30)
            got.raise_for_status()
            payload = got.json().get("data") or {}
            status = payload.get("status")
            if status in ("completed", "succeeded"):
                outputs = payload.get("outputs") or []
                if not outputs:
                    return None
                img = await client.get(outputs[0], timeout=60)
                img.raise_for_status()
                return img.content
            if status in ("failed", "error"):
                log.warning("WaveSpeed отказал: %s", payload.get("error"))
                return None
        log.warning("WaveSpeed не успел за отведённое время")
        return None
    except (httpx.HTTPError, ValueError) as e:
        log.warning("WaveSpeed не сгенерировал %r: %s", prompt[:40], e)
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


async def fetch_asset(
    client: httpx.AsyncClient, item: Broll, cache_dir: Path, cfg: Settings
) -> Asset | None:
    """Видео → фото → генерация → None. Результат кешируется по хешу запроса.

    Порядок именно такой (М4): клип читается как монтаж, фото — как наклейка.
    Но видео есть не на всякий запрос, поэтому откат на фото обязан быть молчаливым
    — для вызывающего кода разницы нет, он получает файл.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(item.query, item.prompt, cfg.wavespeed_size)

    if cfg.broll_video and cfg.pexels_api_key:
        hit = _cached_video(cache_dir, key)
        if hit is not None:
            log.debug("b-roll видео из кеша: %s", item.query)
            return hit
        got = await fetch_pexels_video(client, item.query, cfg.pexels_api_key, item.duration)
        if got is not None:
            data, meta = got
            start = round(float(meta["start"]), 2)
            path = cache_dir / f"{key}{_VIDEO_START_SEP}{start}.mp4"
            path.write_bytes(data)
            return Asset(path, "video", start)

    cached = cache_dir / f"{key}.jpg"
    if cached.exists() and cached.stat().st_size > 1024:
        log.debug("b-roll из кеша: %s", item.query)
        return Asset(cached)

    data_bytes: bytes | None = None
    if cfg.pexels_api_key:
        data_bytes = await fetch_pexels(client, item.query, cfg.pexels_api_key)
    if data_bytes is None:
        prompt = item.prompt or item.query
        data_bytes = await generate_wavespeed(client, prompt, cfg)
    if not data_bytes:
        return None

    cached.write_bytes(data_bytes)
    return Asset(cached)


def asset_aspect(path: str | Path) -> float | None:
    """Пропорция скачанного файла (ширина/высота) или None, если не прочитался.

    Нужна геометрии: карточка наследует пропорцию источника, иначе `cover` режет
    (AUDIT §8.3). Сток отдаёт 2:3, генерация — то, что заказано в
    `wavespeed_size`, и это разные числа, поэтому меряется файл, а не источник.
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
    items = await plan_broll(edl, src, work, llm=llm, info=info, cfg=cfg)
    if not items:
        edl.broll = []
        return 0

    cache_dir = Path(cfg.work_dir) / "broll-cache"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        assets = await asyncio.gather(
            *(fetch_asset(client, item, cache_dir, cfg) for item in items)
        )

    resolved: list[Broll] = []
    for item, asset in zip(items, assets):
        if asset is None:
            log.info("картинка не найдена для %r — вставка отменена", item.query)
            continue
        item.asset = str(asset.path)
        item.kind = asset.kind
        item.asset_start = asset.start
        # Пропорция меряется у ФАЙЛА, а не берётся у источника: сток отдаёт 2:3,
        # генерация — то, что заказано в `wavespeed_size`, а клип приходит своей
        # третьей. Карточка наследует её, иначе `cover` режет источник
        # (AUDIT §8.3); дальше число едет в `broll_rect` через план.
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
        query = sys.argv[1] if len(sys.argv) > 1 else "vintage film camera on a desk"
        item = Broll(word_idx=0, duration=1.5, query=query, anim=BrollAnim.POP_IN, anchor="top_right")
        async with httpx.AsyncClient(follow_redirects=True) as client:
            path = await fetch_asset(client, item, Path(default_settings.work_dir) / "broll-cache", default_settings)
        print(json.dumps({"query": query, "asset": str(path) if path else None}, ensure_ascii=False))

    asyncio.run(_main())
