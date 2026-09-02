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
from pathlib import Path
from typing import Any

import httpx

from src.core.config import Settings, settings as default_settings
from src.core.edl import EDL, Broll, BrollAnim, Timeline
from src.core.proc import ffmpeg
from src.llm.client import LLMBadOutput, LLMClient, LLMUnavailable
from src.llm.schemas import BROLL_SCHEMA
from src.pipeline.director import words_table
from src.pipeline.probe import MediaInfo, probe

log = logging.getLogger(__name__)

BROLL_WINDOW = 8.0
FRAMES_FOR_VISION = 4
PEXELS_URL = "https://api.pexels.com/v1/search"
WAVESPEED_URL = "https://api.wavespeed.ai/api/v3"

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
    return max(0, math.ceil(edl.out_duration / BROLL_WINDOW))


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
        try:
            item = Broll(
                word_idx=idx,
                duration=min(3.0, max(0.8, float(raw.get("duration") or 1.6))),
                query=query,
                prompt=(raw.get("prompt") or None),
                anim=BrollAnim(raw.get("anim") or "pop_in"),
                anchor=raw.get("anchor") if raw.get("anchor") in
                {"top_left", "top_right", "bottom_left", "bottom_right", "center", "fullscreen"}
                else "top_right",
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
            "plan_broll", variables, BROLL_SCHEMA, "vision", images=images, max_tokens=4000
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
    raw = f"{query}|{prompt or ''}|{size}".lower().strip()
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
        url = photos[0]["src"].get("large") or photos[0]["src"].get("original")
        img = await client.get(url, timeout=60)
        img.raise_for_status()
        return img.content
    except (httpx.HTTPError, KeyError, ValueError) as e:
        log.warning("Pexels не отдал картинку по %r: %s", query, e)
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


async def fetch_asset(
    client: httpx.AsyncClient, item: Broll, cache_dir: Path, cfg: Settings
) -> Path | None:
    """Сток → генерация → None. Результат кешируется по хешу запроса."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(item.query, item.prompt, cfg.wavespeed_size)
    cached = cache_dir / f"{key}.jpg"
    if cached.exists() and cached.stat().st_size > 1024:
        log.debug("b-roll из кеша: %s", item.query)
        return cached

    data: bytes | None = None
    if cfg.pexels_api_key:
        data = await fetch_pexels(client, item.query, cfg.pexels_api_key)
    if data is None:
        prompt = item.prompt or item.query
        data = await generate_wavespeed(client, prompt, cfg)
    if not data:
        return None

    cached.write_bytes(data)
    return cached


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
        item.asset = str(asset)
        resolved.append(item)

    edl.broll = resolved
    log.info("b-roll: получено %d картинок из %d запланированных", len(resolved), len(items))
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
