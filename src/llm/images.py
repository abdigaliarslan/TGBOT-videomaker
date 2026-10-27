"""Генерация картинок-вставок: промпт и кадры видео → JPEG/PNG.

Почему это отдельный модуль, а не метод провайдера из `providers.py`: тот
интерфейс возвращает ТЕКСТ по схеме, а здесь на выходе байты картинки, и
общего у двух задач ровно ничего, кроме ключа.

Два бэкенда за одной функцией:

- `gemini` — умеет принимать РЕФЕРЕНСНЫЕ картинки. Ради этого он и выбран:
  вставка, сгенерированная с кадром ролика перед глазами, попадает в его
  палитру и свет, а не живёт отдельной жизнью (см. `pipeline/broll.py`).
- `wavespeed` — прежний путь, дешевле и без референсов. POST, потом опрос
  статуса; остаётся запасным.
- `fal` — та же модель, что у WaveSpeed, без референсов, но синхронно: один
  запрос, ответ за секунды, опрашивать нечего.

Ошибка, таймаут и пустой ответ — это `None`, а не исключение: вставка
необязательна, и ронять из-за неё ролик нельзя (§3, фолбэк).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Sequence

import httpx

from src.core.config import Settings, settings as default_settings

log = logging.getLogger(__name__)

WAVESPEED_URL = "https://api.wavespeed.ai/api/v3"
FAL_URL = "https://fal.run"

# Модель по умолчанию на провайдера. Имя модели — деталь реализации, как и у
# текстовых вызовов (§4): вызывающий код знает только «сгенерируй картинку».
PROVIDER_IMAGE_MODELS: dict[str, str] = {
    "gemini": "gemini-3.1-flash-image",
    "wavespeed": "wavespeed-ai/z-image/turbo",
    "fal": "fal-ai/z-image/turbo",
}

# Сколько ждём картинку. Gemini и fal отвечают одним запросом за единицы
# секунд, WaveSpeed отдаёт задание и просит опрашивать.
GEMINI_TIMEOUT_S = 120.0
FAL_TIMEOUT_S = 90.0
WAVESPEED_POLL_S = 60


def image_model(cfg: Settings) -> str:
    return cfg.image_model or PROVIDER_IMAGE_MODELS.get(cfg.image_provider, "")


def image_key(cfg: Settings) -> str:
    return {
        "gemini": cfg.gemini_api_key,
        "fal": cfg.fal_api_key,
    }.get(cfg.image_provider, cfg.wavespeed_api_key)


def size_tag(cfg: Settings) -> str:
    """Размер, под которым картинка лежит в кеше: сменили его — кеш не подходит."""
    return {
        "gemini": cfg.image_size,
        "fal": cfg.fal_size,
    }.get(cfg.image_provider, cfg.wavespeed_size)


def ready(cfg: Settings | None = None) -> bool:
    """Есть ли чем генерировать. Нет ключа — молча работаем как раньше, стоком."""
    cfg = cfg or default_settings
    return bool(image_key(cfg) and image_model(cfg))


def uses_refs(cfg: Settings | None = None) -> bool:
    """Читает ли выбранный провайдер референсные картинки.

    Вопрос не праздный: на этом держится решение, рисовать вставки по очереди
    или разом. Очередь нужна только затем, чтобы вторая картинка ролика видела
    первую; провайдеру, который референсы выбрасывает, она стоит лишнего
    ожидания и не даёт ничего.
    """
    cfg = cfg or default_settings
    return cfg.image_provider == "gemini"


async def generate_gemini(
    client: httpx.AsyncClient,
    prompt: str,
    *,
    refs: Sequence[bytes] = (),
    aspect: str = "9:16",
    cfg: Settings,
) -> bytes | None:
    """Один запрос, один ответ: картинка приходит base64 внутри JSON.

    `refs` — кадры ролика и/или уже сгенерированные вставки. Модель смотрит на
    них как на образец цвета и света; что именно с них брать, а что не брать,
    сказано словами в самом промпте — API про это ничего не знает.
    """
    key = cfg.gemini_api_key
    model = image_model(cfg)
    if not key or not model:
        return None

    inputs: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for ref in refs:
        inputs.append(
            {
                "type": "image",
                "mime_type": "image/jpeg",
                "data": base64.standard_b64encode(ref).decode("ascii"),
            }
        )
    body = {
        "model": model,
        "input": inputs,
        # Просим сразу вертикаль: картинка, рождённая 9:16, ложится в кадр
        # целиком, и под неё не нужна размытая подложка (см. Broll.tsx).
        "response_format": {
            "type": "image",
            "aspect_ratio": aspect,
            "image_size": cfg.image_size,
        },
    }
    url = f"{cfg.gemini_base_url.rstrip('/')}/interactions"
    try:
        r = await client.post(
            url,
            json=body,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            timeout=GEMINI_TIMEOUT_S,
        )
        r.raise_for_status()
        data = _first_image(r.json())
        if not data:
            log.warning("Gemini не вернул картинку: %s", r.text[:200])
            return None
        return base64.b64decode(data)
    except (httpx.HTTPError, ValueError, TypeError) as e:
        log.warning("Gemini не сгенерировал %r: %s", prompt[:60], e)
        return None


def _first_image(payload: object) -> str | None:
    """Base64 картинки где бы она ни лежала в ответе.

    Форму ответа задаёт не наш код, и она у Google уже менялась: картинка
    приходила то `output_image.data`, то частью `candidates[].content.parts[]`
    как `inlineData`. Поэтому ищем первое поле с данными картинки, а не один
    жёсткий путь: так обновление API не ломает генерацию молча.
    """
    stack: list[object] = [payload]
    while stack:
        node = stack.pop(0)
        if isinstance(node, dict):
            for name in ("output_image", "outputImage", "inline_data", "inlineData", "image"):
                child = node.get(name)
                if isinstance(child, dict) and isinstance(child.get("data"), str):
                    return child["data"]
            if isinstance(node.get("b64_json"), str):
                return node["b64_json"]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


async def generate_wavespeed(
    client: httpx.AsyncClient, prompt: str, *, cfg: Settings
) -> bytes | None:
    """Диффузия без референсов: POST задания и опрос результата."""
    if not cfg.wavespeed_api_key:
        return None
    model = cfg.image_model or cfg.wavespeed_model
    headers = {"Authorization": f"Bearer {cfg.wavespeed_api_key}", "Content-Type": "application/json"}
    body = {"prompt": prompt, "size": cfg.wavespeed_size, "enable_base64_output": False}
    try:
        r = await client.post(f"{WAVESPEED_URL}/{model}", json=body, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json().get("data") or {}
        poll_url = (data.get("urls") or {}).get("get")
        job_id = data.get("id")
        if not poll_url and job_id:
            poll_url = f"{WAVESPEED_URL}/predictions/{job_id}/result"
        if not poll_url:
            log.warning("WaveSpeed не вернул ссылку на результат: %s", r.text[:200])
            return None

        for _ in range(WAVESPEED_POLL_S):
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


async def generate_fal(client: httpx.AsyncClient, prompt: str, *, cfg: Settings) -> bytes | None:
    """Синхронный вызов: картинка готова к моменту ответа, опрос не нужен.

    Сработавший фильтр безопасности запрос не роняет — приходит чёрный кадр с
    флагом `has_nsfw_concepts`, и такой кадр в ролик пускать нельзя.
    """
    if not cfg.fal_api_key:
        return None
    try:
        width, height = (int(v) for v in cfg.fal_size.split("*"))
    except ValueError:
        log.warning("FAL_SIZE=%r не похож на ШИРИНА*ВЫСОТА", cfg.fal_size)
        return None
    body = {
        "prompt": prompt,
        "image_size": {"width": width, "height": height},
        "num_images": 1,
        "output_format": "jpeg",
        "enable_safety_checker": True,
    }
    try:
        r = await client.post(
            f"{FAL_URL}/{image_model(cfg)}",
            json=body,
            headers={"Authorization": f"Key {cfg.fal_api_key}"},
            timeout=FAL_TIMEOUT_S,
        )
        r.raise_for_status()
        payload = r.json()
        if any(payload.get("has_nsfw_concepts") or []):
            log.warning("fal: фильтр безопасности сработал на %r", prompt[:40])
            return None
        images = payload.get("images") or []
        url = images[0].get("url") if images else None
        if not url:
            log.warning("fal не вернул картинку: %s", r.text[:200])
            return None
        img = await client.get(url, timeout=60)
        img.raise_for_status()
        return img.content
    except (httpx.HTTPError, ValueError, AttributeError) as e:
        log.warning("fal не сгенерировал %r: %s", prompt[:40], e)
        return None


async def generate(
    client: httpx.AsyncClient,
    prompt: str,
    *,
    refs: Sequence[bytes] = (),
    aspect: str = "9:16",
    cfg: Settings | None = None,
) -> bytes | None:
    """Единый вход. Провайдер берётся из конфига; референсы понимает не каждый."""
    cfg = cfg or default_settings
    if cfg.image_provider in ("wavespeed", "fal"):
        if refs:
            log.debug("%s не принимает референсы — генерирую по одному промпту", cfg.image_provider)
        if cfg.image_provider == "fal":
            return await generate_fal(client, prompt, cfg=cfg)
        return await generate_wavespeed(client, prompt, cfg=cfg)
    return await generate_gemini(client, prompt, refs=refs, aspect=aspect, cfg=cfg)
