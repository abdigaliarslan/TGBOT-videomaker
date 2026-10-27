"""Картинка рисуется, а не ищется: клиент генерации, промпт, порядок источников.

Сеть подменяется httpx.MockTransport, модель — двойником провайдера: проверяется
наш выбор и наш запрос, а не чужой сервер.
"""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from src.core.config import Settings
from src.core.edl import EDL, Broll, BrollAnim, Cut, Word
from src.llm import images as imagegen
from src.llm.client import LLMClient
from src.pipeline import broll as broll_mod
from src.pipeline.broll import (
    COMPOSITION_HINT,
    FALLBACK_STYLE_HINT,
    NO_TEXT_SUFFIX,
    ReelStyle,
    _ref_frame,
    build_image_prompt,
    cache_key,
    fetch_asset,
)
from tests.conftest import FakeProvider, result_of

pytestmark = pytest.mark.anyio

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 4000


def cfg_without_env(**over) -> Settings:
    return Settings(_env_file=None, **over)


def gen_cfg(**over) -> Settings:
    """Конфиг для ветки Gemini: она единственная читает референсы.

    Провайдер задаётся явно, хотя по умолчанию в проекте стоит `wavespeed`:
    здесь проверяется именно форма запроса к Gemini и работа с референсами, и
    молчаливое переключение умолчания не должно увести тест в другой код.
    """
    base = {
        "gemini_api_key": "g",
        "image_provider": "gemini",
        "broll_verify": False,
    }
    base.update(over)
    return cfg_without_env(**base)


def ws_cfg(**over) -> Settings:
    """Боевой конфиг: рисует WaveSpeed."""
    base = {
        "wavespeed_api_key": "w",
        "image_provider": "wavespeed",
        "broll_verify": False,
    }
    base.update(over)
    return cfg_without_env(**base)


def wavespeed_ok(data: bytes = PNG):
    """Ответ WaveSpeed с картинкой и журнал запросов к нему."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen.append({"url": url, "body": json.loads(request.content) if request.content else {}})
        if url.endswith("/z-image/turbo"):
            return httpx.Response(200, json={"data": {"id": "1", "urls": {"get": "https://ws/r"}}})
        if url == "https://ws/r":
            return httpx.Response(
                200, json={"data": {"status": "completed", "outputs": ["https://ws/i.jpg"]}}
            )
        return httpx.Response(200, content=data)

    return handler, seen


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


def gemini_ok(data: bytes = PNG):
    """Ответ Gemini с картинкой и журнал запросов к нему."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": json.loads(request.content)})
        return httpx.Response(
            200,
            json={"output_image": {"data": base64.standard_b64encode(data).decode("ascii")}},
        )

    return handler, seen


def item_of(**kw) -> Broll:
    fields = {
        "word_idx": 0,
        "duration": 1.5,
        "query": "vintage film camera",
        "anim": BrollAnim.POP_IN,
        "anchor": "fullscreen",
    }
    fields.update(kw)
    return Broll(**fields)


# ------------------------------------------------------------------ клиент Gemini


async def test_request_carries_model_prompt_and_vertical_aspect():
    handler, seen = gemini_ok()
    cfg = gen_cfg()
    async with client_for(handler) as c:
        got = await imagegen.generate_gemini(c, "a camera", cfg=cfg)

    assert got == PNG
    body = seen[0]["body"]
    assert body["model"] == "gemini-3.1-flash-image"
    assert body["input"][0] == {"type": "text", "text": "a camera"}
    # Вертикаль просится у модели, а не дорезается потом: картинка, рождённая
    # 9:16, ложится в кадр без размытой подложки.
    assert body["response_format"]["aspect_ratio"] == "9:16"
    assert seen[0]["url"].endswith("/interactions")


async def test_reference_frames_are_attached_as_images():
    handler, seen = gemini_ok()
    async with client_for(handler) as c:
        await imagegen.generate_gemini(c, "a camera", refs=[b"frame-one", b"frame-two"], cfg=gen_cfg())

    parts = seen[0]["body"]["input"]
    assert parts[0]["type"] == "text"
    images = [p for p in parts if p["type"] == "image"]
    assert [base64.b64decode(p["data"]) for p in images] == [b"frame-one", b"frame-two"]


async def test_image_is_found_wherever_the_answer_puts_it():
    """Форму ответа задаёт Google, и она уже менялась — ищем данные, а не путь."""
    encoded = base64.standard_b64encode(PNG).decode("ascii")
    shapes = [
        {"output_image": {"data": encoded}},
        {"candidates": [{"content": {"parts": [{"inlineData": {"data": encoded}}]}}]},
        {"data": [{"b64_json": encoded}]},
    ]
    for shape in shapes:
        async with client_for(lambda r, s=shape: httpx.Response(200, json=s)) as c:
            assert await imagegen.generate_gemini(c, "x", cfg=gen_cfg()) == PNG


async def test_failure_is_none_and_never_raises():
    """Вставка необязательна: ронять из-за неё ролик нельзя (§3)."""
    cases = [
        lambda r: httpx.Response(500, text="boom"),
        lambda r: httpx.Response(200, json={"candidates": []}),
        lambda r: httpx.Response(200, text="not json"),
    ]
    for handler in cases:
        async with client_for(handler) as c:
            assert await imagegen.generate_gemini(c, "x", cfg=gen_cfg()) is None


async def test_no_key_means_no_generation():
    cfg = cfg_without_env(gemini_api_key="")
    assert imagegen.ready(cfg) is False
    async with client_for(lambda r: httpx.Response(200, json={})) as c:
        assert await imagegen.generate_gemini(c, "x", cfg=cfg) is None


async def test_wavespeed_is_chosen_by_provider_not_by_luck():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "wavespeed" in str(request.url) and request.method == "POST":
            return httpx.Response(200, json={"data": {"id": "1", "urls": {"get": "https://ws/r"}}})
        if str(request.url) == "https://ws/r":
            return httpx.Response(200, json={"data": {"status": "completed", "outputs": ["https://ws/i.jpg"]}})
        return httpx.Response(200, content=PNG)

    cfg = cfg_without_env(image_provider="wavespeed", wavespeed_api_key="w")
    async with client_for(handler) as c:
        got = await imagegen.generate(c, "a camera", refs=[b"ref"], cfg=cfg)

    assert got == PNG
    assert all("generativelanguage" not in url for url in seen)


def fal_cfg(**over) -> Settings:
    base = {"fal_api_key": "f", "image_provider": "fal", "fal_size": "864*1536"}
    base.update(over)
    return cfg_without_env(**base)


def fal_ok(*, nsfw: bool = False, status: int = 200):
    """Ответ fal со ссылкой на картинку и журнал запросов."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "fal.run":
            if status != 200:
                return httpx.Response(status, text="Exhausted balance")
            return httpx.Response(
                200,
                json={"images": [{"url": "https://v3.fal.media/i.jpg"}], "has_nsfw_concepts": [nsfw]},
            )
        return httpx.Response(200, content=PNG)

    return handler, seen


async def test_fal_request_carries_key_model_and_vertical_size():
    handler, seen = fal_ok()
    async with client_for(handler) as c:
        got = await imagegen.generate(c, "a camera", refs=[b"ref"], cfg=fal_cfg())

    assert got == PNG
    req = seen[0]
    assert str(req.url) == "https://fal.run/fal-ai/z-image/turbo"
    assert req.headers["Authorization"] == "Key f"
    body = json.loads(req.content)
    assert body["prompt"] == "a camera"
    assert body["image_size"] == {"width": 864, "height": 1536}
    assert body["num_images"] == 1
    assert imagegen.ready(fal_cfg()) and not imagegen.uses_refs(fal_cfg())


async def test_fal_refusal_is_none_and_paid_only_once():
    """Чёрный кадр фильтра и пустой баланс — не картинка; повтор стоил бы денег."""
    for kw in ({"nsfw": True}, {"status": 403}):
        handler, seen = fal_ok(**kw)
        async with client_for(handler) as c:
            assert await imagegen.generate_fal(c, "x", cfg=fal_cfg()) is None
        assert len(seen) == 1


async def test_fal_without_key_or_with_a_broken_size_calls_nothing():
    for cfg in (fal_cfg(fal_api_key=""), fal_cfg(fal_size="vertical")):
        handler, seen = fal_ok()
        async with client_for(handler) as c:
            assert await imagegen.generate_fal(c, "x", cfg=cfg) is None
        assert seen == []


def test_cache_key_follows_the_fal_size():
    assert imagegen.size_tag(fal_cfg()) == "864*1536"
    assert imagegen.size_tag(fal_cfg(fal_size="720*1280")) == "720*1280"


# ------------------------------------------------------------------ промпт и стиль


def test_prompt_puts_subject_first_and_the_text_ban_last():
    """Предмет обязан выжить при обрезке промпта, запрет надписей — работать."""
    style = ReelStyle(palette="warm amber", light="soft key from the left", mood="cosy interior")
    prompt = build_image_prompt("a single vintage camera", style)

    assert prompt.startswith("a single vintage camera")
    assert prompt.endswith(NO_TEXT_SUFFIX)
    assert "warm amber" in prompt and "cosy interior" in prompt
    assert COMPOSITION_HINT in prompt


def test_prompt_without_style_still_says_something():
    assert FALLBACK_STYLE_HINT in build_image_prompt("a camera", None)
    assert FALLBACK_STYLE_HINT in build_image_prompt("a camera", ReelStyle())


def test_fix_from_the_check_goes_into_the_retry():
    prompt = build_image_prompt("a camera", ReelStyle(), fix="remove the packaging")
    assert "remove the packaging" in prompt
    assert prompt.startswith("a camera")


def test_style_survives_a_model_that_answers_badly():
    assert ReelStyle.from_raw(None) == ReelStyle()
    assert ReelStyle.from_raw("не объект") == ReelStyle()
    assert ReelStyle.from_raw({"palette": 7, "mood": "  dark  "}) == ReelStyle(mood="dark")
    assert len(ReelStyle.from_raw({"palette": "x" * 400}).palette) == 120


def test_ref_frame_is_the_one_nearest_the_insert():
    """Свет по ролику меняется: вставка берёт кадр из своего куска, а не первый."""
    frames = [b"a", b"b", b"c", b"d"]
    assert _ref_frame(frames, 0.0, 20.0) == [b"a"]
    assert _ref_frame(frames, 9.0, 20.0) == [b"b"]
    assert _ref_frame(frames, 12.0, 20.0) == [b"c"]
    assert _ref_frame(frames, 20.0, 20.0) == [b"d"]
    assert _ref_frame([], 5.0, 20.0) == []


def test_cache_key_separates_models_and_styles():
    """Та же картинка другой моделью или под другую палитру — ДРУГАЯ картинка."""
    base = cache_key("cat", None, "1K", model="m1", style="warm")
    assert base != cache_key("cat", None, "1K", model="m2", style="warm")
    assert base != cache_key("cat", None, "1K", model="m1", style="cold")


# ------------------------------------------------------------------ генерация и кеш


async def test_the_insert_is_drawn(tmp_path):
    handler, seen = gemini_ok()
    async with client_for(handler) as c:
        asset = await fetch_asset(c, item_of(), tmp_path, gen_cfg())

    assert asset is not None
    assert asset.path.read_bytes() == PNG
    assert len(seen) == 1


async def test_generated_image_is_cached_and_not_paid_for_twice(tmp_path):
    handler, seen = gemini_ok()
    cfg = gen_cfg()
    async with client_for(handler) as c:
        first = await fetch_asset(c, item_of(), tmp_path, cfg)
        second = await fetch_asset(c, item_of(), tmp_path, cfg)

    assert first is not None and second is not None
    assert first.path == second.path
    assert len(seen) == 1


# ------------------------------------------------------------------ приёмка


def llm_saying(*verdicts) -> LLMClient:
    """Клиент с подставленным провайдером: ключ ему не нужен, ответы заготовлены."""
    return LLMClient(cfg=cfg_without_env(), provider=FakeProvider([result_of(v) for v in verdicts]))


async def test_a_picture_with_letters_is_redrawn(tmp_path):
    handler, seen = gemini_ok()
    cfg = gen_cfg(broll_verify=True)
    llm = llm_saying(
        {"shows_subject": True, "has_text": True, "fix": "remove all labels"},
        {"shows_subject": True, "has_text": False, "fix": None},
    )
    async with client_for(handler) as c:
        asset = await fetch_asset(c, item_of(), tmp_path, cfg, llm=llm)

    assert asset is not None
    assert len(seen) == 2
    assert "remove all labels" in seen[1]["body"]["input"][0]["text"]


async def test_twice_rejected_cancels_the_insert_instead_of_looping(tmp_path):
    """Вторая неудачная приёмка — вставки не будет, третьей попытки тоже."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            200, json={"output_image": {"data": base64.standard_b64encode(PNG).decode()}}
        )

    bad = {"shows_subject": False, "has_text": False, "fix": "show the camera"}
    cfg = gen_cfg(broll_verify=True)
    async with client_for(handler) as c:
        asset = await fetch_asset(c, item_of(), tmp_path, cfg, llm=llm_saying(bad, bad))

    assert asset is None
    assert len(calls) == 2
    assert not list(tmp_path.glob("*.jpg")), "отвергнутая картинка не должна лечь в кеш"


async def test_a_broken_check_never_cancels_a_good_picture(tmp_path):
    """Приёмка — предохранитель, а не судья: её собственная осечка не считается."""
    handler, seen = gemini_ok()
    cfg = gen_cfg(broll_verify=True)
    async with client_for(handler) as c:
        asset = await fetch_asset(c, item_of(), tmp_path, cfg, llm=llm_saying("не json"))

    assert asset is not None
    assert len(seen) == 1


# ------------------------------------------------------------------ боевая настройка


async def test_wavespeed_draws_the_insert(tmp_path):
    handler, seen = wavespeed_ok()
    async with client_for(handler) as c:
        asset = await fetch_asset(c, item_of(), tmp_path, ws_cfg())

    assert asset is not None
    assert asset.path.read_bytes() == PNG
    assert "wavespeed" in seen[0]["url"]


async def test_a_failed_drawing_cancels_the_insert(tmp_path):
    """Запасного пути больше нет: не нарисовалось — вставки не будет.

    Ронять на этом ролик нельзя (§3): вставка необязательна, и `None` здесь —
    штатный ответ, который вызывающий код превращает в «вставка отменена».
    """
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(500, text="boom")

    async with client_for(handler) as c:
        assert await fetch_asset(c, item_of(), tmp_path, ws_cfg()) is None

    assert calls, "генератор обязан был получить запрос"


def test_only_gemini_reads_references():
    """От этого зависит, рисовать вставки по очереди или разом."""
    assert imagegen.uses_refs(gen_cfg()) is True
    assert imagegen.uses_refs(ws_cfg()) is False


async def _fetch_concurrency(cfg: Settings, tmp_path, monkeypatch) -> int:
    """Сколько вставок оказывались в работе ОДНОВРЕМЕННО за один прогон."""
    words = [Word(idx=i, text=f"сл{i}", start=i * 2.0, end=i * 2.0 + 1.0) for i in range(14)]
    edl = EDL(
        source_duration=28.0,
        words=words,
        cuts=[Cut(src_in=0.0, src_out=28.0, reason="keep")],
        captions=[],
        preset="glass",
    )
    items = [item_of(word_idx=0), item_of(word_idx=12)]
    plan = broll_mod.BrollPlan(items, broll_mod.ReelStyle(mood="dark"), [b"f1", b"f2"])

    async def fake_plan(*a, **kw):
        return plan

    live = 0
    peak = 0

    async def fake_fetch(client, item, cache_dir, cfg_, **kw):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        path = tmp_path / f"{item.word_idx}.jpg"
        path.write_bytes(PNG)
        return broll_mod.Asset(path)

    monkeypatch.setattr(broll_mod, "plan_broll", fake_plan)
    monkeypatch.setattr(broll_mod, "fetch_asset", fake_fetch)
    monkeypatch.setattr(broll_mod, "asset_aspect", lambda p: 0.5625)
    used = await broll_mod.plan_and_fetch(edl, "in.mp4", tmp_path, llm=None, cfg=cfg)
    assert used == 2
    return peak


async def test_wavespeed_draws_both_inserts_at_once(tmp_path, monkeypatch):
    """Референсы WaveSpeed выбрасывает, значит очередь стоила бы ожидания зря."""
    assert await _fetch_concurrency(ws_cfg(), tmp_path, monkeypatch) == 2


async def test_gemini_draws_in_turn_so_the_second_sees_the_first(tmp_path, monkeypatch):
    """Очередь остаётся там, где она за что-то платит: связность комплекта."""
    assert await _fetch_concurrency(gen_cfg(), tmp_path, monkeypatch) == 1
