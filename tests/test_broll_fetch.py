"""Подача и добыча картинок для b-roll: что вставляем, как и какой файл берём.

Сеть подменяется httpx.MockTransport — проверяется наш выбор, а не чужой сервер.
"""

from __future__ import annotations

import httpx
import pytest

from src.core.config import Settings
from src.core.edl import EDL, Broll, BrollAnim, Cut, Word
from src.core.proc import ffmpeg
from src.pipeline.broll import (
    FADE_MS,
    FULLSCREEN_SECONDS,
    HOLD_MS,
    HOLD_RATIO,
    INSET_SECONDS,
    MIN_ANIM_FRAMES,
    PEXELS_RENDITION,
    REST_SECONDS,
    TRAVEL_MS,
    asset_aspect,
    cache_key,
    fetch_asset,
    fetch_pexels,
    is_abstract_query,
    sanitize_items,
)

pytestmark = pytest.mark.anyio


def cfg_without_env(**over) -> Settings:
    return Settings(_env_file=None, **over)


def photo(**srcs) -> dict:
    return {"photos": [{"src": srcs}]}


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


# ------------------------------------------------------------------ смысл и подача


def edl_for_broll(count: int = 12) -> EDL:
    """Ролик на 24 с из одного куска: длинный, чтобы потолок вставок не мешал."""
    words = [Word(idx=i, text=f"сл{i}", start=i * 2.0, end=i * 2.0 + 1.0) for i in range(count)]
    return EDL(
        source_duration=count * 2.0,
        words=words,
        cuts=[Cut(src_in=0.0, src_out=count * 2.0, reason="keep")],
        captions=[],
        preset="glass",
    )


def raw(**kw) -> dict:
    item = {"word_idx": 0, "query": "vintage film camera", "anim": "pop_in"}
    item.update(kw)
    return item


@pytest.mark.parametrize(
    "query",
    ["success", "motivation and inspiration", "the future of the mind", "ideas", "   "],
)
def test_abstract_queries_are_dropped(query):
    """Сток на «success» отдаёт рукопожатие в офисе — это шум, а не иллюстрация."""
    assert is_abstract_query(query)


@pytest.mark.parametrize(
    "query",
    ["vintage film camera", "tokyo street at night", "2024 olympic stadium", "coffee"],
)
def test_concrete_queries_survive(query):
    assert not is_abstract_query(query)


def test_abstract_item_never_reaches_the_edl():
    kept = sanitize_items(edl_for_broll(), [raw(query="success")])
    assert kept == []


def test_default_delivery_is_fullscreen():
    """Карточка в углу поверх говорящей головы читается как новостная плашка."""
    kept = sanitize_items(edl_for_broll(), [raw(anchor="top_right")])
    assert [i.anchor for i in kept] == ["fullscreen"]


def test_corner_survives_only_as_the_end_of_a_flight():
    kept = sanitize_items(
        edl_for_broll(), [raw(anim="fullscreen_to_inset", anchor="bottom_left")]
    )
    assert [i.anchor for i in kept] == ["bottom_left"]


def test_center_inset_goes_to_a_corner():
    """В центре кадра лицо, а свободной полосы между лицом и субтитром нет."""
    kept = sanitize_items(edl_for_broll(), [raw(anim="fullscreen_to_inset", anchor="center")])
    assert [i.anchor for i in kept] == ["top_right"]


def test_ken_burns_direction_alternates_and_never_random():
    """М3: направление наезда — данные из EDL, а не бросок в рендерере.

    Две вставки подряд, едущие в одну сторону, читаются как один длинный проезд,
    поэтому знак чередуется. А главное — он детерминированный: `Math.random` в
    рендерере сломал бы инвариант 4, и перерендер того же листа давал бы другое
    видео.
    """
    kept = sanitize_items(edl_for_broll(24), [raw(), raw(word_idx=8)])
    assert [i.kb_dir for i in kept] == [-1, 1]
    again = sanitize_items(edl_for_broll(24), [raw(), raw(word_idx=8)])
    assert [i.kb_dir for i in again] == [-1, 1]


def test_fullscreen_duration_is_readable_but_not_long():
    short, long = sanitize_items(
        edl_for_broll(),
        [raw(duration=0.8), raw(word_idx=8, duration=3.0)],
    )
    assert short.duration == FULLSCREEN_SECONDS[0]
    assert long.duration == FULLSCREEN_SECONDS[1]


def test_flight_to_the_corner_gets_time_to_land():
    """При 0.8 с картинка прилетала в угол уже растворяясь (AUDIT §8.3)."""
    (item,) = sanitize_items(
        edl_for_broll(), [raw(anim="fullscreen_to_inset", duration=0.8)]
    )
    assert item.duration == INSET_SECONDS[0]


@pytest.mark.parametrize("fps", [24, 25, 30, 50, 60])
def test_landed_card_stands_still_long_enough_to_be_seen(fps):
    """Минимум считается из констант анимации, а не назначается числом.

    Прежние 1.6 с закрывали дефект формально: карточка стояла в углу 5-7 кадров
    (0.21 с при 24 fps) и сразу начинала таять. Проверяем ту же арифметику, что
    в Broll.tsx: удержание, перелёт, затухание — и что после посадки остаётся
    настоящий покой.
    """
    def frames(ms: float) -> int:
        return max(MIN_ANIM_FRAMES, round(ms / 1000 * fps))

    span = round(INSET_SECONDS[0] * fps)
    hold = min(frames(HOLD_MS), round(span * HOLD_RATIO))
    arrive = hold + frames(TRAVEL_MS)
    fade_out_starts = span - frames(FADE_MS)
    assert (fade_out_starts - arrive) / fps >= REST_SECONDS - 1e-9


# ------------------------------------------------------------------ пропорция файла


def test_asset_aspect_reads_the_real_file(tmp_path):
    """Пропорцию карточки задаёт источник, а не догадка: файл меряется."""
    path = tmp_path / "p.jpg"
    ffmpeg(["-f", "lavfi", "-i", "color=c=gray:s=800x1200", "-frames:v", "1", str(path)])
    assert asset_aspect(path) == pytest.approx(800 / 1200)


def test_asset_aspect_of_garbage_is_none(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not a jpeg")
    assert asset_aspect(path) is None


# ------------------------------------------------------------------ выбор нарезки


async def test_prefers_portrait_rendition():
    """`large` у Pexels зажат по высоте 650 — для кадра 1080 это мыло."""
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.pexels.com":
            return httpx.Response(200, json=photo(
                large="https://img/large.jpg",
                large2x="https://img/large2x.jpg",
                portrait="https://img/portrait.jpg",
                original="https://img/original.jpg",
            ))
        asked.append(str(request.url))
        return httpx.Response(200, content=b"\xff\xd8jpeg")

    async with client_for(handler) as c:
        data = await fetch_pexels(c, "cat", "key")

    assert data == b"\xff\xd8jpeg"
    assert asked == ["https://img/portrait.jpg"]


async def test_falls_back_when_portrait_missing():
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.pexels.com":
            return httpx.Response(200, json=photo(large="https://img/large.jpg",
                                                  large2x="https://img/large2x.jpg"))
        asked.append(str(request.url))
        return httpx.Response(200, content=b"jpeg")

    async with client_for(handler) as c:
        await fetch_pexels(c, "cat", "key")

    assert asked == ["https://img/large2x.jpg"]


async def test_no_renditions_at_all_is_not_a_crash():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"photos": [{"src": {}}]})

    async with client_for(handler) as c:
        assert await fetch_pexels(c, "cat", "key") is None


async def test_search_asks_for_portrait_photos():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        assert request.headers["Authorization"] == "key"
        return httpx.Response(200, json={"photos": []})

    async with client_for(handler) as c:
        assert await fetch_pexels(c, "vintage camera", "key") is None

    assert seen["orientation"] == "portrait"
    assert seen["query"] == "vintage camera"


async def test_http_error_returns_none_not_raises():
    """Картинка необязательна: ролик важнее вставки (§5 M7)."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "нет"})

    async with client_for(handler) as c:
        assert await fetch_pexels(c, "cat", "плохой ключ") is None


# ------------------------------------------------------------------ кеш


def test_cache_key_depends_on_rendition():
    """Сменили нарезку — старые файлы другого разрешения не должны подойти."""
    import src.pipeline.broll as broll

    before = cache_key("cat", None, "768*1024")
    original = broll.PEXELS_RENDITION
    try:
        broll.PEXELS_RENDITION = "large"
        after = cache_key("cat", None, "768*1024")
    finally:
        broll.PEXELS_RENDITION = original
    assert before != after
    assert PEXELS_RENDITION in ("portrait", "large2x", "original", "large")


def test_cache_key_is_stable_and_case_insensitive():
    assert cache_key("Vintage Camera", None, "s") == cache_key("vintage camera", None, "s")
    assert cache_key("a", None, "s") != cache_key("b", None, "s")


async def test_second_request_comes_from_cache(tmp_path):
    """Один и тот же запрос не должен стоить дважды."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.host == "api.pexels.com":
            return httpx.Response(200, json=photo(portrait="https://img/p.jpg"))
        return httpx.Response(200, content=b"x" * 4096)

    item = Broll(word_idx=0, duration=1.5, query="cat", anim=BrollAnim.POP_IN, anchor="center")
    cfg = cfg_without_env(pexels_api_key="key")

    async with client_for(handler) as c:
        first = await fetch_asset(c, item, tmp_path, cfg)
        assert first is not None and first.exists()
        second = await fetch_asset(c, item, tmp_path, cfg)

    assert second == first
    assert len(calls) == 2  # поиск + скачивание, и ни одного вызова на второй заход


async def test_no_pexels_key_and_no_wavespeed_gives_none(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover — не должен вызваться
        raise AssertionError(f"сети быть не должно: {request.url}")

    item = Broll(word_idx=0, duration=1.5, query="cat", anim=BrollAnim.POP_IN, anchor="center")
    cfg = cfg_without_env(pexels_api_key="", wavespeed_api_key="")

    async with client_for(handler) as c:
        assert await fetch_asset(c, item, tmp_path, cfg) is None
