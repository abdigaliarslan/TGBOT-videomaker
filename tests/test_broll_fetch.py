"""Подача картинок для b-roll: что вставляем, как, и какой файл берём.

Генерация проверяется в test_imagegen.py; здесь — смысл, подача, кеш и то, что
листы времён стока по-прежнему читаются.
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
    REST_SECONDS,
    TRAVEL_MS,
    asset_aspect,
    cache_key,
    fetch_asset,
    is_abstract_query,
    plan_and_fetch,
    sanitize_items,
)

pytestmark = pytest.mark.anyio


def cfg_without_env(**over) -> Settings:
    return Settings(_env_file=None, **over)


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
    """По «success» рисуется рукопожатие в офисе — это шум, а не иллюстрация."""
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


def test_default_delivery_is_a_card_over_the_blurred_frame():
    """Как в эталоне: карточка по центру, говорящий размыт за ней, а не подменён.

    Карточка в углу поверх НЕразмытой головы читается как новостная плашка,
    поэтому угол, который просит модель, сюда не доходит.
    """
    kept = sanitize_items(edl_for_broll(), [raw(anchor="top_right")])
    assert [i.anchor for i in kept] == ["card"]


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


# ------------------------------------------------------------------ кеш


def test_cache_key_is_pinned_so_drawn_pictures_are_not_paid_for_again():
    """Ключ кеша не сдвинулся, когда сток сняли.

    В ключ всегда входила соль (бывшая нарезка стока), в том числе у
    нарисованных картинок. Убрать её — значит сменить ключи всего кеша и
    заново заплатить за каждую картинку, которая уже лежит на диске.
    """
    # Значение снято с кода ДО снятия стока (git show HEAD:src/pipeline/broll.py).
    assert cache_key("cat", None, "768*1024") == "9ea532bbca6407388bf58507"


def test_cache_key_is_stable_and_case_insensitive():
    assert cache_key("Vintage Camera", None, "s") == cache_key("vintage camera", None, "s")
    assert cache_key("a", None, "s") != cache_key("b", None, "s")


async def test_without_a_generator_there_is_no_network_at_all(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover — не должен вызваться
        raise AssertionError(f"сети быть не должно: {request.url}")

    item = Broll(word_idx=0, duration=1.5, query="cat", anim=BrollAnim.POP_IN, anchor="center")
    cfg = cfg_without_env(
        image_provider="wavespeed", wavespeed_api_key="", fal_api_key="", gemini_api_key=""
    )

    async with client_for(handler) as c:
        assert await fetch_asset(c, item, tmp_path, cfg) is None


async def test_without_a_generator_the_plan_is_not_even_asked(tmp_path):
    """Рисовать нечем — значит, и vision-вызов плана вставок не нужен: он стоит денег."""

    class NoCalls:
        ready = True

        async def complete(self, *a, **kw):  # pragma: no cover — не должен вызваться
            raise AssertionError("план вставок спрашивать не должны")

    edl = edl_for_broll()
    edl.broll = [Broll(word_idx=0, duration=1.5, query="cat", anim=BrollAnim.POP_IN, anchor="fullscreen")]
    cfg = cfg_without_env(
        image_provider="wavespeed", wavespeed_api_key="", fal_api_key="", gemini_api_key=""
    )
    got = await plan_and_fetch(edl, tmp_path / "нет.mp4", tmp_path, llm=NoCalls(), cfg=cfg)
    assert got == 0 and edl.broll == []


# ------------------------------------------------------------------ листы времён стока


def _legacy_edl(*broll: dict) -> dict:
    data = edl_for_broll().model_dump(mode="json")
    data["broll"] = list(broll)
    return data


def test_legacy_photo_insert_still_loads():
    """Лист, сохранённый со стоком, читается: лишние поля картинки срезаются."""
    item = {**raw(), "anchor": "fullscreen", "duration": 1.5, "asset": "a.jpg",
            "kind": "photo", "asset_start": 0.0}
    edl = EDL.model_validate(_legacy_edl(item))
    assert [b.asset for b in edl.broll] == ["a.jpg"]


def test_legacy_stock_clip_is_dropped_not_crashing_the_rerender():
    """Клип со стока рендерер больше не играет: вставка выпадает, лист живёт."""
    clip = {**raw(), "anchor": "fullscreen", "duration": 1.5, "asset": "c@3.15.mp4",
            "kind": "video", "asset_start": 3.15}
    pic = {**raw(word_idx=6), "anchor": "fullscreen", "duration": 1.5, "asset": "p.jpg"}
    edl = EDL.model_validate(_legacy_edl(clip, pic))
    assert [b.asset for b in edl.broll] == ["p.jpg"]
