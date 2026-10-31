"""Тесты плана рендера: пересчёт времени, перенос строк, разъезд с b-roll, геометрия."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.core.config import Settings
from src.core.edl import (
    EDL,
    Broll,
    BrollAnim,
    CaptionBlock,
    CaptionWord,
    Cut,
    Emphasis,
    Word,
)
from src.core.geometry import (
    FALLBACK_ASPECT,
    Rect,
    caption_text_width,
    line_gap,
    measure_line,
    measure_word,
    metrics_for,
    space_width,
)
from src.core.preset import Preset, hex_to_rgb, load_preset
from src.core.render_plan import (
    PlanWord,
    RenderPlan,
    _rect_of_lines,
    _resolve_position,
    _Ruler,
    build_plan,
    fallback_line_chars,
    wrap_block,
    wrap_words,
)


def cfg() -> Settings:
    return Settings(sub_tail=0.15, render_fps=30, render_width=1080, render_height=1920)


def sample_edl(**kw) -> EDL:
    words = [Word(idx=i, text=f"сл{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(6)]
    data = dict(
        source_duration=10.0,
        words=words,
        cuts=[
            Cut(src_in=0.0, src_out=1.4, reason="hook"),
            Cut(src_in=1.4, src_out=2.0, reason="drop_silence"),
            Cut(src_in=2.0, src_out=3.2, reason="keep"),
            Cut(src_in=3.2, src_out=10.0, reason="drop_silence"),
        ],
        captions=[
            CaptionBlock(words=[CaptionWord(word_idx=0), CaptionWord(word_idx=1), CaptionWord(word_idx=2)]),
            CaptionBlock(words=[CaptionWord(word_idx=4, emphasis=Emphasis.ACCENT), CaptionWord(word_idx=5)]),
        ],
        preset="glass",
    )
    data.update(kw)
    return EDL(**data)


# ------------------------------------------------------------------ таймлайн


def test_times_are_remapped_to_output_timeline():
    edl = sample_edl()
    plan = build_plan(edl, "v.mp4", apply_cuts=True, cfg=cfg())
    first = plan.blocks[0].words[0]
    second = plan.blocks[1].words[0]
    # слово 4 в исходнике начинается на 2.0 с; после выреза 1.4–2.0 — на 1.4 с
    assert first.start == pytest.approx(0.0, abs=0.01)
    assert second.start == pytest.approx(1.4, abs=0.01)
    assert plan.duration == pytest.approx(2.6, abs=0.01)


def test_without_cuts_times_are_source_times():
    edl = sample_edl()
    plan = build_plan(edl, "v.mp4", apply_cuts=False, cfg=cfg())
    assert plan.blocks[1].words[0].start == pytest.approx(2.0, abs=0.01)
    assert plan.duration == pytest.approx(10.0)


def test_dropped_words_do_not_reach_the_plan():
    edl = sample_edl()
    plan = build_plan(edl, "v.mp4", apply_cuts=True, cfg=cfg())
    idxs = [w.word_idx for b in plan.blocks for w in b.words]
    assert 3 not in idxs  # слово 3 попало в вырез


def test_blocks_never_overlap():
    edl = sample_edl()
    plan = build_plan(edl, "v.mp4", apply_cuts=True, cfg=cfg())
    for a, b in zip(plan.blocks, plan.blocks[1:]):
        assert a.end <= b.start + 1e-6


def test_last_block_does_not_exceed_duration():
    edl = sample_edl()
    plan = build_plan(edl, "v.mp4", apply_cuts=True, cfg=cfg())
    assert plan.blocks[-1].end <= plan.duration + 1e-6


# ------------------------------------------------------------------ пресет


def test_uppercase_preset_uppercases_text():
    edl = sample_edl(preset="bold")
    plan = build_plan(edl, "v.mp4", preset=load_preset("bold"), cfg=cfg())
    assert all(w.text == w.text.upper() for b in plan.blocks for w in b.words)


def test_preset_is_embedded_whole():
    """Рендерер не читает файлы пресетов: всё приходит в плане (§0.3)."""
    plan = build_plan(sample_edl(), "v.mp4", cfg=cfg())
    assert plan.preset["name"] == "glass"
    assert "colors" in plan.preset and "animation" in plan.preset


def test_preset_scales_to_frame_height():
    tall = build_plan(sample_edl(), "v.mp4", height=1920, width=1080, cfg=cfg())
    short = build_plan(sample_edl(), "v.mp4", height=960, width=540, cfg=cfg())
    assert short.preset["font"]["size"] < tall.preset["font"]["size"]


def test_emphasis_survives_into_plan():
    plan = build_plan(sample_edl(), "v.mp4", cfg=cfg())
    accents = [w.emphasis for b in plan.blocks for w in b.words if w.emphasis != "none"]
    assert accents == ["accent"]


# ------------------------------------------------------------------ b-roll и камера


def broll_edl(anchor: str) -> EDL:
    edl = sample_edl(cuts=[Cut(src_in=0.0, src_out=10.0, reason="keep")])
    edl.broll = [
        Broll(word_idx=0, duration=2.0, query="q", anim=BrollAnim.POP_IN, anchor=anchor)
    ]
    return edl


@pytest.mark.parametrize(
    "anchor", ["top_left", "top_right", "bottom_left", "bottom_right", "center"]
)
def test_caption_never_shares_the_frame_with_a_picture(anchor: str):
    """§7 DoD: картинка не должна перекрывать текст — при любом якоре."""
    plan = build_plan(broll_edl(anchor), "v.mp4", cfg=cfg())
    caption = plan.blocks[0].rect
    picture = plan.broll[0].rect
    assert isinstance(caption, Rect) and isinstance(picture, Rect)
    assert not caption.overlaps(picture)


@pytest.mark.parametrize(
    "anchor", ["top_left", "top_right", "bottom_left", "bottom_right", "center"]
)
def test_an_explicit_centre_block_still_gets_its_room(anchor: str):
    """Гарантия Ф7 держится и на явном `center`, который режиссёр ставит сам.

    Прямоугольники блоков видит только сборка плана, и без них `broll_rect`
    уступает текст по резерву `caption_zone`, а когда-то и вовсе отбрасывал у
    glass позицию `center` — то есть карточка получала право встать ровно туда,
    куда режиссёр только что поставил блок. Здесь план обязан передать измеренные
    блоки вниз и получить либо карточку мимо текста, либо полный кадр.
    """
    edl = broll_edl(anchor)
    for block in edl.captions:
        block.position = "center"
    plan = build_plan(edl, "v.mp4", cfg=cfg())
    picture = plan.broll[0].rect
    assert isinstance(picture, Rect)
    for block in plan.blocks:
        assert block.position == "center"
        assert isinstance(block.rect, Rect)
        if picture.width >= plan.width:
            continue  # места под карточку не осталось — осознанный полный кадр
        assert not block.rect.overlaps(picture), anchor


@pytest.mark.parametrize(
    "anchor", ["top_left", "top_right", "bottom_left", "bottom_right", "center"]
)
@pytest.mark.parametrize("position", ["bottom", "top", "center"])
def test_a_block_shouting_on_every_line_still_keeps_the_picture_out(anchor: str, position: str):
    """Ф7 на блоке, который резерв не покрывал: крик в КАЖДОЙ строке.

    Полоса `caption_zone` резервировала «(строк−1)·базовая + самая высокая», то
    есть блок ровно с одним выделенным словом. Контракт разрешает `shout` в
    каждой строке, и такой блок у glass 1080×1920 на 63 px выше резерва —
    картинка вставала внутрь текста, а этот же тест был зелёным только потому,
    что в фикстуре крика не было.
    """
    words = [Word(idx=i, text="невероятно", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(3)]
    edl = EDL(
        source_duration=10.0,
        words=words,
        cuts=[Cut(src_in=0.0, src_out=3.0, reason="keep")],
        captions=[
            CaptionBlock(
                position=position,
                words=[CaptionWord(word_idx=i, emphasis=Emphasis.SHOUT) for i in range(3)],
            )
        ],
        broll=[
            Broll(word_idx=0, duration=2.0, query="q", anim=BrollAnim.POP_IN, anchor=anchor)
        ],
        preset="glass",
    )
    plan = build_plan(edl, "v.mp4", cfg=cfg())
    caption = plan.blocks[0].rect
    picture = plan.broll[0].rect
    assert isinstance(caption, Rect) and isinstance(picture, Rect)
    assert len(plan.blocks[0].lines) == 3, "тест про трёхстрочный блок"
    if picture.width >= plan.width:
        pytest.skip("места под карточку не осталось — осознанный полный кадр")
    assert not caption.overlaps(picture), (anchor, position)


def test_a_centre_block_does_not_shrink_cards_in_other_windows():
    """Позиции считаются ПО ОКНУ вставки, а не на весь ролик.

    Один блок с явным `center` добавлял `center` в общий набор позиций всего
    плана: полоса у glass схлопывалась ниже порога, и полноэкранными становились
    ВСЕ карточки ролика — включая те, что играют, когда центрального блока на
    экране нет.
    """
    words = [Word(idx=i, text=f"сл{i}", start=i * 1.0, end=i * 1.0 + 0.8) for i in range(4)]
    edl = EDL(
        source_duration=10.0,
        words=words,
        cuts=[Cut(src_in=0.0, src_out=10.0, reason="keep")],
        captions=[
            CaptionBlock(position="center", words=[CaptionWord(word_idx=0)]),
            CaptionBlock(position="bottom", words=[CaptionWord(word_idx=3)]),
        ],
        broll=[
            Broll(word_idx=0, duration=0.8, query="q", anim=BrollAnim.POP_IN, anchor="top_right"),
            Broll(word_idx=3, duration=0.8, query="q", anim=BrollAnim.POP_IN, anchor="top_right"),
        ],
        preset="glass",
    )
    plan = build_plan(edl, "v.mp4", cfg=cfg())
    late = plan.broll[1].rect
    assert isinstance(late, Rect)
    assert late.width < plan.width, "вставка вне окна центрального блока осталась карточкой"
    for block in plan.blocks:
        for item in plan.broll:
            if item.end <= block.start or item.start >= block.end:
                continue
            if item.rect is None or block.rect is None or item.rect.width >= plan.width:
                continue
            assert not block.rect.overlaps(item.rect)


def test_caption_moves_away_from_a_picture_over_its_band():
    """Уклонение решает пересечение прямоугольников, а не половина кадра.

    Половинами `center` описывался как «занято всё», и уклонение выключалось
    именно там, где было нужнее всего (AUDIT §8.3). Проверяем само правило:
    геометрия картинки чинится в Ф7 и уводит её из полосы субтитра сама, но
    правило обязано остаться верным.
    """
    assert _resolve_position("auto", "bottom", collides_bottom=True, collides_top=False) == "top"
    assert _resolve_position("auto", "top", collides_top=True, collides_bottom=False) == "bottom"
    # Занято и там и там — уходить некуда, менять один перехлёст на другой незачем.
    assert _resolve_position("auto", "bottom", collides_bottom=True, collides_top=True) == "bottom"
    # Центр — такая же полоса: у bold это позиция по умолчанию, и без этой ветки
    # автоуклонение было выключено у половины поставляемых тем целиком.
    assert _resolve_position("auto", "center", collides_center=True) == "bottom"
    assert (
        _resolve_position("auto", "center", collides_center=True, collides_bottom=True) == "top"
    )
    assert (
        _resolve_position(
            "auto", "center", collides_center=True, collides_bottom=True, collides_top=True
        )
        == "center"
    )
    assert _resolve_position("auto", "center", collides_bottom=True) == "center"


def test_caption_stays_when_broll_is_on_top():
    plan = build_plan(broll_edl("top_right"), "v.mp4", cfg=cfg())
    assert plan.blocks[0].position == "bottom"


def test_explicit_position_is_respected_over_collision():
    edl = broll_edl("bottom_right")
    edl.captions[0].position = "bottom"
    plan = build_plan(edl, "v.mp4", cfg=cfg())
    assert plan.blocks[0].position == "bottom"


def test_broll_is_timed_and_clamped():
    plan = build_plan(broll_edl("center"), "v.mp4", cfg=cfg())
    item = plan.broll[0]
    assert item.start == pytest.approx(0.0, abs=0.05)
    assert item.end <= plan.duration + 1e-6
    assert item.end - item.start == pytest.approx(2.0, abs=0.05)


def test_ken_burns_direction_reaches_the_renderer(tmp_path):
    """М3: знак наезда едет из EDL в план и переживает запись на диск.

    Проверяется именно сквозной путь: рендерер не имеет права выбирать
    направление сам, а значит план обязан его донести — в том числе перерендер,
    который читает сохранённый JSON.
    """
    edl = broll_edl("center")
    edl.broll[0].kb_dir = -1
    plan = build_plan(edl, "v.mp4", cfg=cfg())
    assert plan.broll[0].kb_dir == -1
    assert RenderPlan.from_dict(plan.to_dict()).broll[0].kb_dir == -1


def test_plan_without_ken_burns_still_reads():
    """План, собранный до М3, полей наезда не содержит — читается по-старому."""
    plan = build_plan(broll_edl("center"), "v.mp4", cfg=cfg())
    data = plan.to_dict()
    del data["broll"][0]["kb_dir"]
    assert RenderPlan.from_dict(data).broll[0].kb_dir == 1


def test_plan_round_trips_through_json():
    plan = build_plan(sample_edl(), "v.mp4", cfg=cfg())
    again = RenderPlan.from_dict(plan.to_dict())
    assert again.to_dict() == plan.to_dict()


def test_plan_json_has_no_none_video():
    plan = build_plan(sample_edl(), "v.mp4", cfg=cfg())
    assert plan.to_dict()["video"] == "v.mp4"
    assert plan.to_dict()["fps"] == 30


# ------------------------------------------------------------------ геометрия (Ф4)


def test_every_block_carries_a_rect_inside_the_frame():
    """Рендерер получает готовый прямоугольник, а не половину кадра (AUDIT §8.4)."""
    plan = build_plan(sample_edl(), "v.mp4", cfg=cfg())
    assert plan.blocks
    for b in plan.blocks:
        assert b.rect is not None
        assert 0 <= b.rect.x and b.rect.right <= plan.width + 1e-6
        assert 0 <= b.rect.y and b.rect.bottom <= plan.height + 1e-6


def test_block_rect_follows_the_resolved_position():
    """Прямоугольник считается ПОСЛЕ решения о позиции, а не до него."""
    up = sample_edl()
    up.captions[0].position = "top"
    down = sample_edl()
    down.captions[0].position = "bottom"
    top = build_plan(up, "v.mp4", cfg=cfg()).blocks[0]
    bottom = build_plan(down, "v.mp4", cfg=cfg()).blocks[0]
    assert top.position == "top" and bottom.position == "bottom"
    assert top.rect is not None and bottom.rect is not None
    assert top.rect.y < bottom.rect.y


def test_block_rect_scales_with_the_frame():
    tall = build_plan(sample_edl(), "v.mp4", height=1920, width=1080, cfg=cfg())
    short = build_plan(sample_edl(), "v.mp4", height=960, width=540, cfg=cfg())
    assert tall.blocks[0].rect.width > short.blocks[0].rect.width  # type: ignore[union-attr]
    assert short.blocks[0].rect.right <= 540 + 1e-6  # type: ignore[union-attr]


def test_broll_carries_the_rect_of_its_anchor():
    """Якорь виден в плане: карточка прижата к правому краю и остаётся карточкой.

    Сравнивать с `broll_rect` без контекста больше нельзя: сборка плана разводит
    картинку с ИЗМЕРЕННЫМИ прямоугольниками блоков, а вызов без них уступает
    всё, что разрешает контракт, и у glass честно вырождается в полный кадр.
    """
    plan = build_plan(broll_edl("bottom_right"), "v.mp4", cfg=cfg())
    preset = load_preset("glass").scaled(plan.height)
    rect = plan.broll[0].rect
    assert rect is not None and rect.width < plan.width
    margin = plan.width * preset.geometry.broll_margin_ratio
    assert plan.width - rect.right == pytest.approx(margin)
    assert rect.width / rect.height == pytest.approx(FALLBACK_ASPECT)


def test_fullscreen_to_inset_rect_is_the_corner_it_ends_in():
    edl = broll_edl("fullscreen")
    edl.broll[0].anim = BrollAnim.FULLSCREEN_TO_INSET
    plan = build_plan(edl, "v.mp4", cfg=cfg())
    rect = plan.broll[0].rect
    assert rect is not None and rect.width < plan.width


def test_rects_survive_the_json_round_trip():
    plan = build_plan(broll_edl("bottom_right"), "v.mp4", cfg=cfg())
    again = RenderPlan.from_dict(plan.to_dict())
    assert again.blocks[0].rect == plan.blocks[0].rect.rounded()  # type: ignore[union-attr]
    assert again.broll[0].rect == plan.broll[0].rect.rounded()  # type: ignore[union-attr]


def test_plan_without_rects_still_loads():
    """Планы, собранные до Ф4, читаются: прямоугольников просто нет."""
    data = build_plan(broll_edl("bottom_right"), "v.mp4", cfg=cfg()).to_dict()
    for item in data["blocks"] + data["broll"]:
        item.pop("rect")
    plan = RenderPlan.from_dict(data)
    assert plan.blocks[0].rect is None
    assert plan.broll[0].rect is None


def test_missing_font_metrics_do_not_break_the_plan():
    """Нет метрик — нет прямоугольника, но ролик всё равно рендерится."""
    preset = load_preset("glass").model_copy(deep=True)
    preset.font.file = None
    plan = build_plan(sample_edl(), "v.mp4", preset=preset, cfg=cfg())
    assert plan.blocks and plan.blocks[0].rect is None
    # Без прямоугольника рендерер обязан переносить строку сам — значит и кегли
    # ему не проставляют: раскладку в этом случае не считал никто.
    assert all(w.size is None for b in plan.blocks for w in b.words)


@pytest.mark.parametrize("name", ["glass", "bold", "pulse"])
def test_fallback_line_limit_follows_the_preset(name: str):
    """Запасной лимит по символам считается от кегля и регистра, а не константой.

    Одно число на все пресеты выведено под glass (88 px, строчные) и для bold
    (104 px, UPPERCASE) означает переполнение: пятнадцать прописных дают строку
    в 1098 px при бюджете 842 (AUDIT §8.1 замерил 11.9 символа). Проверяем не
    само число, а свойство: строка запасного пути укладывается в бюджет.
    """
    preset = load_preset(name).scaled(FRAME[1])
    metrics = metrics_for(preset)
    base = float(preset.font.size)
    budget = caption_text_width(preset, FRAME[0])
    limit = fallback_line_chars(preset, FRAME[0])
    assert limit >= 4
    for block in baseline_blocks():
        words = plan_words(block, preset)
        for line in wrap_words(words, limit, preset.caption.max_lines):
            width = measure_line(
                [(w.text, base) for w in line],
                metrics=metrics,
                letter_spacing=preset.font.letter_spacing,
                base_size=base,
            )
            assert width <= budget, (name, [w.text for w in line], width)


def test_caption_and_broll_rects_are_comparable():
    """Разводить субтитр с картинкой теперь можно пересечением, а не половинами."""
    plan = build_plan(broll_edl("bottom_right"), "v.mp4", cfg=cfg())
    caption = plan.blocks[0].rect
    picture = plan.broll[0].rect
    assert isinstance(caption, Rect) and isinstance(picture, Rect)
    assert not caption.overlaps(picture)


# ------------------------------------------------------------------ перенос строк (Ф5)

SAMPLE_PLAN = Path(__file__).resolve().parents[1] / "samples" / "plan-baseline.json"
FRAME = (1080, 1920)

# Слово плана: текст, выделение, эмодзи.
Spoken = tuple[str, str, str | None]


def baseline_blocks() -> list[list[Spoken]]:
    """Блоки настоящего прогона: 12 штук, живые акценты и крики."""
    data = json.loads(SAMPLE_PLAN.read_text(encoding="utf-8"))
    return [
        [(w["text"], w["emphasis"], w.get("emoji")) for line in b["lines"] for w in line]
        for b in data["blocks"]
    ]


def synthetic_blocks() -> list[list[Spoken]]:
    """Досинтезированные случаи: длинные слова, эмодзи, акценты, широкие глифы.

    Прогона на настоящих блоках мало: в нём нет ни эмодзи, ни слова, которое не
    влезает в строку в одиночку, — то есть ровно того, на чём перенос по
    символам и ломался (AUDIT §8.1).
    """
    return [
        [("предпринимательство", "accent", None)],
        [("трансцендентальность", "shout", None)],
        [("высокотехнологичный", "none", None), ("подход", "accent", None)],
        # Самые широкие глифы кириллицы (Ш 1.099 em, Ж 1.109 em) ровно на
        # границе: на поле 0.4 в бюджет строки влезает 19 таких в bold и 22 в
        # glass — см. test_word_wider_than_the_floor_overflows_knowingly.
        [("шшшшшшшшшшшшшшшшшш", "none", None)],
        [("ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ", "shout", None)],
        [("огонь", "accent", "🔥"), ("прямо", "none", None), ("сейчас", "none", "🚀")],
        [("деньги", "none", "💰"), ("любят", "none", None), ("счёт", "accent", "🧮")],
        [("одно", "none", None), ("слово", "none", None), ("в", "none", None),
         ("строке", "shout", None)],
        [("а", "none", None)],
        [("миллиард", "shout", "💥")],
        [(w, "none", None) for w in "раз два три четыре пять шесть семь восемь".split()],
        [(w, "none", None)
         for w in "и ещё двенадцать коротких слов подряд чтобы блок точно не влез".split()],
        [("сверхпроводимость", "accent", None), ("это", "none", None), ("физика", "none", None)],
        [("Ярославль-Главный", "none", None), ("платформа", "accent", None)],
        [("что-нибудь", "none", None), ("да", "none", None), ("выйдет", "shout", None)],
        [("2026", "accent", None), ("год", "none", None), ("уже", "none", None),
         ("здесь", "none", None)],
        [("самоидентификация", "shout", None), ("личности", "none", None)],
        [("Ш", "accent", None), ("Щ", "shout", None), ("Ъ", "none", None), ("Ы", "none", None)],
    ]


def all_blocks() -> list[list[Spoken]]:
    blocks = baseline_blocks() + synthetic_blocks()
    assert len(blocks) >= 30, "DoD Ф5 просит тридцать блоков"
    return blocks


def plan_words(block: list[Spoken], preset: Preset) -> list[PlanWord]:
    """Слова блока в экранной форме: регистр решает пресет, как в build_plan."""
    return [
        PlanWord(
            text=t.upper() if preset.font.uppercase else t,
            start=0.0,
            end=1.0,
            emphasis=e,
            emoji=emoji,
            word_idx=i,
        )
        for i, (t, e, emoji) in enumerate(block)
    ]


def ruler_of(preset: Preset) -> _Ruler:
    return _Ruler(
        metrics=metrics_for(preset),
        letter_spacing=preset.font.letter_spacing,
        base_size=float(preset.font.size),
        budget=caption_text_width(preset, FRAME[0]),
    )


@pytest.mark.parametrize("name", ["glass", "bold", "pulse"])
def test_wrapped_text_never_leaves_the_caption_rect(name: str):
    """DoD Ф5: тридцать блоков, ни один не вылезает за отведённую ширину.

    Ширина строки, где нет эмодзи, сверяется с НЕЗАВИСИМОЙ линейкой
    `geometry.measure_line`: сравнение с `_Ruler` было бы сравнением функции с
    самой собой — им же и считал `wrap_block`. Строку с эмодзи независимой
    линейкой не измерить (глиф рисует системный шрифт), её ширину сторожит
    медленная сверка с Chromium.
    """
    preset = load_preset(name).scaled(FRAME[1])
    ruler = ruler_of(preset)
    metrics = metrics_for(preset)
    base = float(preset.font.size)
    for block in all_blocks():
        words = plan_words(block, preset)
        lines, widths = wrap_block(words, preset, FRAME[0])
        assert [w.word_idx for line in lines for w in line] == [w.word_idx for w in words]
        for line, width in zip(lines, widths):
            if any(w.emoji for w in line):
                assert width == pytest.approx(ruler.line_width(line), abs=0.01)
            else:
                assert width == pytest.approx(
                    measure_line(
                        [(w.text, w.size or base) for w in line],
                        metrics=metrics,
                        letter_spacing=preset.font.letter_spacing,
                        base_size=base,
                    ),
                    abs=0.01,
                )
            # Полпикселя допуска: бюджет решается точно, но кегль уезжает в
            # план округлённым до сотых, и ширина сдвигается на эти сотые.
            assert width <= ruler.budget + 0.5, (name, [w.text for w in line], width)


@pytest.mark.parametrize("name", ["glass", "bold", "pulse"])
def test_words_in_a_line_never_overlap(name: str):
    """DoD Ф5: слова разъезжаются на настоящий пробел, а не наезжают друг на друга.

    Именно это ломал акцент через `transform: scale`: масштаб не участвовал в
    раскладке, и слово накрывало соседа на 5-122 px (AUDIT §8.1).

    Мерить тут `_Ruler` нельзя: им же считал `wrap_block`, и сравнение с ним —
    тождество, истинное при любой раскладке (проверено: тест с раскладкой «все
    слова в одну строку кеглем ×3» проходил). Поэтому коробки слов считает
    НЕЗАВИСИМАЯ реализация из `geometry`, слова кладутся по кеглям из плана, и
    проверяется, что сумма коробок с пробелами равна независимо измеренной
    ширине строки и что правый край строки укладывается в бюджет.

    Эмодзи здесь нет: его глиф рисует системный шрифт, метрик у нас на него нет
    (`EMOJI_ADVANCE_EM` — константа линейки), и независимой линейкой строку с
    эмодзи не измерить. Эти строки покрывает медленный тест в Chromium.
    """
    preset = load_preset(name).scaled(FRAME[1])
    metrics = metrics_for(preset)
    base = float(preset.font.size)
    ls = preset.font.letter_spacing
    budget = caption_text_width(preset, FRAME[0])
    # Пробел между словами меряет `space_width`, а не `measure_word(" ")`: в
    # боксе слова краевой пробел схлопывается и стоит ноль, между боксами — нет.
    space = space_width(base, metrics=metrics, letter_spacing=ls)
    assert space > 0
    seen = 0
    for block in all_blocks():
        words = plan_words(block, preset)
        lines, _ = wrap_block(words, preset, FRAME[0])
        for line in lines:
            if any(w.emoji for w in line):
                continue
            seen += 1
            x = 0.0
            previous_right = 0.0
            for i, word in enumerate(line):
                own = measure_word(
                    word.text,
                    word.size or base,
                    metrics=metrics,
                    letter_spacing=ls,
                    spacing_size=base,
                )
                assert own > 0, (name, word.text)
                if i:
                    # Промежуток — настоящий пробел кеглем СТРОКИ, и он не
                    # зависит от того, насколько крупным вышел сосед.
                    assert x - previous_right == pytest.approx(space, abs=0.01)
                previous_right = x + own
                x = previous_right + space
            # Правый край строки, сложенный из независимо измеренных коробок,
            # совпадает с независимо измеренной шириной строки и не выходит за
            # бюджет: ни наезда, ни щели, ни переполнения.
            independent = measure_line(
                [(w.text, w.size or base) for w in line],
                metrics=metrics,
                letter_spacing=ls,
                base_size=base,
            )
            assert previous_right == pytest.approx(independent, abs=0.01), name
            assert previous_right <= budget + 0.5, (name, [w.text for w in line])
    assert seen >= 30, "проверять надо хотя бы тридцать строк"


@pytest.mark.parametrize("name", ["glass", "bold", "pulse"])
def test_block_rect_holds_every_line(name: str):
    """Прямоугольник блока накрывает текст целиком — и по ширине, и по ВЫСОТЕ.

    Высота проверяется отдельно и по независимой формуле: строчный бокс не
    бывает ниже струта (`font.size * line_height`), потому что `line-height`
    наследуется от контейнера строки и от кегля слова не зависит. Пока высота
    считалась только по самому крупному слову, ужатая строка занижала блок на
    45-164 px, и весь промах уходил вверх — блок прибит к низу кадра.
    """
    preset = load_preset(name).scaled(FRAME[1])
    strut = float(preset.font.size) * preset.caption.line_height
    for block in all_blocks():
        words = plan_words(block, preset)
        lines, widths = wrap_block(words, preset, FRAME[0])
        rect = _rect_of_lines("bottom", lines, widths, preset, FRAME)
        panel = preset.panel
        chrome_x = 2 * (panel.padding_x + panel.border_width) if panel.enabled else 0
        chrome_y = 2 * (panel.padding_y + panel.border_width) if panel.enabled else 0
        text_h = sum(
            max(strut, max((w.size or 0.0) for w in line) * preset.caption.line_height)
            for line in lines
        ) + line_gap(preset) * (len(lines) - 1)
        assert rect.width >= max(widths) + chrome_x - 0.01
        assert rect.height == pytest.approx(text_h + chrome_y, abs=0.01), (
            name,
            [[w.text for w in line] for line in lines],
        )
        assert rect.y >= 0 and rect.bottom <= FRAME[1] + 0.01
        # За кадр прямоугольник выходит ровно в одном случае — когда слово
        # сознательно переполняет бюджет (пол кегля). Тогда это правда о кадре:
        # панель действительно вылезет, и зажимать её в `_rect_of_lines` нельзя,
        # иначе прямоугольник перестанет накрывать свой же текст.
        if max(widths) <= caption_text_width(preset, FRAME[0]) + 0.5:
            assert rect.x >= 0 and rect.right <= FRAME[0] + 0.01


def test_accent_is_a_size_and_not_a_scale():
    """Акцент приезжает в план кеглем: рендереру нечего домножать.

    Абсолютные кегли здесь не проверяются: после С1 блок растёт под ширину
    строки, и оба слова уезжают вверх ОДНИМ множителем. Проверяется то, что
    этот рост не трогает, — отношение между выделенным словом и соседом.
    """
    preset = load_preset("glass").scaled(FRAME[1])
    words = plan_words([("тише", "none", None), ("ор", "accent", None)], preset)
    wrap_block(words, preset, FRAME[0])
    assert words[0].size and words[1].size
    assert words[1].size / words[0].size == pytest.approx(
        preset.animation.accent_scale, abs=0.01
    )

    # Исключение ровно одно и оно старше С1: слово, которое не влезает в строку
    # даже в одиночку, ужимается под бюджет (`fit_size`). Тогда отношение
    # проседает — но выделенное слово обязано остаться КРУПНЕЕ соседа.
    long_words = plan_words([("тише", "none", None), ("громче", "accent", None)], preset)
    wrap_block(long_words, preset, FRAME[0])
    assert long_words[1].size > long_words[0].size
    assert long_words[1].size / long_words[0].size < preset.animation.accent_scale


def test_block_grows_until_the_line_fills_the_budget():
    """С1: кегль подбирается ПОД строку, а не задан числом в пресете.

    У эталона (`STYLE_PLAN.md`) «также» из пяти букв набрано вдвое крупнее, чем
    «звуковые» из восьми, и строка в обоих случаях добирает 65-83 % ширины
    кадра. Фиксированным кеглем это не выражается: при 88 px наша строка
    занимала 27 % ширины, и иерархии в кадре не читалось.

    Проверяются оба конца правила: строка добирает бюджет, и короткая строка
    из-за этого набрана крупнее длинной.
    """
    preset = load_preset("glass").scaled(FRAME[1])
    budget = caption_text_width(preset, FRAME[0])

    short = plan_words([("также", "accent", None)], preset)
    _, short_w = wrap_block(short, preset, FRAME[0])
    long_ = plan_words([("звуковые", "accent", None)], preset)
    _, long_w = wrap_block(long_, preset, FRAME[0])

    assert short_w[0] / budget > 0.7, f"короткая строка не добрала бюджет: {short_w[0]:.0f}"
    assert long_w[0] / budget > 0.9, f"длинная строка не добрала бюджет: {long_w[0]:.0f}"
    assert short[0].size > long_[0].size * 1.5, "короткое слово обязано быть крупнее"
    # И то и другое кратно выше прежнего фиксированного кегля.
    assert long_[0].size > preset.font.size * 1.5


def test_a_preset_without_growth_keeps_its_kegel():
    """Рост включает ТЕМА, а не код: `bold` обязан остаться прежним.

    Дефолт `fill_max_scale` — 1.0, и на нём кегль слова равен кеглю пресета,
    ровно как до С1.
    """
    preset = load_preset("bold").scaled(FRAME[1])
    assert preset.caption.fill_max_scale == 1.0
    words = plan_words([("слово", "none", None)], preset)
    wrap_block(words, preset, FRAME[0])
    assert words[0].size == pytest.approx(preset.font.size)


def test_long_word_shrinks_instead_of_overflowing():
    """Длинное слово ужимается до влезания, а не переполняет контейнер."""
    preset = load_preset("glass").scaled(FRAME[1])
    words = plan_words([("трансцендентальностью", "shout", None)], preset)
    _, widths = wrap_block(words, preset, FRAME[0])
    assert words[0].size is not None and words[0].size < preset.font.size
    assert widths[0] <= caption_text_width(preset, FRAME[0]) + 0.5


def test_emoji_takes_room_in_the_line():
    """Эмодзи считается: раньше он не входил в бюджет вовсе (+114 px на слово)."""
    preset = load_preset("glass").scaled(FRAME[1])
    plain = plan_words([("огонь", "none", None)], preset)
    fiery = plan_words([("огонь", "none", "🔥")], preset)
    _, plain_w = wrap_block(plain, preset, FRAME[0])
    _, fiery_w = wrap_block(fiery, preset, FRAME[0])
    assert fiery_w[0] > plain_w[0] + preset.font.size * 0.5


def test_word_sizes_survive_the_json_round_trip():
    plan = build_plan(sample_edl(), "v.mp4", cfg=cfg())
    again = RenderPlan.from_dict(plan.to_dict())
    assert [w.size for b in again.blocks for w in b.words] == [
        w.size for b in plan.blocks for w in b.words
    ]
    assert all(w.size for b in plan.blocks for w in b.words)


# ------------------------------------------------------------------ контраст (Ф9)


def test_bg_luma_reaches_the_plan():
    """Яркость фона меряет препроцесс, план её только доносит до рендерера."""
    edl = sample_edl()
    edl.captions[0].bg_luma = 0.81
    plan = build_plan(edl, "v.mp4", cfg=cfg())
    assert plan.blocks[0].bg_luma == pytest.approx(0.81)
    assert plan.blocks[1].bg_luma is None
    again = RenderPlan.from_dict(plan.to_dict())
    assert again.blocks[0].bg_luma == pytest.approx(0.81)
    assert again.blocks[1].bg_luma is None


def test_presets_carry_a_contrast_rule():
    """Правило контраста — данные пресета, а не число в компоненте (§0.3)."""
    for name in ("glass", "bold"):
        rule = load_preset(name).contrast
        assert rule.min_ratio >= 3.0
        assert rule.luma_dark < rule.luma_light


def test_glass_blocks_never_overlap_in_time():
    """С4 DoD: на стыке двух блоков нет кадра, где видны оба.

    Раньше блок уходил ПОСЛЕ своего конца и всё время хвоста лежал на
    приходящем — блоки идут встык (`a.end == b.start`), стоят в одном месте
    кадра, и разводить их приходилось ДВИЖЕНИЕМ: уходящий уезжал на собственную
    высоту. Эталон решает иначе — между блоками у него три кадра без текста
    вообще (замерено на кадрах 165-167), нахлёста нет ни на одном стыке.

    Здесь проверяется то, от чего это зависит: уход обязан уложиться в
    собственное время блока и оставить паузу до прихода следующего.
    """
    preset = load_preset("glass")
    a = preset.animation
    assert a.block_gap_ms > 0, "без паузы блок уходит ПОСЛЕ конца и ложится на следующий"

    plan = build_plan(sample_edl(), "v.mp4", cfg=cfg())
    fps = plan.fps

    def frames(ms: float) -> int:
        # Тот же расчёт, что `fadeFrames` в style.ts: пол в один кадр, а не в
        # восемь — восьмикадровый порог про ДВИЖЕНИЕ, а не про прозрачность.
        return max(1, round(ms / 1000 * fps))

    gap, out = frames(a.block_gap_ms), frames(a.block_out_ms)
    for i, block in enumerate(plan.blocks):
        life = round(block.end * fps) - round(block.start * fps)
        assert out + gap <= life, (
            f"блок {i}: уход {out} + пауза {gap} не влезает в его жизнь {life} кадров — "
            "текст начнёт таять на собственных словах"
        )
        if i + 1 < len(plan.blocks):
            leaves = round(block.end * fps) - gap
            arrives = round(plan.blocks[i + 1].start * fps)
            assert leaves <= arrives, f"блоки {i} и {i + 1} пересекаются во времени"


def test_glass_word_states_are_on_or_off():
    """С2: сказанное слово не гаснет, будущего не видно вовсе.

    У эталона (`STYLE_PLAN.md`) промежуточных состояний нет: слово либо
    проявилось и стоит в полную силу, либо его ещё нет. Наши 0.7 на прошедшем и
    0.18 на будущем давали третий и четвёртый оттенок серого в каждом кадре —
    аналога им в эталоне не нашлось ни на одном из 709 кадров.

    Место под будущее слово при этом остаётся: `ghost_opacity` гасит слово, не
    вынимая его из потока (Captions.tsx), иначе блок переезжал бы по мере
    набора — а он у эталона стоит.
    """
    colors = load_preset("glass").colors
    assert colors.past_opacity == 1.0
    assert colors.ghost_opacity == 0.0
    # Акцент — приглушённое золото, замеренное медианой по глифам эталона,
    # а не кислотно-жёлтый.
    assert colors.accent.upper() == "#F3D28F"
    r, g, b = hex_to_rgb(colors.accent)
    assert r > g > b, "золото: красного больше зелёного, зелёного больше синего"


def _luminance(color: str) -> float:
    """Относительная яркость цвета по WCAG — та же формула, что в Captions.tsx."""
    channels = [c / 255 for c in hex_to_rgb(color)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: float, b: float) -> float:
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def _to_srgb(linear: float) -> float:
    """Линейная яркость → sRGB. Chromium смешивает альфу именно здесь."""
    return 12.92 * linear if linear <= 0.0031308 else 1.055 * linear ** (1 / 2.4) - 0.055


def _to_linear(srgb: float) -> float:
    return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4


def _panel_opacity(preset: Any, bg: float) -> float:
    """Прозрачность панели glass под фоном `bg` — тот же расчёт, что в `lookFor`.

    Стилевая часть у этого пресета нулевая (обе опорные прозрачности 0), поэтому
    остаётся только аварийный рычаг: подъём ровно до порога.

    Два выхода в начале — не оптимизация, а повтор `lookFor`: там правило с
    `enabled: false` возвращает базу целиком, а подъём до порога стоит под
    `preset.panel.enabled`. Без них зеркало считало панель у темы, где панели
    нет вовсе, и показывало 0.019 там, где рендерер рисует ноль.
    """
    rule = preset.contrast
    if not rule.enabled or not preset.panel.enabled:
        return preset.panel.opacity if preset.panel.enabled else 0.0
    text = _luminance(preset.colors.active)
    panel = _luminance(preset.panel.color)
    span = rule.luma_light - rule.luma_dark
    t = min(1.0, max(0.0, (bg - rule.luma_dark) / span))
    opacity = rule.panel_opacity_dark + (rule.panel_opacity_light - rule.panel_opacity_dark) * t

    def under(alpha: float) -> float:
        return _to_linear(_to_srgb(bg) + (_to_srgb(panel) - _to_srgb(bg)) * alpha)

    if _contrast(text, under(opacity)) < rule.min_ratio:
        target = (text + 0.05) / rule.min_ratio - 0.05
        lo, hi = _to_srgb(bg), _to_srgb(panel)
        opacity = max(opacity, min(1.0, max(0.0, (_to_srgb(target) - lo) / (hi - lo))))
    return min(1.0, max(0.0, opacity))


def test_glass_has_no_background_under_the_caption_at_all():
    """У glass фона под субтитром нет НИ ПРИ КАКОМ фоне кадра — решение владельца.

    Здесь снят предохранитель, а не украшение, и это осознанно. Правило Ф9
    поднимало тёмную заслонку там, где белый текст не берёт порог 4.5 — то есть
    на светлом фоне, начиная примерно с `bg_luma` 0.18. Владелец продукта видел
    этот аргумент и ответил: фон убрать совсем, случай светлой стены не
    рассматривать. Читаемость держит одна мягкая тень (`softShadow`).

    Замер эталона на его стороне: из тринадцати блоков одиннадцать идут вообще
    без подложки, и ни один — с непрозрачной. Оставшиеся два несли матовое
    стекло, но это украшение, а не предохранитель: с яркостью фона под блоком и
    с его суетливостью оно не коррелирует (STYLE_PLAN §1.4).

    Тест сторожит именно РЕШЕНИЕ. Если кто-то вернёт панель ради контраста, он
    сначала уронит этот тест и прочитает, почему её убрали.
    """
    preset = load_preset("glass")
    assert not preset.panel.enabled, "фон под субтитром убран целиком"
    assert not preset.contrast.enabled, (
        "правило Ф9 выключено: его единственный рычаг у этой темы — панель"
    )
    # Ни одна ручка панели не должна оставаться взведённой: выключенный флаг
    # прячет их сегодня, но включённый обратно вернул бы фон молча.
    assert preset.panel.opacity == 0.0
    assert preset.panel.glass_opacity == 0.0
    assert preset.panel.border_opacity == 0.0
    assert preset.panel.blur == 0
    assert preset.contrast.panel_opacity_dark == 0.0
    assert preset.contrast.panel_opacity_light == 0.0

    # Правило выключено — значит панели нет на ЛЮБОЙ яркости фона, включая ту,
    # ради которой предохранитель и существовал.
    for step in range(101):
        assert _panel_opacity(preset, step / 100) == 0.0


def test_pulse_wears_frosted_glass_without_any_fill():
    """У pulse стекло есть, а заливки нет — и это два разных слоя (§1.4).

    Владелец просил убрать фон под субтитром, и про ЗАЛИВКУ он был прав: в
    эталоне её нет ни у одного блока. Но вместе с заливкой был обнулён и сам
    стеклянный слой — размытие с рамкой, — а он в эталоне есть у 2 блоков из 13
    (3.5 с из 23.6): внутри плашки видимо смягчаются принт на футболке и пальцы,
    снаружи они резкие. Числа сняты с кадров 483 и 848, а не выбраны: подъём
    яркости над чёрной футболкой с 5.6 до 41.5 из 255 — это белая заливка 0.14,
    рамка 1 px при ширине 464 — это 2 px при базе 1920.

    Тест сторожит обе половины решения: стекло взведено, заливка остаётся нулём
    при ЛЮБОЙ яркости фона.
    """
    preset = load_preset("pulse")
    panel = preset.panel
    assert panel.enabled, "стекло носят помеченные блоки — выключать тему нельзя"
    assert (panel.blur, panel.radius) == (12, 30)
    assert (panel.border_color, panel.border_opacity, panel.border_width) == ("#FFFFFF", 0.55, 2)
    assert (panel.glass_color, panel.glass_opacity) == ("#FFFFFF", 0.14)

    # Заливка — другой слой: тёмная заслонка правила Ф9. У pulse она нулевая, и
    # правило выключено, поэтому фон не приходит ни при какой яркости кадра.
    assert panel.opacity == 0.0
    assert not preset.contrast.enabled
    for step in range(101):
        assert _panel_opacity(preset, step / 100) == 0.0

    # Тихий регистр стекла не носит: §1.4.1 — «плашка: иногда» у плаката и
    # «нет» у тихого. Гасится оно вместе с хромом, иначе Python и Chromium
    # разошлись бы на паддинги.
    assert preset.caption.quiet.panel is False
    assert preset.caption.quiet.preset(preset.scaled(FRAME[1])).panel.enabled is False


def test_the_panel_flag_reaches_the_plan_and_only_the_marked_block():
    """Флаг плашки доезжает до рендерера поблочно, а место под неё — у всех.

    Рендерер рисует стекло по `block.panel` (`Captions.tsx`), а прямоугольник
    считает Python по хрому ТЕМЫ — одинаково у всех блоков. Иначе текст прыгал
    бы на высоту плашки между соседними блоками.
    """
    marked = build_plan(
        register_edl("bottom", "Я думаю что каждый".split(), accent=1, quiet=False, panel=True),
        "v.mp4",
        cfg=cfg(),
    )
    plain = build_plan(
        register_edl("bottom", "Я думаю что каждый".split(), accent=1, quiet=False),
        "v.mp4",
        cfg=cfg(),
    )
    assert marked.to_dict()["blocks"][0]["panel"] is True
    assert plain.to_dict()["blocks"][0]["panel"] is False
    assert marked.blocks[0].rect == plain.blocks[0].rect


@pytest.mark.parametrize("name", ["glass", "bold", "pulse"])
def test_contrast_rule_has_a_lever_on_every_background(name: str):
    """У правила Ф9 обязан быть рычаг на ЛЮБОЙ яркости фона, а не только на краях.

    Само правило живёт в `Captions.tsx`, но его гарантия держится на ДАННЫХ: на
    каждом фоне хоть один рычаг обязан брать порог — либо опорный цвет текста,
    либо панель. Пока это не проверялось, bold (панели нет, обе опорные
    прозрачности нулевые) оставался вообще без рычага: линейная смесь
    #FFFFFF → #0A0D12 проходила через яркость фона и на bg_luma 0.40 давала
    контраст 1.05 — хуже, чем 2.33 у белого текста без всякого правила.
    Проверены были только концы диапазона, поэтому провал середины не видели.
    """
    preset = load_preset(name)
    rule = preset.contrast
    if not rule.enabled:
        pytest.skip(f"{name}: правило контраста выключено сознательно — гарантировать нечего")
    anchors = (rule.text_dark, rule.text_light)
    for step in range(101):
        bg = step / 100
        best = max(_contrast(_luminance(color), bg) for color in anchors)
        if preset.panel.enabled:
            # Панель — второй рычаг: на полной непрозрачности под текстом лежит
            # её собственный цвет, фона не видно вовсе.
            panel = _luminance(preset.panel.color)
            best = max(best, *(_contrast(_luminance(color), panel) for color in anchors))
        assert best >= rule.min_ratio, (name, bg, round(best, 2))


def test_word_wider_than_the_floor_overflows_knowingly():
    """Честная граница переноса: ниже пола кегля мы не опускаемся.

    Пол `caption.min_word_scale` — сознательный выбор: слово мельче 0.4 кегля
    строки перестаёт быть субтитром, и переполнение на нём честнее микроскопии.
    Тест фиксирует, где именно проходит граница, чтобы она не уехала молча.

    Глиф здесь самый широкий из возможных — «Щ» (1.135 em). «Ж» (1.109) и «Ш»
    (1.099) уже, и граница по ним оказывалась на символ оптимистичнее реальной:
    девятнадцать «Ж» в бюджет влезают, а девятнадцать «Щ» — нет.
    """
    preset = load_preset("bold").scaled(FRAME[1])
    metrics = metrics_for(preset)
    widest = max("ШЖЩ", key=lambda c: metrics.advance(c))
    assert widest == "Щ"
    budget = caption_text_width(preset, FRAME[0])
    _, at_limit = wrap_block(plan_words([(widest * 18, "none", None)], preset), preset, FRAME[0])
    _, beyond = wrap_block(plan_words([(widest * 19, "none", None)], preset), preset, FRAME[0])
    assert at_limit[0] <= budget + 0.5
    assert beyond[0] > budget
    # Переполнение остаётся в пределах разумного: слово не уезжает за кадр.
    assert beyond[0] <= FRAME[0]


# ------------------------------------------------------------- два регистра (§1.4.1)

# Эталонный тихий блок: «Я лишь отправляю ему сырое видео», 3.6-5.2 с. Шесть
# слов в ЧЕТЫРЕ строки — по нему и подобраны кегль с бюджетом колонки.
QUIET_LINE = "Я лишь отправляю ему сырое видео".split()
# Высота глифов Montserrat ExtraBold: cap_height 700 / upem 1000. Замеры
# STYLE_PLAN даны в высоте глифов, а не в кегле, и сравнивать надо с ними.
CAP_HEIGHT_EM = 0.7


def glyph_share(block, frame_height: int = FRAME[1]) -> float:
    """Самое крупное слово блока как доля высоты кадра — единица замеров §1.1."""
    return max(w.size or 0.0 for w in block.words) * CAP_HEIGHT_EM / frame_height


def register_edl(
    position: str,
    words: list[str] = QUIET_LINE,
    *,
    accent: int | None = None,
    anchor: str | None = None,
    quiet: bool | None = None,
    panel: bool = False,
) -> EDL:
    """Один блок в заданной позиции; `accent` — индекс золотого слова.

    `quiet` — объявленный регистр; `None` — лист без поля, то есть регистр
    решает позиция, как решала до его появления.
    """
    spoken = [Word(idx=i, text=t, start=i * 0.4, end=i * 0.4 + 0.3) for i, t in enumerate(words)]
    return EDL(
        source_duration=10.0,
        words=spoken,
        cuts=[Cut(src_in=0.0, src_out=10.0, reason="keep")],
        captions=[
            CaptionBlock(
                position=position,
                quiet=quiet,
                panel=panel,
                words=[
                    CaptionWord(
                        word_idx=i,
                        emphasis=Emphasis.ACCENT if i == accent else Emphasis.NONE,
                    )
                    for i in range(len(words))
                ],
            )
        ],
        broll=(
            [Broll(word_idx=0, duration=2.0, query="q", anim=BrollAnim.POP_IN, anchor=anchor)]
            if anchor
            else []
        ),
        preset="pulse",
    )


def test_a_top_block_renders_quiet():
    """Верхний блок — другой КЛАСС субтитра, а не уменьшённый плакат (§1.4.1).

    Замеры эталона, которые тут и сторожатся: кегль ~1.5 % высоты кадра против
    6.2 % у плаката, строк четыре, выключка влево у левого края. Пока стиль от
    позиции не зависел, тот же блок выезжал наверх плакатным кеглем 174.8 px
    (6.37 % высоты) с золотым словом — ровно то, чего в верхнем кластере
    эталона нет ни на одном кадре.
    """
    preset = load_preset("pulse").scaled(FRAME[1])
    quiet = preset.caption.quiet
    block = build_plan(register_edl("top"), "v.mp4", cfg=cfg()).blocks[0]

    assert block.position == "top" and block.quiet is True
    assert len(block.lines) == 4, [[w.text for w in line] for line in block.lines]
    assert len(block.lines) <= quiet.max_lines
    assert all(w.size == float(quiet.font_size) for w in block.words), "кегль один на весь блок"
    assert glyph_share(block) == pytest.approx(0.015, abs=0.002), "эталонные ~1.5 % высоты кадра"
    assert quiet.align == "left"
    assert block.rect is not None
    # Блок стоит у ЛЕВОГО края кадра (18 % ширины — единственный измеренный в
    # эталоне левый край, §1.3), а не по центру.
    assert block.rect.x == pytest.approx(FRAME[0] * quiet.edge_ratio)
    assert block.rect.x < (FRAME[0] - block.rect.width) / 2
    assert block.rect.right <= FRAME[0] + 1e-6


def test_a_bottom_block_still_renders_as_a_poster():
    """Нижний блок остаётся плакатом: крупно, по центру, 1-2 строки, с золотом.

    Текст здесь четырёхсловный, а не эталонный шестисловный: плакат длиннее пяти
    слов контракт не допускает вовсе (`edl.BLOCK_POSTER_MAX_WORDS`) — шесть слов
    бывают только у тихого блока, и набранные кеглем плаката они ужимаются
    переносом до 3.7 % высоты, то есть замер про регистр перестал бы быть
    замером про регистр.
    """
    preset = load_preset("pulse").scaled(FRAME[1])
    plan = build_plan(register_edl("bottom", "Я думаю что каждый".split(), accent=1), "v.mp4", cfg=cfg())
    block = plan.blocks[0]

    assert block.position == "bottom" and block.quiet is False
    assert len(block.lines) <= preset.caption.max_lines == 2
    assert [w.emphasis for w in block.words].count("accent") == 1, "золото плаката на месте"
    # Эталонный плакат — до 6.2 % высоты кадра (§1.1); у нас 6.37 %.
    assert glyph_share(block) > 0.05, "плакат набран кеглем плаката"
    assert max(w.size or 0 for w in block.words) > float(preset.caption.quiet.font_size) * 2
    assert block.rect is not None
    # Плакат центрирован по кадру — выключка и место блока не одно и то же.
    assert block.rect.center_x == pytest.approx(FRAME[0] / 2)


def test_the_quiet_register_has_no_emphasis_at_all():
    """У тихого блока выделения нет вовсе: ни золота, ни крика, ни крупного слова.

    Гасится оно в САМИХ словах, а не подменой цвета в рендерере: тогда и кегль,
    и поворот крика, и цвет отпадают разом и у ASS, и у Remotion. В эталоне
    верхний кластер не содержит ни одного выделенного слова — а EDL настоящего
    прогона ставит акцент и в верхние блоки тоже.
    """
    words = "Я думаю что каждый".split()  # верхний блок настоящего EDL, акцент на «думаю»
    quiet = build_plan(register_edl("top", words, accent=1), "v.mp4", cfg=cfg()).blocks[0]
    poster = build_plan(register_edl("bottom", words, accent=1), "v.mp4", cfg=cfg()).blocks[0]

    assert [w.emphasis for w in quiet.words] == ["none"] * len(words)
    assert len({w.size for w in quiet.words}) == 1, "увеличенного слова в тихом блоке нет"
    # Тот же акцент в плакате живёт и крупнее соседей — регистр, а не потеря поля.
    assert [w.emphasis for w in poster.words].count("accent") == 1
    assert len({w.size for w in poster.words}) > 1


@pytest.mark.parametrize("position", ["top", "bottom"])
@pytest.mark.parametrize("anchor", ["top_left", "top_right", "bottom_right", "center"])
def test_the_picture_clears_the_block_in_both_registers(position: str, anchor: str):
    """DoD Ф7 в обоих регистрах: картинка ни разу не пересекает прямоугольник блока.

    Регистр меняет прямоугольник целиком — тихий вдвое ниже, вчетверо уже и
    стоит у левого края, — поэтому гарантия обязана проверяться на обоих.
    """
    plan = build_plan(register_edl(position, anchor=anchor), "v.mp4", cfg=cfg())
    caption = plan.blocks[0].rect
    picture = plan.broll[0].rect
    assert isinstance(caption, Rect) and isinstance(picture, Rect)
    if picture.width >= plan.width:
        pytest.skip("места под карточку не осталось — осознанный полный кадр")
    assert not caption.overlaps(picture), (position, anchor)


def test_the_picture_yields_to_every_register_the_block_might_take():
    """Картинка уступает блоку в КАЖДОЙ позиции, какую он может занять.

    Это тот самый круг, из-за которого раскладка и считалась один раз на все
    позиции: «auto» уклоняется от картинки, картинка считается после субтитров и
    уступает всем его позициям сразу. Со вторым регистром раскладка от позиции
    зависит, и уступать картинка обязана НАСТОЯЩИМ прямоугольникам — тихому
    наверху и плакатному внизу, а не одному из них, растянутому на оба места.

    Проверяется это измеримо: полоса под карточку начинается сразу под ТИХИМ
    верхним прямоугольником. Посчитай верх плакатом — и карточка уехала бы на
    290 px ниже, в полосу, которую текст никогда не занимает.
    """
    preset = load_preset("pulse").model_copy(deep=True)
    # Пришпиленному пресету «auto» выбора не даёт (`caption_positions`), а нам
    # нужен блок, который может встать и наверху, и внизу.
    preset.caption.pinned = False
    scaled = preset.scaled(FRAME[1])

    auto = build_plan(
        register_edl("auto", anchor="top_right"), "v.mp4", preset=preset, cfg=cfg()
    )
    at_top = build_plan(register_edl("top"), "v.mp4", preset=preset, cfg=cfg()).blocks[0]
    at_bottom = build_plan(register_edl("bottom"), "v.mp4", preset=preset, cfg=cfg()).blocks[0]

    picture = auto.broll[0].rect
    assert isinstance(picture, Rect)
    assert at_top.quiet is True and at_bottom.quiet is False
    for rect in (at_top.rect, at_bottom.rect):
        assert isinstance(rect, Rect)
        assert not picture.overlaps(rect)
    gap = FRAME[0] * scaled.geometry.broll_margin_ratio
    assert picture.y == pytest.approx(at_top.rect.y + at_top.rect.height + gap, abs=1.0)


@pytest.mark.parametrize("name", ["glass", "bold"])
def test_themes_without_a_second_register_render_as_before(name: str):
    """Регистр — ДАННЫЕ темы: у glass и bold его нет, и верх у них прежний.

    Наверху блок должен выйти ровно тем же, чем внизу, — тем же кеглем, с тем же
    золотом и центрированным. Иначе второй регистр протёк бы в темы, которые
    его не просили.
    """
    preset = load_preset(name)
    assert preset.caption.quiet.enabled is False
    up = build_plan(register_edl("top", accent=2), "v.mp4", preset=preset, cfg=cfg()).blocks[0]
    down = build_plan(
        register_edl("bottom", accent=2), "v.mp4", preset=preset, cfg=cfg()
    ).blocks[0]

    assert up.quiet is False and down.quiet is False
    assert [w.size for w in up.words] == [w.size for w in down.words]
    assert [w.emphasis for w in up.words] == [w.emphasis for w in down.words]
    assert up.rect is not None and down.rect is not None
    assert up.rect.x == pytest.approx(down.rect.x)
    assert up.rect.width == pytest.approx(down.rect.width)


def test_the_declared_register_beats_the_position():
    """Регистр решает ЗАЯВКА режиссёра, а не место блока в кадре.

    Один и тот же четырёхсловный блок наверху: объявленный плакатом остаётся
    крупным и с золотом, объявленный тихим — мелким и без золота. Пока регистр
    выводился из позиции, верхний блок мог быть только вторым, и золотой плакат
    эталона (9.2-12.0 с, центр отдан вставке — §1.4.1, §1.5) был невыразим.
    """
    words = "Я думаю что каждый".split()
    preset = load_preset("pulse").scaled(FRAME[1])
    poster = build_plan(
        register_edl("top", words, accent=1, quiet=False), "v.mp4", cfg=cfg()
    ).blocks[0]
    quiet = build_plan(
        register_edl("top", words, accent=1, quiet=True), "v.mp4", cfg=cfg()
    ).blocks[0]

    assert poster.position == quiet.position == "top"
    assert poster.quiet is False and quiet.quiet is True
    # Плакат наверху: золото на месте, кегль плакатный, блок центрирован.
    assert [w.emphasis for w in poster.words].count("accent") == 1
    assert glyph_share(poster) > 0.05
    assert poster.rect is not None and poster.rect.center_x == pytest.approx(FRAME[0] / 2)
    # Тихий в той же позиции: золота нет, кегль регистра, левый край кадра.
    assert [w.emphasis for w in quiet.words] == ["none"] * len(words)
    assert all(w.size == float(preset.caption.quiet.font_size) for w in quiet.words)
    assert quiet.rect is not None
    assert quiet.rect.x == pytest.approx(FRAME[0] * preset.caption.quiet.edge_ratio)


def test_a_declared_quiet_block_is_ignored_by_themes_without_the_register():
    """Заявка «тихий» — не приказ теме: у glass и bold второго регистра нет.

    Гейт `caption.quiet.enabled` стоит ПЕРЕД заявкой, иначе объявленный регистр
    протёк бы в темы, которые его не просили, и верх у них перестал бы быть
    плакатом.
    """
    for name in ("glass", "bold"):
        preset = load_preset(name)
        up = build_plan(
            register_edl("top", accent=2, quiet=True), "v.mp4", preset=preset, cfg=cfg()
        ).blocks[0]
        down = build_plan(
            register_edl("bottom", accent=2, quiet=False), "v.mp4", preset=preset, cfg=cfg()
        ).blocks[0]
        assert up.quiet is False, name
        assert [w.size for w in up.words] == [w.size for w in down.words], name
        assert [w.emphasis for w in up.words] == [w.emphasis for w in down.words], name


def test_a_list_without_the_register_renders_by_position_as_before():
    """Лист до появления поля рендерится как рендерился: регистр решает позиция.

    Инвариант 4 — сохранённый лист обязан пересобираться в тот же кадр, — и
    поэтому необъявленный регистр разрешается старой приметой, а не длиной: в
    рендере ею была позиция (`Quiet.plays_at`).
    """
    saved = register_edl("top", quiet=None)
    assert saved.captions[0].quiet is None
    block = build_plan(saved, "v.mp4", cfg=cfg()).blocks[0]
    declared = build_plan(register_edl("top", quiet=True), "v.mp4", cfg=cfg()).blocks[0]
    assert block.quiet is True
    assert [w.size for w in block.words] == [w.size for w in declared.words]
    assert block.rect == declared.rect


def test_the_quiet_register_scales_to_the_preview_frame():
    """Кегль и отступ регистра — пиксели кадра 1920, и на превью они уезжают.

    Забыть их в `_PIXEL_FIELDS` значило бы оставить тихий блок в масштабе 1920:
    на кадре 960 он вышел бы вдвое крупнее задуманного, то есть перестал бы быть
    тихим ровно на том кадре, по которому сверяют композицию.
    """
    tall = build_plan(register_edl("top"), "v.mp4", height=1920, width=1080, cfg=cfg())
    short = build_plan(register_edl("top"), "v.mp4", height=960, width=540, cfg=cfg())
    assert short.preset["caption"]["quiet"]["font_size"] < tall.preset["caption"]["quiet"]["font_size"]
    assert short.preset["caption"]["quiet"]["margin"] < tall.preset["caption"]["quiet"]["margin"]
    # Доля кадра у регистра сохраняется: это и значит «композиция не поехала».
    assert glyph_share(short.blocks[0], 960) == pytest.approx(
        glyph_share(tall.blocks[0], 1920), abs=0.002
    )
    assert short.blocks[0].rect.x == pytest.approx(540 * 0.18)  # type: ignore[union-attr]


# ------------------------------------------------------------------ контракт с рендерером


def test_plan_carries_the_keys_the_renderer_reads():
    """Рендерер читает поля Ф5/Ф9 узкими кастами: опечатка в имени тихо всё выключит.

    `render/src/types.ts` — чужой файл, и пока он не обновлён, `tsc` не поймает
    ни `size`, ни `bg_luma`, ни секцию `contrast`: там, где поля нет, TSX молча
    уходит на фолбэк. Значит имена сторожим с этой стороны.
    """
    edl = sample_edl()
    edl.captions[0].bg_luma = 0.42
    data = build_plan(edl, "v.mp4", cfg=cfg()).to_dict()

    block = data["blocks"][0]
    assert "bg_luma" in block and block["bg_luma"] == pytest.approx(0.42)
    assert "rect" in block and block["rect"]
    word = block["lines"][0][0]
    assert "size" in word and word["size"]

    caption = data["preset"]["caption"]
    assert {"max_lines", "min_word_scale"} <= set(caption)
    assert "max_line_chars" not in caption, "поле удалено в Ф5 — оно не должно вернуться"
    contrast = data["preset"]["contrast"]
    assert {"enabled", "min_ratio", "text_dark", "text_light"} <= set(contrast)


def test_plan_carries_the_quiet_register_the_renderer_reads():
    """Имена второго регистра сторожим с этой стороны — в TSX они читаются мягко.

    Рендерер подменяет числа темы числами регистра (`Captions.tsx`,
    `registerPreset`) и узнаёт о регистре по флагу блока. Поле необязательное:
    план, собранный до регистра, его не несёт, и опечатка в имени тихо
    отключила бы весь тихий блок, вернув плакат наверх.
    """
    data = build_plan(register_edl("top"), "v.mp4", cfg=cfg()).to_dict()

    assert data["blocks"][0]["quiet"] is True
    quiet = data["preset"]["caption"]["quiet"]
    assert {
        "enabled", "position", "font_size", "margin", "line_height",
        "align", "edge", "edge_ratio", "panel",
    } <= set(quiet)
    # Переживает запись на диск: перерендер сохранённого плана обязан дать тот
    # же кадр, а не вернуть верхний блок в плакат.
    again = RenderPlan.from_dict(data)
    assert again.blocks[0].quiet is True
    assert again.to_dict() == data


def test_plans_saved_before_the_second_register_read_as_posters():
    """План до §1.4.1 флага не несёт — читается как плакат, а не ломается."""
    data = build_plan(register_edl("top"), "v.mp4", cfg=cfg()).to_dict()
    del data["blocks"][0]["quiet"]
    del data["preset"]["caption"]["quiet"]
    assert RenderPlan.from_dict(data).blocks[0].quiet is False
    assert Preset(**data["preset"]).caption.quiet.enabled is False


def test_plans_saved_before_the_pixel_wrap_still_load():
    """План, сохранённый до Ф5, обязан читаться: в нём лежит удалённый max_line_chars.

    Пресет уезжает в план целиком, а `extra="forbid"` роняет на нём валидацию —
    и в репозитории такой план лежит прямо сейчас (`samples/plan-baseline.json`).
    Удалённое поле выкидывается при чтении, а обратно в модель не возвращается.
    """
    data = build_plan(sample_edl(), "v.mp4", cfg=cfg()).to_dict()
    data["preset"]["caption"]["max_line_chars"] = 15
    preset = Preset(**data["preset"])
    assert "max_line_chars" not in preset.caption.model_dump()
    assert RenderPlan.from_dict(data).blocks

    saved = json.loads(SAMPLE_PLAN.read_text(encoding="utf-8"))
    assert Preset(**saved["preset"]).caption.max_lines >= 1


# ------------------------------------------------------------------ сверка с Chromium


def _as_word(w: PlanWord) -> dict:
    """Слово для линейки: текст, кегль и эмодзи вложенным спаном, как в Captions.tsx."""
    item: dict = {"text": w.text, "size": w.size}
    if w.emoji:
        item["emoji"] = w.emoji
    return item


def _looks_like_a_missing_browser(stderr: str) -> bool:
    """Chromium не скачан — это не расхождение геометрии, а пустая машина."""
    text = stderr.lower()
    return any(
        mark in text
        for mark in ("openbrowser", "ensurebrowser", "browser executable", "chrome headless shell")
    )


MEASURE = Path(__file__).resolve().parents[1] / "render" / "measure.mjs"
FONTS = Path(__file__).resolve().parents[1] / "assets" / "fonts"


def _measure_in_chromium(request: dict) -> dict:
    """Запуск линейки `render/measure.mjs`: JSON на stdin → измерения на stdout.

    «Браузера нет» и «кадр разъехался» — разные новости: на чистой машине
    `openBrowser` падает, потому что Chromium ещё не скачан, и падение теста
    выглядело бы как поломка кода. Первое — пропуск, второе — провал.
    """
    import subprocess

    out = subprocess.run(
        ["node", str(MEASURE)],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(MEASURE.parent),
    )
    if out.returncode != 0:
        if _looks_like_a_missing_browser(out.stderr):
            pytest.skip(f"нет скачанного Chromium: {out.stderr.strip()[-300:]}")
        pytest.fail(f"measure.mjs упал: {out.stderr.strip()[-2000:]}")
    return json.loads(out.stdout)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["glass", "bold", "pulse"])
def test_chromium_draws_the_block_the_plan_promised(name: str):
    """DoD Ф5 целиком: тот же Chromium и тот же шрифт, что в рендере, — допуск 2 px.

    Без этого теста сверка с браузером существует только числом в отчёте, а
    внутри Python любая проверка переноса кругова: и `wrap_block`, и её эталон
    считает одна и та же линейка. Здесь эталон — сам браузер, и меряется не
    только ширина строки, но и коробка блока: ровно на высоте и жила ошибка,
    из-за которой `rect` занижал блок с ужатым словом на 45-164 px.

    Эмодзи в запросе ЕСТЬ: `measure.mjs` рисует их вложенным спаном ровно так,
    как это делает Captions.tsx, и без них сверка не покрывала бы ни ширину
    кластера, ни пробел перед ним. Ширину цветного глифа даёт системный шрифт —
    та же оговорка, что у `EMOJI_ADVANCE_EM`.
    """
    node = __import__("shutil").which("node")
    if node is None or not (MEASURE.parent / "node_modules").exists():
        pytest.skip("нужен node и render/node_modules")

    preset = load_preset(name).scaled(FRAME[1])
    font = FONTS / (preset.font.file or "")
    if not font.exists():
        pytest.skip(f"нет шрифта {font}")

    request: dict = {
        "font": str(font),
        "weight": preset.font.weight,
        "base_size": preset.font.size,
        "letter_spacing": preset.font.letter_spacing,
        "line_height": preset.caption.line_height,
        "items": [],
        "boxes": [],
    }
    expected_lines: dict[str, float] = {}
    expected_boxes: dict[str, tuple[float, float]] = {}
    for bi, block in enumerate(all_blocks()):
        words = plan_words(block, preset)
        lines, widths = wrap_block(words, preset, FRAME[0])
        rect = _rect_of_lines("bottom", lines, widths, preset, FRAME)
        for li, (line, width) in enumerate(zip(lines, widths)):
            key = f"b{bi}l{li}"
            expected_lines[key] = width
            request["items"].append(
                {"id": key, "words": [_as_word(w) for w in line]}
            )
        expected_boxes[f"b{bi}"] = (rect.width, rect.height)
        panel = preset.panel
        request["boxes"].append({
            "id": f"b{bi}",
            "lines": [[_as_word(w) for w in line] for line in lines],
            # `gap` НЕ передаётся: страница считает его формулой Captions.tsx
            # сама. Подставить сюда `line_gap(preset)` значит поставить одно и то
            # же вычисление по обе стороны сверки — ошибка в нём сократилась бы,
            # и тест прошёл бы при разъехавшемся кадре.
            "padding_x": panel.padding_x if panel.enabled else 0,
            "padding_y": panel.padding_y if panel.enabled else 0,
            "border_width": panel.border_width if panel.enabled else 0,
        })

    measured = _measure_in_chromium(request)
    for key, width in expected_lines.items():
        assert measured["widths"][key] == pytest.approx(width, abs=2.0), (name, key)
    for key, (w, h) in expected_boxes.items():
        box = measured["boxes"][key]
        assert box["width"] == pytest.approx(w, abs=2.0), (name, key, "ширина")
        assert box["height"] == pytest.approx(h, abs=2.0), (name, key, "высота")


@pytest.mark.slow
def test_chromium_draws_the_quiet_block_the_plan_promised():
    """Тот же допуск 2 px, но в ТИХОМ регистре (§1.4.1).

    Регистр подменяет кегль, межстрочие, бюджет колонки и панель — то есть
    ровно те числа, которыми меряется строка и считается коробка блока. Сверка
    плаката этого не ловит: у неё другой `base_size`, а от него зависят и пробел
    между словами, и letter-spacing (он объявлен в em на контейнере строки), и
    СТРУТ строчного бокса. То есть без этого теста тихий блок в Chromium не
    проверял никто, а подмена пресета в Python и в `Captions.tsx` могла
    разъехаться молча.

    Отдельный запуск браузера, а не ещё одна параметризация: кегль контейнера на
    странице линейки один, и два регистра в одну страницу не укладываются.
    """
    node = __import__("shutil").which("node")
    if node is None or not (MEASURE.parent / "node_modules").exists():
        pytest.skip("нужен node и render/node_modules")

    theme = load_preset("pulse").scaled(FRAME[1])
    assert theme.caption.quiet.enabled, "тест про тему со вторым регистром"
    preset = theme.caption.quiet.preset(theme)
    font = FONTS / (preset.font.file or "")
    if not font.exists():
        pytest.skip(f"нет шрифта {font}")

    request: dict = {
        "font": str(font),
        "weight": preset.font.weight,
        "base_size": preset.font.size,
        "letter_spacing": preset.font.letter_spacing,
        "line_height": preset.caption.line_height,
        "items": [],
        "boxes": [],
    }
    expected_lines: dict[str, float] = {}
    expected_boxes: dict[str, tuple[float, float]] = {}
    left = theme.caption.quiet.left_x(FRAME[0])
    for bi, block in enumerate(all_blocks()):
        # Выделение гасит сам регистр (`lay_places`), и мерить надо то, что
        # уедет в план: слова без акцента и без крика.
        words = [replace(w, emphasis="none") for w in plan_words(block, preset)]
        lines, widths = wrap_block(words, preset, FRAME[0])
        rect = _rect_of_lines("top", lines, widths, preset, FRAME, left=left)
        assert rect.x == pytest.approx(left), "тихий блок стоит у левого края кадра"
        for li, (line, width) in enumerate(zip(lines, widths)):
            key = f"q{bi}l{li}"
            expected_lines[key] = width
            request["items"].append({"id": key, "words": [_as_word(w) for w in line]})
        expected_boxes[f"q{bi}"] = (rect.width, rect.height)
        panel = preset.panel
        request["boxes"].append({
            "id": f"q{bi}",
            "lines": [[_as_word(w) for w in line] for line in lines],
            "padding_x": panel.padding_x if panel.enabled else 0,
            "padding_y": panel.padding_y if panel.enabled else 0,
            "border_width": panel.border_width if panel.enabled else 0,
        })

    measured = _measure_in_chromium(request)
    for key, width in expected_lines.items():
        assert measured["widths"][key] == pytest.approx(width, abs=2.0), key
    for key, (w, h) in expected_boxes.items():
        box = measured["boxes"][key]
        assert box["width"] == pytest.approx(w, abs=2.0), (key, "ширина")
        assert box["height"] == pytest.approx(h, abs=2.0), (key, "высота")


def test_the_panel_flag_survives_the_round_trip_to_the_plan():
    """Флаг плашки обязан доехать до рендерера — и вернуться из сохранённого листа.

    Инвариант 4 (перерендер того же EDL даёт то же видео) держится, только пока
    решение живёт в данных. Плашка — решение режиссёрское, вычислить его заново
    нельзя: в эталоне оно не выводится ни из яркости фона, ни из его суетливости.
    """
    edl = sample_edl()
    edl.captions[0].panel = True
    plan = build_plan(edl, "v.mp4", cfg=cfg())
    assert [b.panel for b in plan.blocks] == [True, False]

    again = RenderPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert [b.panel for b in again.blocks] == [True, False]


def test_a_panelled_block_takes_the_same_room_as_a_bare_one():
    """Место под плашку зарезервировано у ВСЕХ блоков одинаково.

    Иначе текст прыгал бы на высоту паддингов между соседними блоками, а Python
    и Chromium разошлись бы в замере прямоугольника: сверки в `test_geometry`
    меряют одну геометрию на всю тему, а не по блокам. Поэтому флаг меняет
    только вид — заливку, размытие и цвет рамки, — но не раскладку.
    """
    bare = build_plan(sample_edl(), "v.mp4", cfg=cfg())
    edl = sample_edl()
    edl.captions[0].panel = True
    glass = build_plan(edl, "v.mp4", cfg=cfg())
    for a, b in zip(bare.blocks, glass.blocks):
        assert a.rect == b.rect


# ------------------------------------------------------------------ титр (STYLE_PLAN §5)

# Заголовок длиной с эталонный титр: подпись в одну строку и короткое слово под
# ней. Текст СВОЙ — титр эталона называет чужой продукт, и копировать там нечего:
# наш титр набирает заголовок самого ролика (`EDL.title`).
CARD_TITLE = "Ролики из сырого видео"
# Полоса титра в эталоне и высота глифов его большого слова — доли высоты кадра.
CARD_BAND = (0.170, 0.277)
CARD_GLYPHS = 0.101


def card_edl(
    title: str | None = CARD_TITLE,
    *,
    speech: list[str] | None = None,
    position: str = "bottom",
    quiet: bool | None = False,
    anchor: str | None = None,
    preset: str = "pulse",
    source_duration: float = 12.0,
) -> EDL:
    """Ролик, который начинает говорить сразу, плюс заголовок для титра."""
    spoken = speech or ["так", "вот", "смотрите"]
    words = [
        Word(idx=i, text=t, start=0.15 + i * 0.45, end=0.15 + i * 0.45 + 0.4)
        for i, t in enumerate(spoken)
    ]
    return EDL(
        source_duration=source_duration,
        words=words,
        cuts=[Cut(src_in=0.0, src_out=source_duration, reason="keep")],
        captions=[
            CaptionBlock(
                words=[CaptionWord(word_idx=i) for i in range(len(words))],
                position=position,
                quiet=quiet,
            )
        ],
        broll=(
            [Broll(word_idx=0, duration=3.0, query="q", anim=BrollAnim.POP_IN, anchor=anchor)]
            if anchor
            else []
        ),
        preset=preset,
        title=title,
    )


def card_of(plan: RenderPlan):
    """Блок-титр плана или None."""
    return next((b for b in plan.blocks if b.title_card), None)


def test_the_title_card_is_the_largest_type_in_the_reel():
    """Титр эталона: полоса 17.0-27.7 % высоты кадра, глифы большого слова 10.1 %.

    Потолок плаката в том же ролике — 7.5 %, то есть титр крупнее ЛЮБОГО
    субтитра, и это единственное, чем он является. Полоса меряется по
    прямоугольнику блока, а глифы — `cap_height`, как все замеры §1.1.
    """
    preset = load_preset("pulse").scaled(FRAME[1])
    card = preset.caption.title_card
    plan = build_plan(card_edl(), "v.mp4", cfg=cfg())
    block = card_of(plan)

    assert block is not None and block.position == "top"
    assert block.quiet is False, "титр — плакат, а не тихий регистр"
    assert (block.start, block.end) == (card.start, card.start + card.duration)
    assert len(block.lines) == 2, [[w.text for w in line] for line in block.lines]
    assert [w.text for w in block.lines[1]] == ["видео"], "крупным идёт последнее слово"

    big = max(w.size or 0.0 for w in block.words)
    assert big == pytest.approx(card.font_size * card.accent_scale)
    assert glyph_share(block) == pytest.approx(CARD_GLYPHS, abs=0.004)
    # Крупнее любого субтитра ролика — иначе это не титр, а второй субтитр.
    speech = [b for b in plan.blocks if not b.title_card]
    assert speech and big > max(w.size or 0.0 for b in speech for w in b.words)
    assert big > preset.font.size * preset.caption.fill_max_scale

    assert block.rect is not None
    # Прямоугольник — это БОКС строк, а глифы при межстрочии 0.5 торчат из него
    # на половину полулидинга в обе стороны (см. `geometry.line_box_height`),
    # поэтому допуск здесь шире замера: сами пиксели меряются на ролике.
    assert block.rect.y / FRAME[1] == pytest.approx(CARD_BAND[0], abs=0.006)
    assert block.rect.bottom / FRAME[1] == pytest.approx(CARD_BAND[1], abs=0.006)
    assert block.rect.center_x == pytest.approx(FRAME[0] / 2), "титр центрирован"


def test_the_title_card_builds_word_by_word():
    """Титр набирается по слову, как обычный блок, и успевает собраться.

    Замер эталона: маска растёт 324 → 664 → 1049 → 1692 пикселей с шагом 0.1 с,
    место под ещё не пришедшее слово зарезервировано заранее (§1.3). У нас это
    обычные ghost-слова, поэтому проверяем ровно тайминги: шаг из темы и
    последнее слово не позже середины жизни титра.
    """
    card = load_preset("pulse").caption.title_card
    short = card_of(build_plan(card_edl(), "v.mp4", cfg=cfg()))
    assert short is not None
    starts = [w.start for w in short.words]
    assert starts[0] == card.start
    assert all(
        b - a == pytest.approx(card.word_step, abs=1e-3) for a, b in zip(starts, starts[1:])
    )
    assert all(w.end == short.end for w in short.words), "слова титра не гаснут поодиночке"

    long_title = card_of(
        build_plan(
            card_edl("Один два три четыре пять шесть семь восемь девять"), "v.mp4", cfg=cfg()
        )
    )
    assert long_title is not None
    middle = long_title.start + (long_title.end - long_title.start) / 2
    assert max(w.start for w in long_title.words) <= middle + 1e-6


def test_the_card_text_lives_in_the_edl_and_survives_a_rebuild():
    """Перерендер сохранённого листа обязан дать тот же титр (инвариант 4).

    Заголовок лежит в самом листе, а не в метаданных задачи: перерендер
    получает EDL и больше ничего. Лист без заголовка — лист до появления поля
    или прогон без LLM — рендерится как раньше, без титра.
    """
    saved = json.loads(card_edl().model_dump_json())
    assert saved["title"] == CARD_TITLE
    again = card_of(build_plan(EDL(**saved), "v.mp4", cfg=cfg()))
    first = card_of(build_plan(card_edl(), "v.mp4", cfg=cfg()))
    assert again is not None and first is not None
    assert [[w.text for w in line] for line in again.lines] == [
        [w.text for w in line] for line in first.lines
    ]
    assert again.rect == first.rect

    bare = build_plan(card_edl(None), "v.mp4", cfg=cfg())
    assert card_of(bare) is None
    assert [b.title_card for b in bare.blocks] == [False]


@pytest.mark.parametrize("name", ["glass", "bold"])
def test_themes_without_a_card_never_draw_one(name: str):
    """Титр — ДАННЫЕ темы: у glass и bold его нет, и заголовок их не меняет.

    Сравниваем планы с заголовком и без: у темы без титра они обязаны совпадать
    целиком, иначе третий регистр протёк бы в темы, которые его не просили.
    """
    preset = load_preset(name)
    assert preset.caption.title_card.enabled is False
    with_title = build_plan(card_edl(preset=name), "v.mp4", preset=preset, cfg=cfg())
    without = build_plan(card_edl(None, preset=name), "v.mp4", preset=preset, cfg=cfg())
    assert card_of(with_title) is None
    assert with_title.to_dict() == without.to_dict()


def test_the_card_yields_to_a_block_in_its_band():
    """Полосу титра занял субтитр — титра нет.

    Проверяется ПРЯМОУГОЛЬНИКАМИ, как и всё остальное в этом модуле: плакат
    живёт внизу кадра и титру не мешает, а четырёхстрочный тихий блок достаёт до
    его полосы. Запретить титру жить, пока кто-то говорит, нельзя: речь у нас
    начинается с первых кадров (`cut_pad_before` 0.15 с), и титра не было бы
    никогда.
    """
    talking = card_edl()  # нижний плакат с первого кадра
    assert card_of(build_plan(talking, "v.mp4", cfg=cfg())) is not None

    upstairs = card_edl(
        speech=QUIET_LINE, position="top", quiet=True, source_duration=12.0
    )
    plan = build_plan(upstairs, "v.mp4", cfg=cfg())
    quiet_block = next(b for b in plan.blocks if not b.title_card)
    assert quiet_block.quiet is True and quiet_block.rect is not None
    assert card_of(plan) is None, "титр и тихий блок в одной полосе"


@pytest.mark.parametrize("anchor", ["fullscreen", "top_right"])
def test_the_card_yields_to_a_picture_in_its_band(anchor: str):
    """Полосу титра заняла картинка — титра нет. Двигать её титр не вправе."""
    with_pic = build_plan(card_edl(anchor=anchor), "v.mp4", cfg=cfg())
    assert card_of(with_pic) is None

    # И не двигает: картинка встаёт туда же, что и в ролике без заголовка.
    without = build_plan(card_edl(None, anchor=anchor), "v.mp4", cfg=cfg())
    assert [x.rect for x in with_pic.broll] == [x.rect for x in without.broll]


def test_a_title_too_big_for_the_card_drops_it():
    """Титр, ужатый до плакатного кегля, не рисуется вовсе.

    Длинный заголовок перенос ужимает целиком (`wrap_block`), и в какой-то
    момент большое слово перестаёт быть крупнее любого субтитра — то есть титр
    перестаёт быть титром. Половинчатый титр хуже его отсутствия.
    """
    preset = load_preset("pulse").scaled(FRAME[1])
    ceiling = preset.font.size * preset.caption.fill_max_scale
    long_title = (
        "Как за одну короткую минуту собрать вертикальный ролик "
        "из совершенно сырого исходного видео"
    )
    assert card_of(build_plan(card_edl(long_title), "v.mp4", cfg=cfg())) is None

    medium = card_of(build_plan(card_edl("Монтаж за минуту"), "v.mp4", cfg=cfg()))
    assert medium is not None
    assert max(w.size or 0.0 for w in medium.words) > ceiling


def test_the_card_is_dropped_when_the_reel_is_shorter_than_it():
    """Ролик кончается раньше титра — титра нет: обрезанный титр читается как сбой."""
    card = load_preset("pulse").caption.title_card
    short = card_edl(source_duration=card.start + card.duration - 0.2)
    assert card_of(build_plan(short, "v.mp4", cfg=cfg())) is None


def test_the_card_is_not_cut_short_by_the_next_block():
    """Титр живёт свои 1.1 с, даже если субтитр внизу начался раньше.

    Обрезка соседей разводит блоки, стоящие в одном месте кадра друг за другом;
    титр стоит в своей полосе, и обрывать его началом нижнего плаката не за что.
    """
    plan = build_plan(card_edl(), "v.mp4", cfg=cfg())
    block = card_of(plan)
    speech = next(b for b in plan.blocks if not b.title_card)
    assert block is not None
    assert speech.start < block.start, "ролик заговорил раньше титра"
    assert block.end - block.start == pytest.approx(
        load_preset("pulse").caption.title_card.duration
    )


def test_the_card_scales_to_the_preview_frame():
    """Кегль и отступ титра — пиксели кадра 1920, на превью они уезжают вместе с ним."""
    tall = build_plan(card_edl(), "v.mp4", height=1920, width=1080, cfg=cfg())
    short = build_plan(card_edl(), "v.mp4", height=960, width=540, cfg=cfg())
    tall_card, short_card = card_of(tall), card_of(short)
    assert tall_card is not None and short_card is not None
    assert (
        short.preset["caption"]["title_card"]["font_size"]
        < tall.preset["caption"]["title_card"]["font_size"]
    )
    assert glyph_share(short_card, 960) == pytest.approx(glyph_share(tall_card, 1920), abs=0.004)
    assert short_card.rect is not None and tall_card.rect is not None
    assert short_card.rect.y / 960 == pytest.approx(tall_card.rect.y / 1920, abs=0.004)


def test_plan_carries_the_card_the_renderer_reads():
    """Имена третьего регистра сторожим здесь: в TSX они читаются мягким кастом.

    Рендерер узнаёт титр по флагу блока и подменяет числа темы числами регистра
    (`Captions.tsx`, `registerPreset`). Опечатка в имени тихо вернула бы титру
    отступ и межстрочие плаката, то есть увела бы его полосу мимо замера.
    """
    data = build_plan(card_edl(), "v.mp4", cfg=cfg()).to_dict()
    block = data["blocks"][0]
    assert block["title_card"] is True
    assert all(b["title_card"] is False for b in data["blocks"][1:])
    card = data["preset"]["caption"]["title_card"]
    assert {
        "enabled", "start", "duration", "word_step", "font_size", "margin",
        "accent_scale", "line_height", "max_lines", "panel",
    } <= set(card)

    again = RenderPlan.from_dict(data)
    assert again.blocks[0].title_card is True
    assert again.to_dict() == data

    # План до появления титра поля не несёт — читается как обычный блок.
    del data["blocks"][0]["title_card"]
    del data["preset"]["caption"]["title_card"]
    assert RenderPlan.from_dict(data).blocks[0].title_card is False
    assert Preset(**data["preset"]).caption.title_card.enabled is False


async def test_the_pipeline_puts_the_title_in_the_edl_before_the_plan(tmp_path, monkeypatch):
    """Заголовок обязан попасть в лист ДО сборки плана, иначе титра не будет.

    Раньше `make_metadata` вызывался ПОСЛЕ рендера, и заголовок выбрасывался:
    `mark_done` его не сохранял, читателей у него не было. Теперь он считается
    сразу после режиссёра и ложится в EDL — тот самый лист, который уезжает в
    `jobs.edl` и из которого перерендер собирает ролик заново.

    Прогон обрывается на кропе: всё, что после, к проводке отношения не имеет и
    стоит минуты.
    """
    import shutil as _shutil

    from src.pipeline.asr import ASRResult
    from src.pipeline.job import process_video
    from src.pipeline.probe import MediaInfo

    edl = card_edl(None)  # лист без заголовка: его обязан проставить прогон
    src = tmp_path / "src.mp4"
    src.write_bytes(b"")
    settings = Settings(_env_file=None, default_preset="pulse", audio_post_enabled=False)

    class Reached(RuntimeError):
        pass

    async def fake_direct(*a, **k):
        return edl

    async def fake_metadata(*a, **k):
        return {"title": CARD_TITLE, "description": "", "hashtags": []}

    def stop(*a, **k):
        raise Reached

    monkeypatch.setattr(
        "src.pipeline.job.probe",
        lambda *a, **k: MediaInfo(
            path="—", duration=12.0, width=1080, height=1920, fps=30.0, has_audio=True,
            audio_channels=2, audio_rate=48000, rotation=0, video_codec="h264", bitrate=0,
        ),
    )
    monkeypatch.setattr("src.pipeline.job.is_playable", lambda *a, **k: True)
    monkeypatch.setattr(
        "src.pipeline.job.apply_cuts", lambda e, s, d, **k: _shutil.copy2(s, d)
    )
    monkeypatch.setattr(
        "src.pipeline.job.transcribe",
        lambda *a, **k: ASRResult(words=list(edl.words), language="ru", backend="test", model="—"),
    )
    monkeypatch.setattr("src.pipeline.job.direct", fake_direct)
    monkeypatch.setattr("src.pipeline.job.make_metadata", fake_metadata)
    monkeypatch.setattr("src.pipeline.job.autocrop", stop)

    with pytest.raises(Reached):
        await process_video(src, tmp_path / "work", preset="pulse", cfg=settings)

    assert edl.title == CARD_TITLE, "заголовок не доехал до листа к моменту кропа"
    assert card_of(build_plan(edl, "v.mp4", cfg=cfg())) is not None
