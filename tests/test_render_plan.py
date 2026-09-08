"""Тесты плана рендера: пересчёт времени, перенос строк, разъезд с b-roll, геометрия."""

from __future__ import annotations

import json
from pathlib import Path

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
    Rect,
    broll_rect,
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

    Позиции блоков видит только сборка плана, и без них `broll_rect` берёт свою
    лестницу по умолчанию: у glass в кадре 1080×1920 места на все три полосы нет,
    и лестница отбрасывает `center` — то есть карточка получала право встать
    ровно туда, куда режиссёр только что поставил блок. Здесь план обязан
    передать позиции вниз и получить либо карточку мимо текста, либо полный кадр.
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
    plan = build_plan(broll_edl("bottom_right"), "v.mp4", cfg=cfg())
    preset = load_preset("glass")
    assert plan.broll[0].rect == broll_rect("bottom_right", preset, (plan.width, plan.height))


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


@pytest.mark.parametrize("name", ["glass", "bold"])
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


@pytest.mark.parametrize("name", ["glass", "bold"])
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


@pytest.mark.parametrize("name", ["glass", "bold"])
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


@pytest.mark.parametrize("name", ["glass", "bold"])
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
    """Акцент приезжает в план кеглем: рендереру нечего домножать."""
    preset = load_preset("glass").scaled(FRAME[1])
    words = plan_words([("тише", "none", None), ("громче", "accent", None)], preset)
    wrap_block(words, preset, FRAME[0])
    assert words[0].size == pytest.approx(preset.font.size)
    assert words[1].size == pytest.approx(preset.font.size * preset.animation.accent_scale)


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


def _luminance(color: str) -> float:
    """Относительная яркость цвета по WCAG — та же формула, что в Captions.tsx."""
    channels = [c / 255 for c in hex_to_rgb(color)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: float, b: float) -> float:
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


@pytest.mark.parametrize("name", ["glass", "bold"])
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


MEASURE = Path(__file__).resolve().parents[1] / "render" / "measure.mjs"
FONTS = Path(__file__).resolve().parents[1] / "assets" / "fonts"


def _measure_in_chromium(request: dict) -> dict:
    """Запуск линейки `render/measure.mjs`: JSON на stdin → измерения на stdout."""
    import subprocess

    out = subprocess.run(
        ["node", str(MEASURE)],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(MEASURE.parent),
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["glass", "bold"])
def test_chromium_draws_the_block_the_plan_promised(name: str):
    """DoD Ф5 целиком: тот же Chromium и тот же шрифт, что в рендере, — допуск 2 px.

    Без этого теста сверка с браузером существует только числом в отчёте, а
    внутри Python любая проверка переноса кругова: и `wrap_block`, и её эталон
    считает одна и та же линейка. Здесь эталон — сам браузер, и меряется не
    только ширина строки, но и коробка блока: ровно на высоте и жила ошибка,
    из-за которой `rect` занижал блок с ужатым словом на 45-164 px.

    Эмодзи в запрос не идут: их рисует системный шрифт, и `measure.mjs` не
    моделирует вложенный спан.
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
        if any(w.emoji for w in words):
            continue
        rect = _rect_of_lines("bottom", lines, widths, preset, FRAME)
        for li, (line, width) in enumerate(zip(lines, widths)):
            key = f"b{bi}l{li}"
            expected_lines[key] = width
            request["items"].append(
                {"id": key, "words": [{"text": w.text, "size": w.size} for w in line]}
            )
        expected_boxes[f"b{bi}"] = (rect.width, rect.height)
        panel = preset.panel
        request["boxes"].append({
            "id": f"b{bi}",
            "lines": [[{"text": w.text, "size": w.size} for w in line] for line in lines],
            "gap": line_gap(preset),
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
