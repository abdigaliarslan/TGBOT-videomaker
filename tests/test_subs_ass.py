"""Тесты ASS-генератора (M3). Чистая функция — тестируется без ffmpeg."""

from __future__ import annotations

import re

import pytest

from src.core.edl import EDL, CaptionBlock, CaptionWord, Cut, Emphasis, Word
from src.core.preset import load_preset, to_ass_alpha, to_ass_colour
from src.core.render_plan import build_plan, wrap_words, PlanWord
from src.pipeline.subs_ass import ass_time, build_ass, escape


def edl_with_accent() -> EDL:
    words = [
        Word(idx=0, text="это", start=0.5, end=0.8),
        Word(idx=1, text="очень", start=0.85, end=1.2),
        Word(idx=2, text="важно", start=1.25, end=1.8),
        Word(idx=3, text="запомни", start=2.0, end=2.6),
    ]
    return EDL(
        source_duration=4.0,
        words=words,
        cuts=[Cut(src_in=0.0, src_out=4.0, reason="keep")],
        captions=[
            CaptionBlock(
                words=[
                    CaptionWord(word_idx=0),
                    CaptionWord(word_idx=1),
                    CaptionWord(word_idx=2, emphasis=Emphasis.ACCENT),
                ]
            ),
            CaptionBlock(words=[CaptionWord(word_idx=3, emphasis=Emphasis.SHOUT)]),
        ],
        preset="glass",
    )


def plan():
    return build_plan(edl_with_accent(), "in.mp4", width=1080, height=1920, fps=30)


# ------------------------------------------------------------------ время


@pytest.mark.parametrize(
    "seconds,expected",
    [(0.0, "0:00:00.00"), (1.5, "0:00:01.50"), (61.234, "0:01:01.23"), (3661.0, "1:01:01.00")],
)
def test_ass_time(seconds, expected):
    assert ass_time(seconds) == expected


def test_ass_time_never_negative():
    assert ass_time(-5.0) == "0:00:00.00"


# ------------------------------------------------------------------ экранирование


def test_escape_braces_and_newlines():
    assert "{" not in escape("текст {со скобками}")
    assert "}" not in escape("текст {со скобками}")
    assert "\n" not in escape("две\nстроки")


# ------------------------------------------------------------------ структура


def test_header_matches_frame_size():
    ass = build_ass(plan())
    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass
    assert "Montserrat ExtraBold" in ass


def test_one_dialogue_per_active_word():
    ass = build_ass(plan())
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    # 3 слова в первом блоке + 1 во втором
    assert len(dialogues) == 4


def test_dialogues_are_ordered_and_non_overlapping():
    ass = build_ass(plan())
    times = [
        (m.group(1), m.group(2))
        for m in re.finditer(r"Dialogue: 0,([\d:.]+),([\d:.]+),", ass)
    ]
    assert times == sorted(times)
    for (_, end), (start, _) in zip(times, times[1:]):
        assert end <= start


def test_word_states_present_in_one_dialogue():
    """В строке для второго слова первое — past, третье — ghost."""
    ass = build_ass(plan())
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    second = dialogues[1]
    preset = load_preset("glass")
    assert to_ass_alpha(preset.colors.past_opacity) in second
    assert to_ass_alpha(preset.colors.ghost_opacity) in second
    assert to_ass_alpha(1.0) in second  # активное слово непрозрачно


def test_accent_word_uses_accent_colour_when_active():
    ass = build_ass(plan())
    preset = load_preset("glass")
    accent = to_ass_colour(preset.colors.accent)
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    # «важно» активно в третьей реплике
    assert accent in dialogues[2]
    assert accent not in dialogues[0]


def test_shout_word_gets_rotation_and_shout_colour():
    ass = build_ass(plan())
    preset = load_preset("glass")
    last = [line for line in ass.splitlines() if line.startswith("Dialogue:")][-1]
    assert to_ass_colour(preset.colors.shout) in last
    assert "\\frz" in last


def test_pop_animation_present_on_active_word():
    ass = build_ass(plan())
    assert "\\t(0," in ass


# ------------------------------------------------------------------ мягкая тень


def style_fields() -> list[str]:
    """Единственная строка Style: → список полей по формату V4+."""
    line = next(x for x in build_ass(plan()).splitlines() if x.startswith("Style:"))
    return line.split(":", 1)[1].strip().split(",")


def test_outline_is_a_halo_not_a_contour():
    """Ф1: чёрный кант убран, ширина обводки втрое меньше кегля × 0.07."""
    size = load_preset("glass").font.size
    outline = float(style_fields()[16])
    assert 0 < outline <= size * 0.04


def test_outline_is_blurred_into_a_shadow():
    """Без \\blur тонкая обводка осталась бы просто тонким кантом."""
    ass = build_ass(plan())
    assert "\\blur" in ass
    shadow = float(style_fields()[17])
    assert shadow > 0  # тень никуда не делась: она и держит читаемость


def test_position_tag_from_block():
    ass = build_ass(plan())
    assert "\\an2" in ass  # пресет glass ставит субтитры внизу


def test_preset_swap_changes_output_without_touching_code():
    """§0.3: смена пресета меняет вид, не трогая генератор."""
    from src.core.preset import load_preset as lp

    glass = build_ass(build_plan(edl_with_accent(), "in.mp4", preset=lp("glass")))
    bold = build_ass(build_plan(edl_with_accent(), "in.mp4", preset=lp("bold")))
    assert glass != bold
    assert "\\an2" in glass and "\\an5" in bold  # bold ставит по центру
    assert "ВАЖНО" in bold and "ВАЖНО" not in glass  # bold — uppercase


# ------------------------------------------------------------------ перенос строк


def test_wrap_respects_char_limit():
    words = [PlanWord(text=t, start=0, end=1) for t in ["один", "два", "три", "четыре"]]
    lines = wrap_words(words, max_chars=9)
    assert all(sum(len(w.text) for w in line) + len(line) - 1 <= 9 or len(line) == 1 for line in lines)
    assert [w.text for line in lines for w in line] == ["один", "два", "три", "четыре"]


def test_wrap_never_exceeds_max_lines():
    words = [PlanWord(text="слово", start=0, end=1) for _ in range(12)]
    lines = wrap_words(words, max_chars=6, max_lines=3)
    assert len(lines) <= 3


def test_wrap_keeps_all_words():
    words = [PlanWord(text=f"w{i}", start=0, end=1) for i in range(7)]
    lines = wrap_words(words, max_chars=5)
    assert sum(len(line) for line in lines) == 7
