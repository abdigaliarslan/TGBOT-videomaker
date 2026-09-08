"""Тесты ASS-генератора (M3). Чистая функция — тестируется без ffmpeg."""

from __future__ import annotations

import re

import pytest

from src.core.edl import EDL, CaptionBlock, CaptionWord, Cut, Emphasis, Word
from src.core.preset import Preset, load_preset, to_ass_alpha, to_ass_colour
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


# ------------------------------------------------------------------ тайминг (Ф8)


def dialogues(p) -> list[str]:
    return [line for line in build_ass(p).splitlines() if line.startswith("Dialogue:")]


def _dialogue_ms(line: str) -> int:
    """Длительность реплики в миллисекундах по её собственным меткам времени."""
    m = re.match(r"Dialogue: 0,([\d:.]+),([\d:.]+),", line)
    assert m, line

    def parse(stamp: str) -> float:
        h, mm, rest = stamp.split(":")
        return int(h) * 3600 + int(mm) * 60 + float(rest)

    return round((parse(m.group(2)) - parse(m.group(1))) * 1000)


def _pop_end_ms(line: str) -> int | None:
    """Конец последней фазы «пшика» или None, если анимации в реплике нет."""
    ends = [int(m.group(2)) for m in re.finditer(r"\\t\((\d+),(\d+),", line)]
    return max(ends) if ends else None


@pytest.mark.parametrize("fps", [24, 30, 60])
@pytest.mark.parametrize("source", ["synthetic", "baseline"])
def test_pop_either_lasts_eight_frames_or_is_not_drawn(fps, source):
    """Пол в восемь кадров — закон обоих рендереров, но мерить его надо по РЕПЛИКЕ.

    Теги `\\t` отсчитываются от начала реплики, а реплика живёт от слова i до
    слова i+1: анимацию ограничивает сверху речь, а не пресет. Пол, взятый из
    пресета как есть, этого не знал — на 24 fps он просит 334 мс, и 15 реплик
    эталонного ролика из 35 короче, то есть «пшик» обрывался на промежуточном
    `\\fscx` и следующая реплика возвращала кегль скачком.

    Поэтому проверяется поведение, а не число в теге: либо анимации нет вовсе,
    либо она И длится не меньше восьми кадров, И укладывается в свою реплику.
    Эталонный EDL здесь обязателен: на синтетике таких коротких реплик нет, и
    тест зеленел бы при анимации, обрезанной до одного кадра.
    """
    edl = (
        edl_with_accent()
        if source == "synthetic"
        else EDL(**{k: v for k, v in _baseline().items() if k != "camera"})
    )
    p = build_plan(edl, "in.mp4", fps=fps, width=1080, height=1920)
    lines = dialogues(p)
    assert lines
    drawn = 0
    for line in lines:
        end = _pop_end_ms(line)
        if end is None:
            continue
        drawn += 1
        window = _dialogue_ms(line)
        assert end <= window, f"«пшик» {end} мс длиннее своей реплики {window} мс: {line}"
        assert end / 1000 * fps >= 8 - 1e-6, f"«пшик» короче восьми кадров: {line}"
    assert drawn, "ни в одной реплике не осталось анимации"


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_pop_starts_at_the_word_size_and_returns_to_it(fps):
    """«Пшик» транзиентный: свой кегль → пик → свой кегль.

    Раньше слово ПОЯВЛЯЛОСЬ уже увеличенным и только съезжало вниз — то есть
    скачок на первом же кадре, ровно то, что Ф8 убирала в TSX.
    """
    p = build_plan(edl_with_accent(), "in.mp4", fps=fps, width=1080, height=1920)
    line = dialogues(p)[0]
    m = re.search(
        r"\\fscx(\d+)\\fscy\1\\t\(0,\d+,\\fscx(\d+)\\fscy\2\)\\t\(\d+,\d+,\\fscx(\d+)", line
    )
    assert m, line
    base, peak, back = (int(m.group(i)) for i in (1, 2, 3))
    assert peak > base
    assert back == base


def test_accent_word_keeps_one_size_in_every_state():
    """libass меряет `\\fscx` при раскладке: разный кегль по состояниям двигает строку.

    Выделение жило только в `active`, а в `past` и `ghost` слово возвращалось к
    100 %, поэтому вся строка меняла ширину и прыгала по горизонтали каждый раз,
    когда акцентное слово меняло состояние. Инвариант «контейнер не прыгает»
    (Ф1) обязан держаться в обоих рендерерах.
    """
    lines = dialogues(plan())
    # «важно» — акцент в первом блоке: ghost в первой реплике, active в третьей.
    sizes = set()
    for line in lines[:3]:
        chunk = line.split("важно")[0].split("{")[-1]
        m = re.search(r"\\fscx(\d+)", chunk)
        assert m, chunk
        sizes.add(int(m.group(1)))
    assert len(sizes) == 1, f"акцентное слово меняет кегль по состояниям: {sizes}"
    assert sizes.pop() == round(load_preset("glass").animation.accent_scale * 100)


def test_ass_uses_the_size_python_measured_not_the_emphasis():
    """Ужатое длинное слово известно только Python — ASS обязан взять его кегль.

    Кегль это и есть раскладка: `wrap_block` мерил строку именно им. Вывести его
    из выделения нельзя — про ужатие выделение ничего не знает.
    """
    p = plan()
    word = p.blocks[0].lines[0][0]
    word.size = 40.0
    expected = round(word.size / Preset(**p.preset).font.size * 100)
    line = dialogues(p)[0]
    chunk = line.split(word.text)[0].split("{")[-1]
    assert f"\\fscx{expected}" in chunk, chunk
    assert expected < 100


# ------------------------------------------------------------------ пресет под кадр


@pytest.mark.parametrize("height", [1280, 960, 3840])
def test_smaller_frame_wraps_the_same_lines(height):
    """Превью другого размера обязано показывать ТУ ЖЕ композицию, что финал.

    Кегль — целое число, а бюджет строки считается от ширины кадра в float,
    поэтому округление кегля к ближайшему делало текст относительно бюджета
    крупнее (88 → 59 при k = 0.667 — это +0.57 %), и строка, влезавшая на 1080p
    с запасом в доли процента, на превью 720p переносилась. Проверяется на
    реальном ролике, а не на синтетике: запас в доли процента бывает только у
    настоящих строк.
    """
    edl = EDL(**{k: v for k, v in _baseline().items() if k != "camera"})
    ref = build_plan(edl, "in.mp4", fps=24, width=1080, height=1920)
    small = build_plan(edl, "in.mp4", fps=24, width=round(height * 1080 / 1920), height=height)
    for a, b in zip(ref.blocks, small.blocks):
        assert [[w.text for w in line] for line in a.lines] == [
            [w.text for w in line] for line in b.lines
        ]


def _baseline() -> dict:
    """Эталонный EDL из репозитория. Путь от файла теста, а не от рабочей папки."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "samples" / "edl-baseline.json"
    return json.loads(path.read_text(encoding="utf-8"))


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
