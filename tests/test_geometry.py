"""Геометрия кадра: метрики шрифта, ширина строки, прямоугольники.

Главный тест здесь — `test_python_predicts_chromium_line_width`: он сверяет
предсказание Python с настоящим измерением в Chromium (DoD Ф4, допуск 2 px).
Без него все остальные проверки говорят только о самосогласованности модели, а
нужна согласованность с рендерером.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.core.geometry import (
    CAPTION_POSITIONS,
    FALLBACK_ADVANCE_EM,
    MIN_BAND_RATIO,
    STOCK_ASPECT,
    Rect,
    SizedWord,
    _emoji_cluster,
    _round_px,
    blocks_caption,
    broll_free_band,
    broll_rect,
    caption_rect,
    caption_zone,
    caption_text_width,
    emphasis_size,
    line_box_height,
    line_gap,
    load_metrics,
    measure_line,
    measure_word,
    metrics_for,
    min_band_ratio,
    resting_anchor,
    safe_margin_y,
    sized_words,
    space_width,
)
from src.core.preset import load_preset
from src.core.render_plan import PlanBlock, PlanWord, _rect_of_lines, wrap_block

REPO = Path(__file__).resolve().parents[1]
FRAME = (1080, 1920)
FONT = "Montserrat-ExtraBold.ttf"


@pytest.fixture(scope="module")
def metrics():
    return load_metrics(FONT)


def block(
    *lines: list[str], position: str = "bottom", accent: str = "", shout: str = ""
) -> PlanBlock:
    """Блок плана из готовых строк. `accent`/`shout` — слова крупнее остальных."""

    def emphasis(w: str) -> str:
        if w and w == shout:
            return "shout"
        return "accent" if w and w == accent else "none"

    return PlanBlock(
        start=0.0,
        end=1.0,
        position=position,  # type: ignore[arg-type]
        lines=[
            [
                PlanWord(text=w, start=0.0, end=1.0, emphasis=emphasis(w), word_idx=i)
                for i, w in enumerate(line)
            ]
            for line in lines
        ],
    )


def roomy_preset():
    """Пресет, в кадре которого карточке ЕСТЬ где встать при любой позиции блока.

    Поставляемые пресеты такого места не дают: три трёхстрочные полосы субтитра
    занимают больше кадра, чем в нём есть (см. `broll_free_band`), и `broll_rect`
    честно отдаёт на них полный кадр. Поэтому здесь же берут пресет и проверки
    якорей, пропорции и безопасной зоны: на glass они мерили бы не правило, а его
    вырождение.
    """
    preset = load_preset("glass").model_copy(deep=True)
    preset.font.size = 44
    preset.caption.max_lines = 1
    preset.caption.margin = 100
    return preset


# ------------------------------------------------------------------ прямоугольник


def test_rect_edges_and_center():
    r = Rect(10, 20, 100, 50)
    assert (r.right, r.bottom) == (110, 70)
    assert (r.center_x, r.center_y) == (60, 45)


def test_rect_overlap_is_a_real_intersection():
    """Половины кадра врали: соседство по половине — это не пересечение."""
    top = Rect(0, 0, 100, 100)
    bottom = Rect(0, 200, 100, 100)
    assert not top.overlaps(bottom)
    assert top.overlaps(Rect(50, 50, 100, 100))
    # С зазором «почти касаются» уже считается конфликтом.
    assert top.overlaps(bottom, gap=120)


def test_rect_round_trips_through_dict():
    r = Rect(1.234567, 2.0, 3.5, 4.25)
    assert Rect.from_dict(r.to_dict()) == r.rounded()


# ------------------------------------------------------------------ метрики


def test_metrics_are_generated_for_every_bundled_font():
    """Метрики коммитятся: без них geometry не работает, а fontTools не в рантайме."""
    for ttf in sorted((REPO / "assets" / "fonts").glob("*.ttf")):
        assert load_metrics(ttf.name).units_per_em > 0


def test_committed_metrics_are_exactly_what_the_tool_extracts():
    """Данные в `presets/metrics` обязаны совпадать с выгрузкой из TTF, байт в байт.

    JSON лежит в репозитории и в рантайме читается без fontTools — то есть это
    ЕДИНСТВЕННЫЙ источник ширин для всей геометрии. Ни один тест не проверял, что
    он соответствует шрифту: правка руками, обновление TTF без перевыгрузки или
    смена репертуара в `tools/font_metrics.py` проходили молча, а разъезжалась бы
    от этого вся раскладка сразу.
    """
    from tools.font_metrics import extract

    for ttf in sorted((REPO / "assets" / "fonts").glob("*.ttf")):
        committed = json.loads(
            (REPO / "src" / "core" / "presets" / "metrics" / f"{ttf.stem}.json").read_text(
                encoding="utf-8"
            )
        )
        assert extract(ttf) == committed, ttf.name


def test_kerning_is_extracted(metrics):
    """«ГА» без кернинга шире почти на 12 px при кегле 88 — это шесть допусков DoD."""
    assert metrics.kern("Г", "А") < 0
    naive = (metrics.advance("Г") + metrics.advance("А")) / metrics.units_per_em * 88
    assert naive - measure_word("ГА", 88, metrics=metrics) > 10


def test_kerning_stops_at_a_script_boundary(metrics):
    """Через границу письменности Chromium не кернит — HarfBuzz шейпит прогоны врозь.

    Числа не из нашей же модели, а с линейки: «Сбербankовский» на кегле 88 со
    спейсингом −0.015 занимает в Chromium 809.34 px. Наивная модель применяла
    кернинг пар «б|a» (−12) и «k|о» (−24) и давала 806.17 — промах 3.2 px,
    полтора бюджета DoD.
    """
    assert metrics.kern("б", "a") < 0, "в шрифте пара есть — это не про её отсутствие"
    assert metrics.kern("k", "о") < 0
    assert measure_word(
        "Сбербankовский", 88, metrics=metrics, letter_spacing=-0.015
    ) == pytest.approx(809.34, abs=0.1)


def test_neutral_character_joins_the_script_on_its_left(metrics):
    """Дефис в «PDF-файл» уходит в латиницу, и пара «-|ф» кернинг не получает.

    Chromium: 469.57 px на кегле 88 со спейсингом −0.015. С кернингом через
    границу выходило 470.45 — 0.9 px, половина бюджета DoD на ровном месте.
    """
    assert metrics.kern("-", "ф") > 0
    assert measure_word("PDF-файл", 88, metrics=metrics, letter_spacing=-0.015) == pytest.approx(
        469.57, abs=0.1
    )


def test_emoji_is_measured_as_one_cluster(metrics):
    """Эмодзи — один кластер, одна ширина, один letter-spacing, сколько бы в нём
    ни было кодовых точек.

    108.48 px на кегле 88 со спейсингом −0.015 — замер в Chromium, одинаковый
    для 🔥 (одна точка), ❤️ (две), 1️⃣ (три) и 👨‍👩‍👧 (пять). Модель, считающая
    по кодовым точкам, давала на семье впятеро больше.
    """
    for emoji in ("🔥", "❤️", "1️⃣", "👨‍👩‍👧", "👍🏽", "🇷🇺"):
        assert metrics.split(emoji) == [emoji], emoji
        assert measure_word(
            emoji, 88, metrics=metrics, letter_spacing=-0.015
        ) == pytest.approx(108.48, abs=0.1), emoji
    # Два эмодзи подряд — два кластера, а не один: склеивать их значит занизить
    # строку вдвое.
    assert metrics.split("😀🔥") == ["😀", "🔥"]


def test_glyphs_of_the_font_are_not_hijacked_by_the_emoji_rule(metrics):
    """Стрелка и математика есть в Montserrat — их меряет шрифт, а не оценка.

    51.48 и 52.27 px на кегле 88 (Chromium). Правило «символ вне шрифта» обязано
    смотреть на шрифт, а не на диапазон Unicode: стрелки лежат ровно в тех же
    блоках, что эмодзи.
    """
    assert measure_word("→", 88, metrics=metrics, letter_spacing=-0.015) == pytest.approx(
        51.48, abs=0.1
    )
    assert measure_word("≈", 88, metrics=metrics, letter_spacing=-0.015) == pytest.approx(
        52.27, abs=0.1
    )


def test_no_glyph_of_the_font_is_measured_by_the_emoji_rule():
    """Ни один символ cmap не меряется эмодзи-правилом — и это не про выгрузку.

    Раньше правило смотрело только на диапазон Unicode и на наличие символа в
    `advances`, поэтому «™», «▲», «▶», «◆», «◊» становились эмодзи ровно потому,
    что их не выгрузил `tools/font_metrics.py`: 108.50 px вместо 51.48. Условие
    обязано быть про НАЧЕРТАНИЕ, а не про выгрузку, — тогда сужение репертуара
    его не ломает.
    """
    fontTools = pytest.importorskip("fontTools.ttLib")
    font = fontTools.TTFont(REPO / "assets" / "fonts" / "Montserrat-ExtraBold.ttf")
    hijacked = [
        chr(code) for code in font.getBestCmap() if _emoji_cluster(chr(code), 0) == 1
    ]
    assert not hijacked, f"эмодзи-правило перехватило свои же глифы: {hijacked}"


def test_text_presentation_symbols_are_not_emoji(metrics):
    """«✓» и «▪» рисует обычный шрифт, «☑️» — цветной: разница в селекторе.

    Chromium на кегле 88 со спейсингом −0.015: «☑» 77.58 px, «☑️» (тот же символ
    с VS16) 108.48. Правило по диапазону давало обоим 108.50 и промахивалось на
    31 px на каждом текстовом символе; у «▪» промах доходил до 79 px.
    """
    for text in ("✓", "▪", "☑", "☀", "✉"):
        assert _emoji_cluster(text, 0) == 0, text
        assert text not in metrics.advances, text  # иначе тест не про то
        assert measure_word(text, 88, metrics=metrics) == pytest.approx(
            FALLBACK_ADVANCE_EM * 88
        ), text
    # Тот же символ с VS16 — уже цветной глиф и эмодзи-ширина.
    assert measure_word(
        "☑️", 88, metrics=metrics, letter_spacing=-0.015
    ) == pytest.approx(108.48, abs=0.1)
    # А VS15 отменяет цвет даже у эмодзи по умолчанию.
    assert _emoji_cluster("☀︎", 0) == 0


def test_unknown_glyph_is_estimated_high_not_measured(metrics):
    """Символ ВНЕ ШРИФТА — оценка, и оценка сознательно завышенная.

    Проверять здесь равенство `FALLBACK_ADVANCE_EM * кегль` бессмысленно: это
    измерение модели её же собственной константой. Настоящая ширина зависит от
    того, какой фолбэк подставит система (замер этой машины — в слоу-тесте
    `test_fallback_never_underestimates_chromium`), и допуск 2 px на эту ветку не
    распространяется. Что обязано быть верным всегда — направление промаха:
    оценка НЕ МЕНЬШЕ настоящей ширины, иначе прямоугольник блока окажется у́же
    текста и картинка встанет на буквы.
    """
    assert "ஃ" not in metrics.advances
    assert measure_word("ஃ", 100, metrics=metrics) == pytest.approx(
        FALLBACK_ADVANCE_EM * 100
    )
    # Оценка сверху: ни один известный фолбэк, кроме цветного эмодзи, не шире em.
    assert FALLBACK_ADVANCE_EM >= 1.0


def test_a_stray_combining_mark_takes_no_room(metrics):
    """Диакритика и селекторы начертания ширины не занимают ни в одном шрифте.

    Пока для них брался em, одинокий селектор стоил 88 px, которых в кадре нет.
    """
    assert measure_word("️", 88, metrics=metrics) == 0.0
    assert measure_word("́", 88, metrics=metrics) == 0.0


def test_edge_spaces_of_a_word_collapse(metrics):
    """Бокс слова — inline-block: краевые пробелы CSS убирает, подряд идущие сводит.

    Chromium на « пробел »: 348.92 px, ровно как на «пробел». Наивная модель
    считала оба края и давала 397.50 — 48 px, два десятка бюджетов DoD.
    """
    plain = measure_word("пробел", 88, metrics=metrics, letter_spacing=-0.015)
    assert plain == pytest.approx(348.92, abs=0.1)
    assert measure_word(" пробел ", 88, metrics=metrics, letter_spacing=-0.015) == pytest.approx(
        plain
    )
    assert measure_word("а  б", 88, metrics=metrics, letter_spacing=-0.015) == pytest.approx(
        measure_word("а б", 88, metrics=metrics, letter_spacing=-0.015)
    )
    # Пробел ЦЕЛИКОМ схлопывается: в кадре у такого слова ширины нет.
    assert measure_word(" ", 88, metrics=metrics) == 0.0
    # А между словами он настоящий — там ему не с чем схлопываться.
    assert space_width(88, metrics=metrics, letter_spacing=-0.015) == pytest.approx(24.29, abs=0.1)


def test_all_css_whitespace_collapses(metrics):
    """CSS сводит к одному пробелу весь набор, а не только U+0020.

    Таба и перевода строки в шрифте нет, поэтому «а\tб» уходило в фолбэк:
    197.65 px против 135.27 в Chromium — 62 px, три десятка бюджетов DoD.
    Расшифровка их не приносит, но `text.strip()` на входе снимает только
    краевые, и перевод строки из ручного EDL доживал до плана.
    """
    plain = measure_word("а б", 88, metrics=metrics, letter_spacing=-0.015)
    for text in ("а\tб", "а\nб", "а\r\nб", "а \t б", "а\fб"):
        assert measure_word(
            text, 88, metrics=metrics, letter_spacing=-0.015
        ) == pytest.approx(plain), repr(text)
    assert measure_word("\n\tслово\n", 88, metrics=metrics) == measure_word(
        "слово", 88, metrics=metrics
    )


def test_thin_spaces_are_measured_by_the_font(metrics):
    """Тонкие пробелы в шрифте ЕСТЬ и не схлопываются: 11.53 px у U+2009 (Chromium)."""
    assert measure_word(
        "\u2009", 88, metrics=metrics, letter_spacing=-0.015
    ) == pytest.approx(11.53, abs=0.1)
    assert measure_word("а\u2009б", 88, metrics=metrics, letter_spacing=-0.015) == pytest.approx(
        122.50, abs=0.1
    )


def test_letter_spacing_skips_zero_width_units(metrics):
    """Спейсинг достаётся только тому, у чего есть своя ширина.

    Chromium не даёт letter-spacing диакритике и селекторам начертания: advance у
    них ноль, и спейсинг тоже. «и́мя» на кегле 88 со спейсингом −0.015 занимает
    188.06 px, а счёт «по единицам» давал 186.74 — минус 1.32 px на каждый знак,
    и на bold (ls −0.02, кегль 104) одного знака хватало, чтобы выйти за DoD.
    """
    assert metrics.units_count("и́мя") == 4
    assert metrics.spacing_units("и́мя") == 3
    assert measure_word("и́мя", 88, metrics=metrics, letter_spacing=-0.015) == pytest.approx(
        188.06, abs=0.1
    )
    assert measure_word(
        "абв́где́ж", 88, metrics=metrics, letter_spacing=-0.015
    ) == pytest.approx(406.56, abs=0.1)


def test_non_breaking_space_is_not_collapsed(metrics):
    """Неразрывный пробел рисуется, а не схлопывается: 135.27 px на «а б» (Chromium)."""
    assert measure_word("а б", 88, metrics=metrics, letter_spacing=-0.015) == pytest.approx(
        135.27, abs=0.1
    )


# ------------------------------------------------------------------ измерение


def test_empty_text_has_no_width(metrics):
    assert measure_word("", 88, metrics=metrics) == 0.0
    assert measure_line([], metrics=metrics) == 0.0


def test_width_is_proportional_to_size(metrics):
    small = measure_word("слово", 44, metrics=metrics)
    big = measure_word("слово", 88, metrics=metrics)
    assert big == pytest.approx(2 * small)


def test_letter_spacing_shrinks_the_word(metrics):
    wide = measure_word("человек", 88, metrics=metrics, letter_spacing=0.0)
    tight = measure_word("человек", 88, metrics=metrics, letter_spacing=-0.015)
    # Спейсинг ставится после каждого символа, включая последний.
    assert wide - tight == pytest.approx(0.015 * 88 * 7)


def test_uppercase_is_measured_after_uppercasing(metrics):
    lower = measure_word("слово", 88, metrics=metrics)
    upper = measure_word("слово", 88, metrics=metrics, uppercase=True)
    assert upper == measure_word("СЛОВО", 88, metrics=metrics)
    assert upper > lower


def test_line_uses_a_real_space_not_a_gap(metrics):
    words = [SizedWord("один", 88), SizedWord("два", 88)]
    line = measure_line(words, metrics=metrics, base_size=88)
    solo = sum(measure_word(w.text, w.size, metrics=metrics) for w in words)
    assert line - solo == pytest.approx(space_width(88, metrics=metrics))


def test_accent_word_widens_the_line(metrics):
    """Акцент — кегль, а не transform: он занимает место, и строка это знает."""
    plain = measure_line([SizedWord("одну", 88), SizedWord("истину", 88)], metrics=metrics)
    accented = measure_line(
        [SizedWord("одну", 88), SizedWord("истину", 88 * 1.18)], metrics=metrics, base_size=88
    )
    assert accented > plain


def test_an_empty_word_still_costs_a_space(metrics):
    """Пустое слово не выбрасывается: вместе с ним пропал бы разделяющий пробел.

    Captions.tsx ставит пробел перед каждым словом с индексом больше нуля, не
    заглядывая в его текст, и Chromium это подтверждает: «раз»/«»/«два» это
    377.81 px, а не 353.52. На этом же пробеле «независимая линейка» тестов
    расходилась с производственным `_Ruler.line_width` — то есть две реализации
    ширины строки уже разъехались.
    """
    words = [SizedWord("раз", 88), SizedWord("", 88), SizedWord("два", 88)]
    line = measure_line(words, metrics=metrics, letter_spacing=-0.015, base_size=88)
    assert line == pytest.approx(377.81, abs=0.1)
    without = measure_line(
        [SizedWord("раз", 88), SizedWord("два", 88)],
        metrics=metrics, letter_spacing=-0.015, base_size=88,
    )
    assert line - without == pytest.approx(
        space_width(88, metrics=metrics, letter_spacing=-0.015)
    )


def test_letter_spacing_of_an_accent_word_comes_from_the_line(metrics):
    """`letter-spacing` объявлен на строке и наследуется в пикселях, не в em.

    Считать его от кегля акцента — ошибка порядка 1.6 px на слове из семи букв,
    почти весь бюджет DoD.
    """
    line = measure_line(
        [SizedWord("человек", 88 * 1.18)], metrics=metrics, letter_spacing=-0.015, base_size=88
    )
    from_own_size = measure_word("человек", 88 * 1.18, metrics=metrics, letter_spacing=-0.015)
    assert line > from_own_size
    assert line - from_own_size == pytest.approx(0.015 * (88 * 1.18 - 88) * 7)


def test_line_box_is_never_shorter_than_the_strut():
    """Строчный бокс не бывает ниже кегля КОНТЕЙНЕРА, как бы ни ужали слова.

    Chromium: строка из слова кеглем 35 в glass занимает 93.27 px — ровно
    струт 88×1.06, а не 37. Пока высоту считали по самому крупному СЛОВУ, блок,
    прибитый к низу кадра, занижался на каждую ужатую строку, и весь промах
    уходил вверх.
    """
    preset = load_preset("glass")
    strut = preset.font.size * preset.caption.line_height
    assert line_box_height([35.0], preset) == pytest.approx(93.28, abs=0.05)
    assert line_box_height([35.0], preset) == pytest.approx(strut)
    assert line_box_height([], preset) == pytest.approx(strut)
    # Крупное слово бокс поднимает: 88×1.18×1.06 = 110.07 (Chromium 110.06).
    assert line_box_height(
        [preset.font.size * preset.animation.accent_scale], preset
    ) == pytest.approx(110.07, abs=0.05)


def test_pixel_rounding_matches_the_renderer():
    """Округление половинок — вверх, как `Math.round` в TSX, а не банковское.

    `round(2.5)` в Python это 2, а в JS 3. Оба числа считаются по обе стороны, и
    на некоторых высотах кадра после `Preset.scaled` они расходились на пиксель.
    """
    assert _round_px(2.5) == 3
    assert _round_px(3.5) == 4
    assert _round_px(-0.5) == 0
    preset = load_preset("glass").model_copy(deep=True)
    preset.font.size = 50
    preset.caption.line_height = 1.05  # 50 × 0.05 = 2.5 ровно
    assert line_gap(preset) == 3


def test_emphasis_size_follows_the_preset():
    preset = load_preset("glass")
    assert emphasis_size("none", preset) == preset.font.size
    assert emphasis_size("accent", preset) == preset.font.size * preset.animation.accent_scale
    assert emphasis_size("shout", preset) == preset.font.size * preset.animation.shout_scale


# ------------------------------------------------------------------ субтитр


def test_caption_rect_sits_inside_the_frame():
    preset = load_preset("glass")
    for position in ("top", "center", "bottom"):
        rect = caption_rect(block(["одну", "истину"], position=position), preset, FRAME)
        assert rect.x >= 0 and rect.right <= FRAME[0]
        assert rect.y >= 0 and rect.bottom <= FRAME[1]


def test_caption_rect_respects_the_safe_margin():
    preset = load_preset("glass")
    margin = safe_margin_y(preset, FRAME[1])
    bottom = caption_rect(block(["одну", "истину"], position="bottom"), preset, FRAME)
    top = caption_rect(block(["одну", "истину"], position="top"), preset, FRAME)
    assert FRAME[1] - bottom.bottom == pytest.approx(margin)
    assert top.y == pytest.approx(margin)


def test_caption_rect_grows_with_the_number_of_lines():
    preset = load_preset("glass")
    one = caption_rect(block(["одну"]), preset, FRAME)
    two = caption_rect(block(["одну"], ["истину"]), preset, FRAME)
    assert two.height > one.height


def test_caption_rect_never_exceeds_the_width_budget():
    preset = load_preset("glass")
    long_line = ["невероятно", "длинная", "строка", "которая", "не", "влезает"]
    rect = caption_rect(block(long_line), preset, FRAME)
    chrome = 2 * (preset.panel.padding_x + preset.panel.border_width)
    assert rect.width <= caption_text_width(preset, FRAME[0]) + chrome + 1e-6


def test_caption_text_width_leaves_room_for_the_panel():
    """Доля кадра не должна выталкивать панель за безопасную зону."""
    preset = load_preset("glass").model_copy(deep=True)
    preset.geometry.max_line_width_ratio = 0.99
    budget = caption_text_width(preset, FRAME[0])
    safe = FRAME[0] - 2 * FRAME[0] * preset.geometry.safe_area_ratio
    chrome = 2 * (preset.panel.padding_x + preset.panel.border_width)
    assert budget + chrome <= safe + 1e-6


def test_panelless_preset_has_no_padding_in_the_rect():
    """Без панели прямоугольник блока — это ровно ширина текста.

    Регистр здесь уже применён: в план слово приезжает после `display_text`,
    поэтому caption_rect меряет текст как есть и второй раз его не трогает.
    """
    bold = load_preset("bold")
    rect = caption_rect(block(["СЛОВО"]), bold, FRAME)
    width = measure_word(
        "СЛОВО",
        bold.font.size,
        metrics=metrics_for(bold),
        letter_spacing=bold.font.letter_spacing,
    )
    assert rect.width == pytest.approx(width, abs=0.01)


def test_missing_font_file_is_an_explicit_error():
    preset = load_preset("glass").model_copy(deep=True)
    preset.font.file = None
    with pytest.raises(ValueError):
        caption_rect(block(["слово"]), preset, FRAME)


# ------------------------------------------------------------------ картинки


def test_broll_rect_anchors_go_to_their_sides():
    preset = roomy_preset()
    margin = FRAME[0] * preset.geometry.broll_margin_ratio
    left = broll_rect("top_left", preset, FRAME)
    right = broll_rect("top_right", preset, FRAME)
    assert left.x == pytest.approx(margin)
    assert FRAME[0] - right.right == pytest.approx(margin)
    assert left.y == right.y


def test_broll_fullscreen_is_the_whole_frame():
    rect = broll_rect("fullscreen", load_preset("glass"), FRAME)
    assert (rect.x, rect.y, rect.width, rect.height) == (0.0, 0.0, 1080.0, 1920.0)


def test_broll_center_is_centered_horizontally():
    """По вертикали центр кадра занят субтитром, картинка живёт в свободной полосе."""
    preset = roomy_preset()
    rect = broll_rect("center", preset, FRAME)
    band = broll_free_band(preset, FRAME)
    assert rect.center_x == pytest.approx(FRAME[0] / 2)
    assert rect.center_y == pytest.approx((band[0] + band[1]) / 2)


def test_broll_inherits_source_aspect_by_default():
    """Квадратная карточка резала стоковые 2:3 на треть — теперь пропорция наследуется."""
    preset = roomy_preset()
    rect = broll_rect("top_right", preset, FRAME)
    assert rect.width / rect.height == pytest.approx(STOCK_ASPECT)


def test_broll_aspect_changes_shape_not_area_rule():
    preset = roomy_preset()
    wide = broll_rect("top_right", preset, FRAME, aspect=3 / 2)
    tall = broll_rect("top_right", preset, FRAME, aspect=2 / 3)
    assert wide.width / wide.height == pytest.approx(3 / 2)
    assert tall.width / tall.height == pytest.approx(2 / 3)
    # Ширину задаёт пресет, пока картинка влезает в свободную полосу по высоте.
    assert wide.width == pytest.approx(FRAME[0] * preset.geometry.broll_inset_ratio)


ANCHORS = ("top_left", "top_right", "bottom_left", "bottom_right", "center")


def test_broll_never_crosses_the_caption_band():
    """DoD Ф7: картинка ни разу не пересекает субтитр — при любом якоре и позиции.

    Позиции берутся все три, включая `center`: он разрешён контрактом
    (`CaptionBlock.position`) и в `bold` стоит позицией пресета по умолчанию.
    Перебирать только `top`/`bottom` значит обходить ровно тот случай, где
    гарантия ломалась.
    """
    preset = roomy_preset()
    for anchor in ANCHORS:
        rect = broll_rect(anchor, preset, FRAME)
        assert rect.width < FRAME[0], anchor  # место есть, карточка настоящая
        for position in CAPTION_POSITIONS:
            assert not blocks_caption(rect, preset, FRAME, position), (anchor, position)


def test_bundled_presets_never_put_a_card_over_their_own_captions():
    """Гарантия на поставляемых пресетах: либо карточка мимо субтитра, либо полный кадр.

    Позиции перебираются ВСЕ, какие разрешает контракт, а не только те, которые
    пресет даёт сам: `caption_positions` исключал ровно `center`, на котором
    гарантия и ломалась, — тест обходил стороной свой единственный интересный
    случай.
    """
    for name in ("glass", "bold"):
        preset = load_preset(name)
        for anchor in ANCHORS:
            rect = broll_rect(anchor, preset, FRAME)
            if rect.width >= FRAME[0]:
                continue  # места под карточку нет — осознанный полный кадр
            for position in CAPTION_POSITIONS:
                assert not blocks_caption(rect, preset, FRAME, position), (name, anchor, position)


def test_a_centered_caption_leaves_no_room_for_a_card():
    """`bold` держит субтитр по центру: карточка накрывала бы КАЖДЫЙ блок.

    Полный кадр честнее: он хотя бы не ставит картинку поверх текста.
    """
    bold = load_preset("bold")
    assert bold.caption.position == "center"
    rect = broll_rect("top_right", bold, FRAME)
    assert (rect.x, rect.y, rect.width, rect.height) == (0.0, 0.0, 1080.0, 1920.0)


def test_a_dense_preset_falls_back_instead_of_landing_on_the_text():
    """Вырожденная полоса не даёт карточку полного размера прямо в полосу субтитра."""
    preset = load_preset("glass").model_copy(deep=True)
    preset.font.size = 210
    preset.caption.line_height = 1.2
    preset.caption.margin = 120
    band = broll_free_band(preset, FRAME)
    assert band[1] - band[0] < FRAME[1] * MIN_BAND_RATIO
    rect = broll_rect("top_right", preset, FRAME)
    assert (rect.x, rect.y, rect.width, rect.height) == (0.0, 0.0, 1080.0, 1920.0)


def test_the_band_floor_comes_from_the_preset():
    """Порог полосы — рычаг пресета, а не константа модуля.

    Доля в одиночку решает, будет ли у пресета карточка вообще: у glass со всеми
    тремя полосами субтитра остаётся 128 px против порога 230 — и все вставки
    становятся полноэкранными. Пресет обязан уметь этот порог подкрутить, поэтому
    `MIN_BAND_RATIO` здесь только значение по умолчанию (`min_band_ratio`).
    """
    preset = load_preset("glass").model_copy(deep=True)
    assert min_band_ratio(preset) == MIN_BAND_RATIO
    band = broll_free_band(preset, FRAME, CAPTION_POSITIONS)
    height = band[1] - band[0]
    assert height < FRAME[1] * MIN_BAND_RATIO  # по умолчанию карточки нет
    object.__setattr__(preset.geometry, "min_band_ratio", height / FRAME[1] / 2)
    assert broll_rect("top_right", preset, FRAME, positions=CAPTION_POSITIONS).width < FRAME[0]


def test_broll_rect_stays_inside_the_safe_area():
    preset = roomy_preset()
    safe = FRAME[1] * preset.geometry.safe_area_ratio
    for anchor in ("top_left", "bottom_right", "center"):
        rect = broll_rect(anchor, preset, FRAME)
        assert rect.y >= safe - 1e-6
        assert rect.bottom <= FRAME[1] - safe + 1e-6


def test_broll_shrinks_when_the_free_band_is_short():
    """Не влезло по высоте — уменьшается целиком: растянуть значит снова резать."""
    preset = roomy_preset()
    band_h = broll_free_band(preset, FRAME)[1] - broll_free_band(preset, FRAME)[0]
    rect = broll_rect("top_right", preset, FRAME)
    assert rect.height == pytest.approx(band_h)
    assert rect.width == pytest.approx(band_h * STOCK_ASPECT)


def test_caption_zone_reserves_the_worst_case_block():
    """Резерв обязан вмещать самый высокий блок, какой разрешает КОНТРАКТ.

    Контракт разрешает `shout` в КАЖДОЙ строке, а не в одной: формула
    «(строк−1)·базовая + самая высокая» описывала блок ровно с одним выделенным
    словом, и трёхстрочный блок с криком в каждой строке оказывался выше резерва
    на 63 px — картинка вставала внутрь текста. Прошлый тест этого не видел,
    потому что клал ровно одно выделенное слово: он был написан под реализацию.
    """
    preset = load_preset("glass")
    zone = caption_zone(preset, FRAME, "bottom")
    lines = (["раз", "два"], ["три", "четыре"], ["пять"])
    for emphasis in ("accent", "shout"):
        every = PlanBlock(
            start=0.0,
            end=1.0,
            position="bottom",
            lines=[
                [
                    PlanWord(text=w, start=0.0, end=1.0, emphasis=emphasis, word_idx=i)
                    for i, w in enumerate(line)
                ]
                for line in lines
            ],
        )
        worst = caption_rect(every, preset, FRAME)
        assert zone.height >= worst.height - 1e-6, emphasis
        assert zone.y <= worst.y + 1e-6, emphasis


def test_caption_zone_does_not_promise_what_it_cannot_know():
    """Резерв НЕ покрывает блок, вылезший за `max_lines`, — и не притворяется.

    `wrap_block` при упоре в пол кегля сознательно отдаёт больше строк, чем
    разрешено (шесть вместо трёх на шести длинных словах), а сколько их будет,
    без текста не знает никто. Поэтому сборка плана разводит картинку не с
    резервом, а с измеренными прямоугольниками; тест фиксирует границу, чтобы
    следующий читатель не принял резерв за гарантию.
    """
    preset = load_preset("glass")
    words = [
        PlanWord(text="достопримечательность", start=0.0, end=1.0, word_idx=i)
        for i in range(6)
    ]
    lines, widths = wrap_block(words, preset, FRAME[0])
    assert len(lines) > preset.caption.max_lines, "блок обязан вылезти за лимит строк"
    rect = _rect_of_lines("bottom", lines, widths, preset, FRAME)
    assert rect.height > caption_zone(preset, FRAME, "bottom").height


def test_caption_zone_counts_lines_from_the_preset():
    """Число строк — поле пресета, а не константа рядом: иначе резерв врёт."""
    preset = load_preset("glass").model_copy(deep=True)
    preset.caption.max_lines = 4
    zone = caption_zone(preset, FRAME, "bottom")
    worst = caption_rect(
        block(["раз", "два"], ["три"], ["четыре"], ["пять"], shout="пять"), preset, FRAME
    )
    assert zone.height >= worst.height - 1e-6
    assert zone.y <= worst.y + 1e-6


def test_fullscreen_to_inset_rests_in_a_corner():
    """Разводить с субтитром надо угол, а не полный кадр, через который картинка едет."""
    assert resting_anchor("fullscreen", "fullscreen_to_inset") == "top_right"
    assert resting_anchor("bottom_left", "fullscreen_to_inset") == "bottom_left"
    assert resting_anchor("fullscreen", "pop_in") == "fullscreen"


# ------------------------------------------------------------------ DoD: сверка с Chromium


def _measure_in_node(request: dict) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("нет node")
    script = REPO / "render" / "measure.mjs"
    if not (REPO / "render" / "node_modules" / "@remotion" / "renderer").exists():
        pytest.skip("не установлены зависимости render/")
    done = subprocess.run(
        [node, str(script)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        cwd=REPO / "render",
        timeout=180,
    )
    if done.returncode != 0:
        # «Браузера нет» и «кадр разъехался» — разные новости. На чистой машине
        # `openBrowser` падает, потому что Chromium ещё не скачан, и провал
        # выглядел бы как поломка кода, а не как пустое окружение.
        text = done.stderr.lower()
        if any(
            mark in text
            for mark in (
                "openbrowser",
                "ensurebrowser",
                "browser executable",
                "chrome headless shell",
            )
        ):
            pytest.skip(f"нет скачанного Chromium: {done.stderr.strip()[-300:]}")
        pytest.fail(f"measure.mjs упал: {done.stderr.strip()[-2000:]}")
    return json.loads(done.stdout)


def _baseline_lines(base: float, accent: float, shout: float, *, uppercase: bool = False):
    """Строки базового плана как слова с кеглями. Синтетика тут ловит не то.

    Реальные субтитры приносят и кернинг («Ту», «То», «ГА»), и акцентные слова
    другого кегля, и отрицательный letter-spacing — то есть ровно те три места,
    где модель может разъехаться с браузером.
    """
    plan = json.loads((REPO / "samples" / "plan-baseline.json").read_text(encoding="utf-8"))
    scale = {"accent": accent, "shout": shout, "none": 1.0}
    lines: dict[str, list[SizedWord]] = {}
    for bi, plan_block in enumerate(plan["blocks"]):
        for li, line in enumerate(plan_block["lines"]):
            lines[f"b{bi}l{li}"] = [
                SizedWord(
                    w["text"].upper() if uppercase else w["text"],
                    base * scale[w["emphasis"]],
                )
                for w in line
            ]
    # Отдельные слова меряем тоже: если разъедется строка, по ним видно, кто виноват.
    unique = sorted({w for ws in lines.values() for w in ws}, key=lambda w: (w.text, w.size))
    lines.update({f"w{i}": [w] for i, w in enumerate(unique)})
    return lines


def _assert_matches_chromium(font_file: str, weight: int, base: float, spacing: float, items):
    request = {
        "font": str(REPO / "assets" / "fonts" / font_file),
        "weight": weight,
        "base_size": base,
        "letter_spacing": spacing,
        "items": [
            {"id": key, "words": [{"text": w.text, "size": w.size} for w in ws]}
            for key, ws in items.items()
        ],
    }
    measured = _measure_in_node(request)["widths"]
    metrics = load_metrics(font_file)
    report: list[str] = []
    for key, ws in items.items():
        predicted = measure_line(ws, metrics=metrics, letter_spacing=spacing, base_size=base)
        if abs(predicted - measured[key]) >= 2.0:
            text = " ".join(w.text for w in ws)
            report.append(f"{key} {text!r}: python {predicted:.2f}, chromium {measured[key]:.2f}")
    assert not report, "расхождение больше 2 px:\n" + "\n".join(report)


def _hard_lines(base: float, accent: float, shout: float) -> dict[str, list[SizedWord]]:
    """Случаи, которых в базовом плане нет, а в кадре они бывают.

    Реальный прогон ловит только то, что уже случилось: в нём нет ни эмодзи, ни
    смешанной письменности, ни ужатого слова. А ломалась модель ровно на них —
    каждая строка здесь стоит на найденном промахе, не на фантазии.
    """
    return {
        # Кернинг: пары с самым большим значением в шрифте.
        "kern": [SizedWord("ГА", base), SizedWord("АУ", base), SizedWord("Ту", base)],
        # Граница письменности внутри слова и через дефис: кернинг там не работает.
        "script-inword": [SizedWord("Сбербankовский", base)],
        "script-hyphen": [SizedWord("PDF-файл", base), SizedWord("v2.0", base)],
        # Пробелы по краям слова схлопываются.
        "spaces": [SizedWord(" пробел ", base), SizedWord("а  б", base)],
        # Эмодзи: кластер целиком, сколько бы в нём ни было кодовых точек.
        "emoji-word": [SizedWord("огонь 🔥", base)],
        "emoji-zwj": [SizedWord("семья 👨‍👩‍👧", base)],
        "emoji-keycap": [SizedWord("раз 1️⃣", base)],
        "emoji-accent": [SizedWord("это", base), SizedWord("огонь 🔥", accent)],
        "emoji-two": [SizedWord("двойной 😀🔥", base)],
        # Свои символы шрифта в тех же блоках Unicode, что эмодзи.
        "arrows": [SizedWord("→", base), SizedWord("≈", base), SizedWord("№1", base)],
        # Три кегля в одной строке и лигатуры.
        "sizes": [SizedWord("тише", base), SizedWord("едешь", accent), SizedWord("дальше", shout)],
        "ligature": [SizedWord("fiction", base), SizedWord("affine", base)],
        # Длинное слово ужатым кеглем — то, что делает `fit_size`.
        "shrunk": [SizedWord("труднопроизносимое", base * 0.4)],
        "long": [SizedWord("труднопроизносимое", base)],
        "one-char": [SizedWord("я", base), SizedWord("Щ", shout)],
        "punct": [SizedWord("«ёлки-палки»", base), SizedWord("…", base)],
        # Символы, которые в шрифте ЕСТЬ, но не попадали в репертуар выгрузки:
        # они уходили в оценку и промахивались на 15-80 px каждый.
        "symbols": [SizedWord("▲▶◆◊", base), SizedWord("Ozon™", base)],
        "greek": [SizedWord("Δλμ", base), SizedWord("Ω", base)],
        "math-letters": [SizedWord("ℓ", base), SizedWord("⅓", base), SizedWord("H₂O", base)],
        "latin-extra": [SizedWord("Việt", base)],
        # Пробельные символы CSS и тонкие пробелы шрифта.
        "css-space": [SizedWord("а\tб", base), SizedWord("а\nб", base)],
        "thin-space": [SizedWord("а\u2009б", base), SizedWord("а\u200aб", base)],
        # Комбинирующие знаки: своей ширины нет, спейсинга тоже.
        "combining": [SizedWord("и́мя", base), SizedWord("абв́где́ж", base)],
        # Пустое слово: ширины нет, а разделяющий пробел есть.
        "empty-word": [SizedWord("раз", base), SizedWord("", base), SizedWord("два", base)],
    }


@pytest.mark.slow
def test_python_predicts_chromium_on_hard_lines():
    """DoD Ф4 на трудных случаях, а не только на том, что уже было в прогоне."""
    preset = load_preset("glass")
    items = _hard_lines(
        preset.font.size, preset.animation.accent_scale * preset.font.size,
        preset.animation.shout_scale * preset.font.size,
    )
    _assert_matches_chromium(
        preset.font.file, preset.font.weight, preset.font.size, preset.font.letter_spacing, items
    )


@pytest.mark.slow
def test_python_predicts_chromium_on_hard_lines_for_the_dense_preset():
    """Те же трудные случаи на bold: капс, кегль 104 и спейсинг −0.02.

    Промах, который на glass укладывался в допуск, на bold из него выходит:
    спейсинг там плотнее, а кегль крупнее, и одного комбинирующего знака хватало,
    чтобы пробить 2 px.
    """
    preset = load_preset("bold")
    items = _hard_lines(
        preset.font.size, preset.animation.accent_scale * preset.font.size,
        preset.animation.shout_scale * preset.font.size,
    )
    if preset.font.uppercase:
        # Заглавная «Μ» (U+039C) в Montserrat отсутствует — после капса греческий
        # случай уезжает в фолбэк, а на него DoD не распространяется по контракту
        # (есть отдельный тест `test_fallback_never_underestimates_chromium`).
        items = {
            key: [SizedWord(w.text.upper(), w.size) for w in ws]
            for key, ws in items.items()
            if key != "greek"
        }
    _assert_matches_chromium(
        preset.font.file, preset.font.weight, preset.font.size, preset.font.letter_spacing, items
    )


@pytest.mark.slow
def test_fallback_never_underestimates_chromium():
    """Ветка фолбэка: DoD 2 px на неё не распространяется, но направление — да.

    Символ, которого в шрифте нет, рисует шрифт СИСТЕМЫ, и его метрик у нас нет
    (замер этой машины: «✓» 72.42 px, «α» 47.80, «ஃ» 77.53 против оценки 86.68).
    Требовать здесь 2 px — значит подгонять модель под конкретный хост. Что
    обязано держаться всегда — знак промаха: оценка НЕ МЕНЬШЕ настоящей ширины,
    иначе прямоугольник блока окажется у́же текста и картинка встанет на буквы.
    """
    preset = load_preset("glass")
    metrics = load_metrics(preset.font.file)
    unknown = [c for c in "✓✔✉☑☀▪●ஃαβ،" if c not in metrics.advances]
    assert unknown, "тест про символы вне шрифта — а их не осталось"
    request = {
        "font": str(REPO / "assets" / "fonts" / preset.font.file),
        "weight": preset.font.weight,
        "base_size": preset.font.size,
        "letter_spacing": preset.font.letter_spacing,
        "items": [{"id": c, "text": c, "size": preset.font.size} for c in unknown],
    }
    measured = _measure_in_node(request)["widths"]
    for char in unknown:
        predicted = measure_word(
            char,
            preset.font.size,
            metrics=metrics,
            letter_spacing=preset.font.letter_spacing,
        )
        assert predicted >= measured[char] - 1e-6, (
            f"{char!r}: оценка {predicted:.2f} УЖЕ настоящих {measured[char]:.2f} — "
            "промах внутрь ставит картинку на текст"
        )


@pytest.mark.slow
def test_python_predicts_chromium_line_width():
    """DoD Ф4: расхождение с настоящим измерением в Chromium меньше 2 px."""
    preset = load_preset("glass")
    items = _baseline_lines(
        preset.font.size, preset.animation.accent_scale, preset.animation.shout_scale
    )
    assert len(items) >= 20, "в базовом плане должно быть с чего мерить"
    _assert_matches_chromium(
        preset.font.file, preset.font.weight, preset.font.size, preset.font.letter_spacing, items
    )


@pytest.mark.slow
def test_python_predicts_chromium_for_every_bundled_font():
    """Тот же тест для второго шрифта и второго пресета: капсом, крупнее, плотнее.

    Метрики выгружаются на каждый TTF из assets/fonts, и каждая выгрузка должна
    быть проверена — иначе второй файл это непроверенные данные, которые молча
    врут, когда пресет на него переключат.
    """
    bold = load_preset("bold")
    for ttf in sorted((REPO / "assets" / "fonts").glob("*.ttf")):
        items = _baseline_lines(
            bold.font.size,
            bold.animation.accent_scale,
            bold.animation.shout_scale,
            uppercase=bold.font.uppercase,
        )
        _assert_matches_chromium(
            ttf.name, bold.font.weight, bold.font.size, bold.font.letter_spacing, items
        )


@pytest.mark.slow
def test_caption_rect_matches_the_chromium_box():
    """Не только ширина строки, но и весь блок вместе с панелью.

    Ширины мало: пересечение с картинкой (Ф7) считается по всему
    прямоугольнику, и высота там такой же контракт. Строчный бокс поднимает
    самое крупное слово, между строками стоит `gap`, вокруг — паддинг и рамка
    панели; каждое из этих слагаемых может разъехаться с браузером молча.
    """
    preset = load_preset("glass")
    plan = json.loads((REPO / "samples" / "plan-baseline.json").read_text(encoding="utf-8"))
    blocks = {
        f"b{i}": block(*[[w["text"] for w in line] for line in b["lines"]])
        for i, b in enumerate(plan["blocks"])
    }
    for key, b in blocks.items():
        for line, source in zip(b.lines, plan["blocks"][int(key[1:])]["lines"]):
            for word, w in zip(line, source):
                word.emphasis = w["emphasis"]

    request = {
        "font": str(REPO / "assets" / "fonts" / preset.font.file),
        "weight": preset.font.weight,
        "base_size": preset.font.size,
        "letter_spacing": preset.font.letter_spacing,
        "line_height": preset.caption.line_height,
        "items": [],
        "boxes": [
            {
                "id": key,
                "lines": [
                    [{"text": w.text, "size": w.size} for w in sized_words(line, preset)]
                    for line in b.lines
                ],
                # `gap` намеренно не передаём: его считает сама страница
                # формулой Captions.tsx, и сверка проверяет заодно `line_gap`.
                "padding_x": preset.panel.padding_x,
                "padding_y": preset.panel.padding_y,
                "border_width": preset.panel.border_width,
                "max_width": caption_text_width(preset, FRAME[0]),
            }
            for key, b in blocks.items()
        ],
    }
    measured = _measure_in_node(request)["boxes"]

    report: list[str] = []
    for key, b in blocks.items():
        rect = caption_rect(b, preset, FRAME)
        box = measured[key]
        if abs(rect.width - box["width"]) >= 2.0 or abs(rect.height - box["height"]) >= 2.0:
            report.append(
                f"{key}: python {rect.width:.2f}×{rect.height:.2f}, "
                f"chromium {box['width']:.2f}×{box['height']:.2f}"
            )
    assert not report, "прямоугольник блока разъехался с Chromium:\n" + "\n".join(report)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["glass", "bold"])
def test_the_production_block_rect_matches_chromium(name: str):
    """Тот же DoD, но на ПРОИЗВОДСТВЕННОМ пути: `wrap_block` + `_rect_of_lines`.

    В план едет прямоугольник, который считает `render_plan`, а не
    `geometry.caption_rect`: по нему меряется яркость фона (Ф9) и разводится
    картинка (Ф7). Пока с Chromium сверяли только `caption_rect`, проверенным
    оказывался код, которого в проде нет, — а расходились они как раз на том,
    что в проде и бывает: на эмодзи и на ужатом переносом кегле.
    """
    preset = load_preset(name).scaled(FRAME[1])
    blocks: dict[str, list[PlanWord]] = {}
    for i, words in enumerate(
        (
            [("тише", "none", None), ("едешь", "accent", None)],
            [("огонь", "none", "🔥"), ("прямо", "none", None), ("сейчас", "shout", "🚀")],
            [("трансцендентальностью", "shout", None)],
            [("Сбербankовский", "none", None), ("PDF-файл", "accent", None)],
            [(w, "none", None) for w in "раз два три четыре пять шесть семь восемь".split()],
            [("миллиард", "shout", "💥")],
        )
    ):
        blocks[f"p{i}"] = [
            PlanWord(
                text=t.upper() if preset.font.uppercase else t,
                start=0.0,
                end=1.0,
                emphasis=e,
                emoji=emoji,
                word_idx=j,
            )
            for j, (t, e, emoji) in enumerate(words)
        ]

    laid: dict[str, tuple[list[list[PlanWord]], list[float]]] = {
        key: wrap_block(words, preset, FRAME[0]) for key, words in blocks.items()
    }
    request = {
        "font": str(REPO / "assets" / "fonts" / preset.font.file),
        "weight": preset.font.weight,
        "base_size": preset.font.size,
        "letter_spacing": preset.font.letter_spacing,
        "line_height": preset.caption.line_height,
        "items": [],
        "boxes": [
            {
                "id": key,
                "lines": [
                    [{"text": w.text, "size": w.size, "emoji": w.emoji} for w in line]
                    for line in lines
                ],
                # `gap` намеренно не передаём: его считает сама страница
                # формулой Captions.tsx, и сверка проверяет заодно `line_gap`.
                "padding_x": preset.panel.padding_x,
                "padding_y": preset.panel.padding_y,
                "border_width": preset.panel.border_width,
            }
            for key, (lines, _) in laid.items()
        ],
    }
    measured = _measure_in_node(request)["boxes"]

    report: list[str] = []
    for key, (lines, widths) in laid.items():
        rect = _rect_of_lines("bottom", lines, widths, preset, FRAME)
        box = measured[key]
        if abs(rect.width - box["width"]) >= 2.0 or abs(rect.height - box["height"]) >= 2.0:
            report.append(
                f"{name}/{key}: python {rect.width:.2f}×{rect.height:.2f}, "
                f"chromium {box['width']:.2f}×{box['height']:.2f}"
            )
    assert not report, "прямоугольник плана разъехался с Chromium:\n" + "\n".join(report)
