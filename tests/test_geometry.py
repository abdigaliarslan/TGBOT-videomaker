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
    blocks_caption,
    broll_free_band,
    broll_rect,
    caption_positions,
    caption_rect,
    caption_zone,
    caption_text_width,
    emphasis_size,
    line_gap,
    load_metrics,
    measure_line,
    measure_word,
    metrics_for,
    resting_anchor,
    safe_margin_y,
    sized_words,
)
from src.core.preset import load_preset
from src.core.render_plan import PlanBlock, PlanWord

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
    занимают больше кадра, чем в нём есть (см. `broll_free_band`). Чтобы проверять
    саму гарантию, а не её вырождение, нужен пресет пореже.
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


def test_kerning_is_extracted(metrics):
    """«ГА» без кернинга шире почти на 12 px при кегле 88 — это шесть допусков DoD."""
    assert metrics.kern("Г", "А") < 0
    naive = (metrics.advance("Г") + metrics.advance("А")) / metrics.units_per_em * 88
    assert naive - measure_word("ГА", 88, metrics=metrics) > 10


def test_unknown_glyph_falls_back_to_em(metrics):
    """Эмодзи рисует чужой шрифт; его ширину этот TTF не знает и не выдумывает."""
    assert measure_word("🔥", 100, metrics=metrics) == pytest.approx(
        FALLBACK_ADVANCE_EM * 100
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
    assert line - solo == pytest.approx(measure_word(" ", 88, metrics=metrics))


def test_accent_word_widens_the_line(metrics):
    """Акцент — кегль, а не transform: он занимает место, и строка это знает."""
    plain = measure_line([SizedWord("одну", 88), SizedWord("истину", 88)], metrics=metrics)
    accented = measure_line(
        [SizedWord("одну", 88), SizedWord("истину", 88 * 1.18)], metrics=metrics, base_size=88
    )
    assert accented > plain


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
    preset = load_preset("glass")
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
    preset = load_preset("glass")
    rect = broll_rect("center", preset, FRAME)
    band = broll_free_band(preset, FRAME)
    assert rect.center_x == pytest.approx(FRAME[0] / 2)
    assert rect.center_y == pytest.approx((band[0] + band[1]) / 2)


def test_broll_inherits_source_aspect_by_default():
    """Квадратная карточка резала стоковые 2:3 на треть — теперь пропорция наследуется."""
    preset = load_preset("glass")
    rect = broll_rect("top_right", preset, FRAME)
    assert rect.width / rect.height == pytest.approx(STOCK_ASPECT)


def test_broll_aspect_changes_shape_not_area_rule():
    preset = load_preset("glass")
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
    """Гарантия на поставляемых пресетах: либо карточка мимо субтитра, либо полный кадр."""
    for name in ("glass", "bold"):
        preset = load_preset(name)
        for anchor in ANCHORS:
            rect = broll_rect(anchor, preset, FRAME)
            if rect.width >= FRAME[0]:
                continue  # места под карточку нет — осознанный полный кадр
            for position in caption_positions(preset):
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


def test_broll_rect_stays_inside_the_safe_area():
    preset = load_preset("glass")
    safe = FRAME[1] * preset.geometry.safe_area_ratio
    for anchor in ("top_left", "bottom_right", "center"):
        rect = broll_rect(anchor, preset, FRAME)
        assert rect.y >= safe - 1e-6
        assert rect.bottom <= FRAME[1] - safe + 1e-6


def test_broll_shrinks_when_the_free_band_is_short():
    """Не влезло по высоте — уменьшается целиком: растянуть значит снова резать."""
    preset = load_preset("glass")
    band_h = broll_free_band(preset, FRAME)[1] - broll_free_band(preset, FRAME)[0]
    rect = broll_rect("top_right", preset, FRAME)
    assert rect.height == pytest.approx(band_h)
    assert rect.width == pytest.approx(band_h * STOCK_ASPECT)


def test_caption_zone_reserves_the_worst_case_block():
    """Полоса резерва обязана вмещать самый высокий блок, какой разрешает контракт.

    Самый высокий — это `shout` (1.34), а не `accent` (1.18): подставить сюда
    акцент значит написать тест под реализацию, а блок с криком выше резерва.
    """
    preset = load_preset("glass")
    zone = caption_zone(preset, FRAME, "bottom")
    for emphasis in ("accent", "shout"):
        worst = caption_rect(
            block(["раз", "два"], ["три", "четыре"], ["пять"], **{emphasis: "пять"}),
            preset,
            FRAME,
        )
        assert zone.height >= worst.height - 1e-6, emphasis
        assert zone.y <= worst.y + 1e-6, emphasis


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
                "gap": line_gap(preset),
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
