"""Тайминг появления — числами, а не на глаз: карточка b-roll (М5) и субтитр.

Проверять это внутри Python нечем: пружина и `interpolate` живут в Remotion, и
любая их модель на стороне тестов проверяла бы саму себя. Поэтому здесь
запускаются пробники — `render/card_probe.mjs` и `render/caption_probe.mjs`.
Они импортируют ту же библиотеку, читают те же пресеты и считают ту же
арифметику, что `layers/Broll.tsx` и `layers/Captions.tsx`.

Зачем тест, если Ф-фазы уже привели вход в порядок: числа входа целиком заданы
ПРЕСЕТОМ, то есть данными, которые правит владелец темы. Без замера правка
`broll_in_ms` молча возвращает вход вставки в ступеньку — ровно тот дефект
(AUDIT §8.3), ради которого М5 и написан, а правка `word_in_ms` так же молча
возвращает слово к включению за один кадр.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

RENDER = Path(__file__).resolve().parents[1] / "render"
PROBE = RENDER / "card_probe.mjs"
CAPTION_PROBE = RENDER / "caption_probe.mjs"


def _run(script: Path, fps: int) -> dict[str, dict]:
    node = shutil.which("node")
    if node is None or not (RENDER / "node_modules").exists():
        pytest.skip("нужен node и render/node_modules")
    done = subprocess.run(
        [node, str(script), str(fps), "--json"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(RENDER),
    )
    if done.returncode != 0:
        pytest.fail(f"{script.name} упал: {done.stderr.strip()[-2000:]}")
    return json.loads(done.stdout)


def probe(fps: int) -> dict[str, dict]:
    return _run(PROBE, fps)


def captions(fps: int) -> dict[str, dict]:
    return _run(CAPTION_PROBE, fps)


@pytest.mark.slow
@pytest.mark.parametrize("fps", [24, 30, 60])
def test_card_enter_is_long_enough_to_be_seen(fps: int):
    """DoD М5: не меньше 10 кадров видимого движения при 30 fps.

    Было 3 кадра (90 мс) — карточка не появлялась, а возникала. Порог «видимого»
    — 0.2 % полного хода за кадр: на карточке 475 px это меньше пикселя.
    """
    for name, r in probe(fps).items():
        floor = 10 if fps >= 30 else 8
        assert r["moving_frames"] >= floor, f"{name}: {r['moving_frames']} кадров движения"


@pytest.mark.slow
@pytest.mark.parametrize("fps", [24, 30, 60])
def test_card_enters_without_a_bounce(fps: int):
    """Вставка проступает, а не подпрыгивает.

    Раньше вход вела пружина, и тест требовал перелёта 2.5-10 %: картинка
    проскакивала своё место и возвращалась. Для полноэкранной картинки,
    которая закрывает говорящего, это читается как рывок, поэтому теперь
    обратное: ни альфа, ни наезд не идут назад и не проскакивают цель.
    """
    for name, r in probe(fps).items():
        assert r["overshoot_pct"] == 0, f"{name}: наезд проскочил на {r['overshoot_pct']:.2f} %"
        assert r["alpha_monotonic"], f"{name}: альфа шла назад"
        assert r["scale_monotonic"], f"{name}: наезд шёл назад"


@pytest.mark.slow
@pytest.mark.parametrize("fps", [24, 30, 60])
def test_card_alpha_never_jumps(fps: int):
    """Ни один кадр не берёт больше 17 % альфы.

    Прежний `easeOut` за восемь кадров отдавал 33 % альфы ПЕРВЫМ же кадром —
    картинка не проступала, а включалась.
    """
    for name, r in probe(fps).items():
        assert r["alpha_max_step"] <= 0.17, f"{name}: шаг альфы {r['alpha_max_step']:.3f}"


@pytest.mark.slow
@pytest.mark.parametrize("fps", [24, 30, 60])
def test_card_push_is_seen_not_spent_while_transparent(fps: int):
    """Когда картинка уже видна (альфа 0.9), впереди ещё треть наезда и больше.

    На кривой с быстрым стартом 70 % хода уходило, пока картинка была
    полупрозрачной: наезд формально был, а глазу доставался хвост в пару
    процентов.
    """
    for name, r in probe(fps).items():
        assert r["push_left_when_visible"] >= 0.33, (
            f"{name}: к альфе 0.9 осталось {r['push_left_when_visible']:.0%} наезда"
        )


@pytest.mark.slow
@pytest.mark.parametrize("fps", [24, 30, 60])
def test_card_is_in_focus_by_the_time_it_is_opaque(fps: int):
    """Расфокус уходит раньше, чем альфа доходит до единицы: смотрят уже на резкое."""
    for name, r in probe(fps).items():
        assert 0 < r["blur_gone_frame"] <= r["alpha_full_frame"], (
            f"{name}: фокус на кадре {r['blur_gone_frame']}, альфа=1 на {r['alpha_full_frame']}"
        )


@pytest.mark.slow
def test_alpha_arrives_before_the_scale_settles():
    """Рассинхрон альфы и масштаба: карточка сначала «есть», потом встаёт на место.

    Если вести оба параметра одной кривой, появление читается как механика.
    Альфа доезжает до единицы за `broll_in_ms`, наезд садится позже —
    проверяем именно порядок, а не конкретные кадры.
    """
    for name, r in probe(30).items():
        assert r["alpha_full_frame"] < r["scale_settled_frame"], (
            f"{name}: альфа на кадре {r['alpha_full_frame']}, "
            f"масштаб сел на {r['scale_settled_frame']}"
        )


# ------------------------------------------------------------------ субтитры


@pytest.mark.slow
@pytest.mark.parametrize("fps", [24, 30, 60])
def test_word_appears_over_several_frames_not_one(fps: int):
    """Слово обязано ПРОЯВЛЯТЬСЯ, а не включаться за кадр.

    Это единственное расхождение с эталоном, оставшееся после С1-С6: у него
    слово набирает яркость за 4 кадра (замер по «вставляет», 8.9-9.2 с), у нас
    оно переключалось из ghost в active мгновенно. Пол здесь низкий (2 кадра) —
    закон про восемь кадров к прозрачности не относится, — но и нулю места нет.
    """
    for name, r in captions(fps).items():
        assert r["word_reveal_frames"] >= 2, f"{name}: проявление {r['word_reveal_frames']} кадров"


@pytest.mark.slow
def test_reference_theme_keeps_the_reference_timing():
    """`glass` — измеренная копия эталона, и обязана ею остаться.

    Эталон при 30 fps: блок входит за 3 кадра чистой альфой (0.25 / 0.60 / 1.00
    по яркости внутри силуэта глифов), слово — за 4. Масштаба и сдвига нет
    вовсе. Тема с движением у нас отдельная (`pulse`), и заводить движение в
    `glass` через правку его чисел эта проверка не даёт.
    """
    glass = captions(30)["glass"]
    assert glass["block_in_frames"] == 3
    assert glass["word_reveal_frames"] == 4
    assert glass["word_rise_px"] == 0
    assert glass["word_pop_peak"] == 1.0
    assert glass["block_zoom"] == [1.0] * len(glass["block_zoom"])


@pytest.mark.slow
@pytest.mark.parametrize("fps", [24, 30, 60])
def test_live_theme_actually_moves(fps: int):
    """`pulse` заводится ради движения — значит движение обязано быть.

    Проверяется поведение, а не числа темы: слово и выезжает, и пульсирует, а
    подъём успевает закончиться (иначе слово так и осталось бы смещённым).
    """
    r = captions(fps)["pulse"]
    assert r["word_rise_px"] > 0, "слово не выезжает"
    assert r["word_pop_peak"] > 1.0, "«пшика» нет"
    assert r["word_rise"][0] > 0 and r["word_rise"][-1] == 0, "подъём не доехал до нуля"
    assert r["block_zoom"][0] < 1.0 and r["block_zoom"][-1] == pytest.approx(1.0)


@pytest.mark.slow
@pytest.mark.parametrize("fps", [24, 30, 60])
def test_word_alpha_never_goes_backwards(fps: int):
    """Проявление монотонно: любая просадка читается как моргание, а не как вход."""
    for name, r in captions(fps).items():
        alpha = r["word_alpha"]
        assert all(b >= a - 1e-9 for a, b in zip(alpha, alpha[1:])), f"{name}: альфа падает"
        assert alpha[-1] == pytest.approx(1.0), f"{name}: слово не доехало до полной силы"
