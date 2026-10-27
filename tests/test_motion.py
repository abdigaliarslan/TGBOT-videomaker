"""Тайминг появления — числами, а не на глаз: карточка b-roll (М5) и субтитр.

Проверять это внутри Python нечем: пружина и `interpolate` живут в Remotion, и
любая их модель на стороне тестов проверяла бы саму себя. Поэтому здесь
запускаются пробники — `render/card_probe.mjs` и `render/caption_probe.mjs`.
Они импортируют ту же библиотеку, читают те же пресеты и считают ту же
арифметику, что `layers/Broll.tsx` и `layers/Captions.tsx`.

Зачем тест, если Ф-фазы уже привели вход в порядок: числа входа целиком заданы
ПРЕСЕТОМ, то есть данными, которые правит владелец темы. Без замера правка
`spring_stiffness` молча возвращает вход в три кадра — ровно тот дефект
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
def test_card_overshoot_exists_and_is_not_clipped(fps: int):
    """Перелёт обязан быть виден и не срезан клампом.

    Раньше пружина перелетала на 12.4 %, но `durationInFrames` сжимал её в 1.8
    раза и кламп срезал вершину: движение было, а упругости не было. Верхняя
    граница здесь широкая (10 %) намеренно — это предохранитель от «пружина
    снова превратилась в удар», а не вкусовой коридор: 4 % у glass и 6.6 % у
    bold — сознательно разный характер двух тем.
    """
    for name, r in probe(fps).items():
        assert 2.5 <= r["overshoot_pct"] <= 10.0, f"{name}: перелёт {r['overshoot_pct']:.2f} %"
        assert max(r["scale"]) > 1.0, f"{name}: вершина срезана клампом"


@pytest.mark.slow
def test_alpha_arrives_before_the_scale_settles():
    """Рассинхрон альфы и масштаба: карточка сначала «есть», потом встаёт на место.

    Если вести оба параметра одной пружиной, появление читается как механика.
    Альфа доезжает до единицы за `FADE_MS`, масштаб успокаивается позже —
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
