"""М5: тайминг появления карточки b-roll — числами, а не на глаз.

Проверять это внутри Python нечем: пружина живёт в Remotion, и любая её модель
на стороне тестов проверяла бы саму себя. Поэтому здесь запускается
`render/card_probe.mjs` — он импортирует ту же `spring` из той же библиотеки,
читает те же пресеты и считает ту же арифметику, что `layers/Broll.tsx`.

Зачем тест, если Ф-фазы уже привели вход в порядок: числа входа целиком заданы
пружиной ПРЕСЕТА, то есть данными, которые правит владелец темы. Без замера
правка `spring_stiffness` молча возвращает вход в три кадра — ровно тот дефект
(AUDIT §8.3), ради которого М5 и написан.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[1] / "render" / "card_probe.mjs"


def probe(fps: int) -> dict[str, dict]:
    node = shutil.which("node")
    if node is None or not (PROBE.parent / "node_modules").exists():
        pytest.skip("нужен node и render/node_modules")
    done = subprocess.run(
        [node, str(PROBE), str(fps), "--json"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROBE.parent),
    )
    if done.returncode != 0:
        pytest.fail(f"card_probe.mjs упал: {done.stderr.strip()[-2000:]}")
    return json.loads(done.stdout)


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
