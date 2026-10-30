"""Пробник движения — копия кривой входа из рендера. Этот тест сторожит копию.

`render/card_probe.mjs` считает те же кадры, что рисует
`render/src/layers/Broll.tsx`: в него переписаны константы входа, кривые безье и
обе половины `popBump` из `render/src/style.ts`. Импортировать оригиналы нечем —
TSX с хуками Remotion вне композиции не исполняется, а числа нужны пробнику до
всякого рендера, — поэтому копия неизбежна.

Сверка здесь ТЕКСТОВАЯ, и это сказано прямо: regexp по двум файлам, а не прогон
кода. Прогон и не помог бы. Кадры в `tests/test_motion.py` считает САМ пробник
по своим копиям, поэтому правка Broll.tsx его замеров не меняет вообще —
проверено на живом файле: удаление `+ bump` из обеих ветвей масштаба оставило
зелёным весь набор, включая `-m slow`. Комментарий в пробнике утверждал
обратное («тест поймает это по кадрам»); теперь ловит — здесь и по тексту.

Чего тест НЕ гарантирует: что формулы эквивалентны. Переписанную форму кривой в
Broll.tsx мимо этих констант он не увидит. Зато видит каждое число и каждую
букву `popBump`, и сообщение называет, что именно разошлось.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BROLL = ROOT / "render" / "src" / "layers" / "Broll.tsx"
STYLE = ROOT / "render" / "src" / "style.ts"
PROBE = ROOT / "render" / "card_probe.mjs"

# Константы входа, переписанные в пробник. Каждая здесь потому, что от неё
# зависят кадры, закреплённые `test_motion.py`: пол подъёма (`CARD_POP`,
# `INSET_POP`), ход наезда (`FULL_PUSH`), вершина горки (`OVERSHOOT_RISE`) и
# дефолты на случай пресета без этих полей (`ENTER_MS`, `ENTER_BLUR`).
SHARED_NUMBERS = (
    "ENTER_MS",
    "ENTER_BLUR",
    "FULL_PUSH",
    "CARD_POP",
    "INSET_POP",
    "OVERSHOOT_RISE",
)
SHARED_BEZIERS = ("SOFT", "SETTLE", "PUSH")


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _flat(code: str) -> str:
    """Без переносов и типов TS: `(t: number): number =>` ≡ `(t) =>`.

    Иначе тест падал бы от переформатирования prettier, то есть ровно там, где
    расхождения нет.
    """
    return re.sub(r"\s+", " ", code.replace(": number", "")).strip()


def _number(src: str, name: str, where: Path) -> float:
    m = re.search(rf"^const {name} = (-?[\d.]+);", src, re.M)
    assert m, f"{where.name}: нет `const {name} = <число>` — константу переименовали или убрали"
    return float(m.group(1))


def _bezier(src: str, name: str, where: Path) -> tuple[float, ...]:
    m = re.search(rf"^const {name} = Easing\.bezier\(([^)]*)\);", src, re.M)
    assert m, f"{where.name}: нет `const {name} = Easing.bezier(...)`"
    return tuple(float(x) for x in m.group(1).split(","))


def _arrow(src: str, name: str, where: Path) -> str:
    """Стрелочная функция целиком — аргументы и тело, без типов и отбивки."""
    m = re.search(
        rf"^(?:export )?const {name} = (\([^)]*\))\s*(?::\s*\w+\s*)?=> \{{\n(.*?)\n\}};",
        src,
        re.M | re.S,
    )
    assert m, f"{where.name}: нет функции {name}"
    return _flat(m.group(1) + " " + m.group(2))


def _default_rise(src: str, where: Path) -> float:
    m = re.search(r"const popBump = \([^)]*rise = (-?[\d.]+)[^)]*\)", src)
    assert m, f"{where.name}: у popBump нет значения rise по умолчанию"
    return float(m.group(1))


def test_probe_copies_the_same_numbers_as_the_renderer():
    """Каждая переписанная константа входа совпадает с оригиналом."""
    broll, probe = _src(BROLL), _src(PROBE)
    for name in SHARED_NUMBERS:
        rendered = _number(broll, name, BROLL)
        measured = _number(probe, name, PROBE)
        assert rendered == measured, (
            f"{name} разошлась: Broll.tsx {rendered}, card_probe.mjs {measured}. "
            "Видео рисуется по первому числу, а замеры test_motion.py считаются по второму"
        )


def test_probe_copies_the_same_easing_curves():
    """Безье входа — те же контрольные точки: кривая и есть вся анимация."""
    broll, probe = _src(BROLL), _src(PROBE)
    for name in SHARED_BEZIERS:
        rendered = _bezier(broll, name, BROLL)
        measured = _bezier(probe, name, PROBE)
        assert rendered == measured, (
            f"безье {name} разошлось: Broll.tsx {rendered}, card_probe.mjs {measured}"
        )


def test_probe_copies_popbump_itself_and_its_default_rise():
    """`popBump` и её `smoothStep` переписаны из style.ts буква в букву.

    Форма горки — это и есть то, чем вход отличается от пружины: ноль на обоих
    концах окна и ровно одна вершина. Числа совпали, а тело разошлось — и
    `card_turns` в замере по-прежнему 1, тогда как в кадре колебание.
    """
    style, probe = _src(STYLE), _src(PROBE)
    for fn in ("smoothStep", "popBump"):
        original = _arrow(style, fn, STYLE)
        copy = _arrow(probe, fn, PROBE)
        assert original == copy, (
            f"{fn} разошлась:\n  style.ts:       {original}\n  card_probe.mjs: {copy}"
        )
    original_rise = _default_rise(style, STYLE)
    copy_rise = _default_rise(probe, PROBE)
    assert original_rise == copy_rise, (
        f"popBump(rise=…) по умолчанию разошлось: style.ts {original_rise}, "
        f"card_probe.mjs {copy_rise}"
    )


def test_both_scale_branches_still_get_the_overshoot_the_probe_measures():
    """Горка прибавляется к ОБЕИМ ветвям подъёма — именно это и пропало незаметно.

    Проверка текстовая по той же причине, что и остальные: перелёт в замере
    считает пробник, и `+ bump`, удалённое из Broll.tsx, не меняет в его выводе
    ни одного числа.
    """
    broll = _flat(_src(BROLL))
    for const in ("CARD_POP", "INSET_POP"):
        assert re.search(rf"interpolate\(enter, \[0, 1\], \[{const}, 1\]\) \+ bump", broll), (
            f"ветвь {const} больше не прибавляет `bump`: пробник мерит перелёт, "
            "которого в кадре уже нет"
        )
    assert re.search(r"popBump\(framesIn / enterFrames, OVERSHOOT_RISE\)", broll), (
        "Broll.tsx считает `bump` не как `popBump(framesIn / enterFrames, OVERSHOOT_RISE)` — "
        "пробник считает именно так"
    )
    assert re.search(r"popBump\(f / enter, OVERSHOOT_RISE\)", _flat(_src(PROBE))), (
        "card_probe.mjs считает горку не как `popBump(f / enter, OVERSHOOT_RISE)` — "
        "это уже не та кривая, что в Broll.tsx"
    )
