"""Договор о шрифте: рендер обязан объявлять шрифт так же, как линейка.

Раскладку строк считает Python, а сверяется она с `render/measure.mjs` — той
самой headless-страницей, про которую в шапке линейки написано «расхождение
здесь это расхождение в кадре». Утверждение верно ровно настолько, насколько
страница объявляет шрифт так же, как настоящий рендер.

Расхождение тут и жило. Линейка пишет `font-weight: 1 1000`, то есть один файл
закрывает весь диапазон и синтезировать жирность Chromium не станет. `Root.tsx`
же создавал `new FontFace("ReelbotCaption", url)` БЕЗ дескриптора веса — по
спецификации это `normal`, то есть 400, — а `Captions.tsx` просит
`fontWeight: 800`. На такую пару Chromium ВПРАВЕ нарисовать синтетический жир.

Замером проверено: на нынешней сборке он ширину не меняет (объявленные 400 и
100 против запрошенных 800 дали те же пиксели), то есть активной ошибки не
было — была несогласованность объявлений. Сторожим именно её: сверка 2 px
осмысленна ровно настолько, насколько рендер объявляет шрифт так же, как
страница замера. С чужими шрифтами, где собственный вес файла не совпадает с
весом пресета, цена этой несогласованности уже не предсказуема.

Проверка текстовая, как у пробника движения (`test_probe_drift.py`): исполнить
TSX вне композиции Remotion нечем, а число нужно сравнить.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULER = ROOT / "render" / "measure.mjs"
ROOT_TSX = ROOT / "render" / "src" / "Root.tsx"
CAPTIONS = ROOT / "render" / "src" / "layers" / "Captions.tsx"


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _ruler_weight_range() -> str:
    """`font-weight` из @font-face линейки."""
    m = re.search(r"@font-face\s*\{.*?font-weight:\s*([^;]+);", _src(RULER), re.S)
    assert m, "measure.mjs: в @font-face больше нет font-weight"
    return m.group(1).strip()


def test_the_renderer_declares_the_same_weight_range_as_the_ruler():
    """Иначе Python сверен с линейкой, а кадр нарисован синтетическим жиром."""
    declared = re.search(
        r"new FontFace\(\s*\"ReelbotCaption\",\s*`url\(\$\{staticFile\(fontUrl\)\}\)`\s*,"
        r"\s*\{([^}]*)\}\s*\)",
        _src(ROOT_TSX),
    )
    assert declared, (
        "Root.tsx: у `new FontFace(\"ReelbotCaption\", …)` нет третьего аргумента с "
        "дескрипторами. Без него лицо регистрируется весом 400, Captions.tsx просит "
        f"свой вес, и Chromium синтезирует жир — линейка объявляет {_ruler_weight_range()!r}"
    )
    m = re.search(r"weight:\s*\"([^\"]+)\"", declared.group(1))
    assert m, f"Root.tsx: в дескрипторах FontFace нет weight: {declared.group(1)!r}"
    assert m.group(1).strip() == _ruler_weight_range(), (
        f"диапазон веса разошёлся: Root.tsx {m.group(1)!r}, "
        f"measure.mjs {_ruler_weight_range()!r}. Меряется один шрифт, рисуется другой"
    )


def test_the_caption_asks_for_the_weight_the_preset_names():
    """Вторая половина той же пары: кегль и вес субтитра берутся из пресета.

    Если `fontWeight` перестанет приходить из пресета, диапазон в объявлении
    шрифта станет бессмысленным — сверять будет нечего.
    """
    assert re.search(r"fontWeight:\s*ps\.font\.weight", _src(CAPTIONS)), (
        "Captions.tsx больше не берёт вес из пресета — проверьте, что сверка "
        "с линейкой всё ещё про тот же шрифт"
    )
