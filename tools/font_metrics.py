#!/usr/bin/env python3
"""Выгрузка метрик TTF в JSON: Python учится мерить текст без браузера.

Зачем это вообще. Геометрия субтитра жила константами в TSX, а Python оперировал
половинами кадра и поэтому не мог предсказать, куда попадёт строка (AUDIT §8.4).
Предсказать её можно ровно одним способом — посчитать ширину теми же числами,
которыми её считает Chromium: advance-ширины глифов, кернинг и лигатуры из самого
шрифта. Всё это лежит в TTF, вопрос только в том, чтобы вытащить и разложить в
формат, который читается за миллисекунды.

Запускается руками при добавлении шрифта, результат коммитится:

    python tools/font_metrics.py                 # все шрифты из assets/fonts
    python tools/font_metrics.py assets/fonts/Montserrat-Bold.ttf

Формат результата (`src/core/presets/metrics/<stem>.json`) описан в
`src/core/geometry.py`, который его и читает. Ключи — символы, а не имена
глифов: измеряем мы текст, а не глифы, и лишний слой имён только мешает.

Кернинг занимает 85 тысяч пар на репертуаре из 535 символов, если хранить его
парами. Хранить парами незачем: кернинг в OpenType устроен классами, поэтому
матрица факторизуется обратно в 95×83 с 2560 ненулевыми клетками. Факторизация
делается по готовой матрице, а не разбором классов из шрифта: так не нужно
повторять в Python порядок применения подтаблиц, а результат совпадает точно.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from fontTools.ttLib import TTFont

REPO_ROOT = Path(__file__).resolve().parents[1]
FONTS_DIR = REPO_ROOT / "assets" / "fonts"
OUT_DIR = REPO_ROOT / "src" / "core" / "presets" / "metrics"

# Репертуар: что вообще может приехать в субтитр. Русский и латиница с
# диакритикой, цифры, типографская пунктуация, валюты. Всё остальное (эмодзи,
# иероглифы) рисуется фолбэк-шрифтом системы, и мерить его по этому TTF
# бессмысленно — для него в geometry.py отдельная ветка.
CHAR_RANGES: tuple[tuple[int, int], ...] = (
    (0x0020, 0x007E),  # ASCII
    (0x00A0, 0x024F),  # Latin-1, Latin Extended-A/B
    (0x0400, 0x052F),  # кириллица и дополнение
    (0x2010, 0x205E),  # тире, кавычки, многоточие, №
    (0x20A0, 0x20BF),  # валюты
    (0x2116, 0x2116),  # №
    (0x2190, 0x21FF),  # стрелки
    (0x2200, 0x22FF),  # математика (× ÷ ≈ ≠)
)


def repertoire() -> list[str]:
    out: list[str] = []
    for lo, hi in CHAR_RANGES:
        out.extend(chr(c) for c in range(lo, hi + 1))
    return out


# --------------------------------------------------------------------------- кернинг


def _unwrap(subtable: Any) -> Any:
    """Extension-лукап (тип 9) — обёртка ради 32-битных смещений, а не логика."""
    while type(subtable).__name__ == "ExtensionPos":
        subtable = subtable.ExtSubTable
    return subtable


def _kern_subtables(font: TTFont) -> list[list[Any]]:
    """Подтаблицы фичи `kern`, сгруппированные по лукапам и в порядке применения.

    Порядок важен: внутри лукапа срабатывает первая подтаблица, которая знает
    пару, а вклады разных лукапов складываются. Ровно так это делает HarfBuzz,
    а значит и Chromium.
    """
    if "GPOS" not in font:
        return []
    gpos = font["GPOS"].table
    indices: list[int] = []
    for record in gpos.FeatureList.FeatureRecord:
        if record.FeatureTag == "kern":
            indices.extend(record.Feature.LookupListIndex)
    return [
        [_unwrap(st) for st in gpos.LookupList.Lookup[i].SubTable] for i in sorted(set(indices))
    ]


def _prepare_subtable(subtable: Any) -> tuple[str, set[str], Any] | None:
    """Подтаблица → быстрая структура (тип, покрытие, данные)."""
    fmt = getattr(subtable, "Format", None)
    if type(subtable).__name__ != "PairPos":
        return None  # контекстный кернинг в наших шрифтах не встречается
    coverage = set(subtable.Coverage.glyphs)
    if fmt == 1:
        pairs: dict[tuple[str, str], int] = {}
        for first, pair_set in zip(subtable.Coverage.glyphs, subtable.PairSet):
            for record in pair_set.PairValueRecord:
                pairs[(first, record.SecondGlyph)] = getattr(record.Value1, "XAdvance", 0) or 0
        return ("pairs", coverage, pairs)
    class1 = subtable.ClassDef1.classDefs if subtable.ClassDef1 else {}
    class2 = subtable.ClassDef2.classDefs if subtable.ClassDef2 else {}
    matrix = [
        [getattr(rec.Value1, "XAdvance", 0) or 0 for rec in row.Class2Record]
        for row in subtable.Class1Record
    ]
    counts = (subtable.Class1Count, subtable.Class2Count)
    return ("classes", coverage, (class1, class2, matrix, *counts))


def _kern_value(lookups: list[list[tuple[str, set[str], Any]]], left: str, right: str) -> int:
    total = 0
    for subtables in lookups:
        for kind, coverage, data in subtables:
            if left not in coverage:
                continue
            if kind == "pairs":
                if (left, right) in data:
                    total += data[(left, right)]
                    break
                continue  # покрытие есть, пары нет — подтаблица не сработала
            class1, class2, matrix, n1, n2 = data
            c1 = class1.get(left, 0)
            c2 = class2.get(right, 0)
            if c1 < n1 and c2 < n2:
                total += matrix[c1][c2]
            break
    return total


def _factorize(
    units: list[str], value: Any
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Матрицу пар сжимаем обратно в классы: одинаковые строки — один класс.

    Возвращает (левые классы, правые классы, разреженная матрица «l:r» → units).
    Класс 0 всегда нулевой, поэтому символы без кернинга в словари не попадают.
    """
    rows = {u: tuple(value(u, other) for other in units) for u in units}

    zero_row = (0,) * len(units)
    left_ids: dict[tuple[int, ...], int] = {zero_row: 0}
    left: dict[str, int] = {}
    for unit in units:
        row = rows[unit]
        left[unit] = left_ids.setdefault(row, len(left_ids))

    by_id: list[tuple[int, ...]] = [()] * len(left_ids)
    for row, idx in left_ids.items():
        by_id[idx] = row

    zero_col = (0,) * len(left_ids)
    right_ids: dict[tuple[int, ...], int] = {zero_col: 0}
    right: dict[str, int] = {}
    for column, unit in enumerate(units):
        col = tuple(by_id[i][column] for i in range(len(left_ids)))
        right[unit] = right_ids.setdefault(col, len(right_ids))

    matrix: dict[str, int] = {}
    for col, r in right_ids.items():
        for l, adjustment in enumerate(col):  # noqa: E741 — l/r это индексы классов
            if adjustment:
                matrix[f"{l}:{r}"] = adjustment

    return (
        {u: c for u, c in left.items() if c},
        {u: c for u, c in right.items() if c},
        matrix,
    )


# --------------------------------------------------------------------------- лигатуры


def _ligatures(font: TTFont, cmap: dict[int, str]) -> dict[str, str]:
    """Стандартные лигатуры (`liga`) — строка исходных символов → имя глифа.

    Chromium включает `liga` по умолчанию, поэтому «fi» в кадре занимает ширину
    одного глифа, а не двух. Для кириллицы правил нет, но два латинских есть,
    и без них ширина латинского слова уедет.
    """
    if "GSUB" not in font:
        return {}
    gsub = font["GSUB"].table
    indices: set[int] = set()
    for record in gsub.FeatureList.FeatureRecord:
        if record.FeatureTag == "liga":
            indices.update(record.Feature.LookupListIndex)
    by_glyph = defaultdict(list)
    for char, glyph in cmap.items():
        by_glyph[glyph].append(char)

    def as_text(glyph: str) -> str | None:
        chars = by_glyph.get(glyph)
        return chr(min(chars)) if chars else None

    out: dict[str, str] = {}
    for index in sorted(indices):
        lookup = gsub.LookupList.Lookup[index]
        for subtable in lookup.SubTable:
            subtable = _unwrap(subtable)
            for first, ligatures in getattr(subtable, "ligatures", {}).items():
                head = as_text(first)
                if head is None:
                    continue
                for ligature in ligatures:
                    tail = [as_text(g) for g in ligature.Component]
                    if any(t is None for t in tail):
                        continue
                    out[head + "".join(tail)] = ligature.LigGlyph  # type: ignore[operator]
    return out


# --------------------------------------------------------------------------- сборка


def extract(path: Path) -> dict[str, Any]:
    font = TTFont(path)
    cmap = font.getBestCmap()
    hmtx = font["hmtx"]
    upem = font["head"].unitsPerEm
    os2 = font["OS/2"]

    chars = [c for c in repertoire() if ord(c) in cmap]
    ligatures = {
        text: glyph
        for text, glyph in _ligatures(font, cmap).items()
        if glyph in hmtx.metrics
    }

    # Единица измерения — то, что реально ложится в строку одним глифом: символ
    # или лигатура. Кернинг считается между единицами, поэтому и классы у них
    # общие.
    units: dict[str, str] = {c: cmap[ord(c)] for c in chars}
    units.update(ligatures)

    lookups = [
        [prepared for st in subtables if (prepared := _prepare_subtable(st)) is not None]
        for subtables in _kern_subtables(font)
    ]
    keys = sorted(units)
    left, right, matrix = _factorize(
        keys, lambda a, b: _kern_value(lookups, units[a], units[b])
    )

    return {
        "font": path.stem,
        "family": font["name"].getDebugName(1) or path.stem,
        "units_per_em": upem,
        "ascender": font["hhea"].ascent,
        "descender": font["hhea"].descent,
        "line_gap": font["hhea"].lineGap,
        "cap_height": getattr(os2, "sCapHeight", None),
        "x_height": getattr(os2, "sxHeight", None),
        # Ширина неизвестного символа. Не ноль: пропущенный глиф в Chromium
        # рисуется прямоугольником .notdef и место занимает.
        "notdef_advance": hmtx.metrics[font.getGlyphOrder()[0]][0],
        "advances": {u: hmtx.metrics[g][0] for u, g in sorted(units.items())},
        "ligatures": sorted(ligatures),
        "kern": {"left": left, "right": right, "matrix": matrix},
    }


def write(path: Path, out_dir: Path = OUT_DIR) -> Path:
    data = extract(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}.json"
    out.write_text(json.dumps(data, ensure_ascii=False, sort_keys=False), encoding="utf-8")
    return out


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted(FONTS_DIR.glob("*.ttf"))
    if not paths:
        print(f"нет шрифтов в {FONTS_DIR}", file=sys.stderr)
        return 1
    for path in paths:
        out = write(path)
        size = out.stat().st_size
        data = json.loads(out.read_text(encoding="utf-8"))
        print(
            f"{path.name}: {len(data['advances'])} символов, "
            f"{len(data['kern']['matrix'])} клеток кернинга, {size // 1024} КБ → {out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
