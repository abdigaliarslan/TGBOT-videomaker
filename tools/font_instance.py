#!/usr/bin/env python3
"""Нарезка статического начертания из переменного шрифта.

Зачем. Google Fonts отдаёт сегодня переменные шрифты: один файл с осью `wght`,
у которого ДЕФОЛТНЫЙ инстанс почти всегда 400. А `tools/font_metrics.py` читает
`hmtx` именно дефолтного инстанса — то есть по переменному файлу Python
намерил бы Regular, тогда как Chromium нарисовал бы запрошенный пресетом вес.
Промах вышел бы далеко за договорные 2 px и не локализовался бы ничем: обе
стороны честно делают своё.

Поэтому в репозиторий кладётся не переменный файл, а вырезанный из него
статический. Нарезка здесь, а не разовой командой из истории шелла: шрифт
придётся перевыпускать при обновлении Google Fonts, и повторить надо будет
ровно то же самое.

    python tools/font_instance.py <источник> <вес> [--name Имя-Файла]

Источник — путь к файлу или URL. Вес — число по оси `wght`.

    python tools/font_instance.py \\
        https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf \\
        800 --name PlayfairDisplay-ExtraBold

После нарезки обязательно выгрузить метрики, иначе геометрия упадёт в рантайме:

    python tools/font_metrics.py assets/fonts/PlayfairDisplay-ExtraBold.ttf

⚠ `updateFontNames=True` не косметика. ASS-бэкенд (`src/pipeline/subs_ass.py`)
ищет шрифт по ИМЕНИ СЕМЕЙСТВА через fontconfig, а не по файлу. Без обновления
имён все нарезки одного шрифта назывались бы одинаково, и libass выбрал бы
из них любую.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

REPO_ROOT = Path(__file__).resolve().parents[1]
FONTS_DIR = REPO_ROOT / "assets" / "fonts"


def fetch(source: str) -> bytes:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as r:  # noqa: S310 — адрес даёт человек
            return r.read()
    return Path(source).read_bytes()


def slice_static(raw: bytes, weight: float) -> TTFont:
    """Статический инстанс по оси `wght`. Шрифт без осей возвращается как есть."""
    import io

    font = TTFont(io.BytesIO(raw))
    if "fvar" not in font:
        return font
    axes = {a.axisTag: (a.minValue, a.maxValue) for a in font["fvar"].axes}
    if "wght" not in axes:
        raise SystemExit(f"у шрифта нет оси wght, есть только {sorted(axes)}")
    lo, hi = axes["wght"]
    if not lo <= weight <= hi:
        raise SystemExit(f"вес {weight} вне оси wght ({lo}…{hi})")
    return instancer.instantiateVariableFont(font, {"wght": weight}, updateFontNames=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="путь к ttf или URL")
    ap.add_argument("weight", type=float, help="вес по оси wght")
    ap.add_argument("--name", help="имя файла без расширения (по умолчанию — из имён шрифта)")
    ap.add_argument("--out-dir", type=Path, default=FONTS_DIR)
    args = ap.parse_args(argv)

    font = slice_static(fetch(args.source), args.weight)
    family = font["name"].getDebugName(1) or "Font"
    name = args.name or family.replace(" ", "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{name}.ttf"
    font.save(out)

    os2 = font["OS/2"]
    upem = font["head"].unitsPerEm
    print(
        f"{out.relative_to(REPO_ROOT)}: семейство {family!r}, "
        f"usWeightClass {os2.usWeightClass}, "
        f"высота прописных {os2.sCapHeight / upem:.3f} em, {out.stat().st_size // 1024} КБ"
    )
    print(f"теперь выгрузите метрики: python tools/font_metrics.py {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
