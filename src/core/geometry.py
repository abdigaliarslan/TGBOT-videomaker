"""Геометрия кадра в пикселях: Python меряет текст сам, а не гадает по половинам.

Инвариант §0: рендерер ничего не решает. До сих пор он его нарушал — половина
геометрии жила константами в TSX (`INSET_RATIO`, `SAFE_AREA_RATIO`, `0.16`,
`0.58`), а Python оперировал «верхней» и «нижней» половиной кадра и поэтому не
мог ни предсказать, куда попадёт строка, ни развести субтитр с картинкой
(AUDIT §8.4). Здесь это чинится: модуль умеет измерить строку в пикселях теми же
числами, которыми её считает Chromium, и посчитать прямоугольники, в которые
попадут субтитр и картинка.

Как меряется текст. Ширина строки — это сумма advance-ширин глифов из TTF плюс
кернинг из GPOS плюс letter-spacing на каждый символ. Все три слагаемых берутся
из `presets/metrics/<шрифт>.json`, который делает `tools/font_metrics.py`.
Кернинг не роскошь: пара «ГА» в Montserrat это −0.138 em, на кегле 88 — почти
12 пикселей, шесть бюджетов допуска DoD Ф4.

Три места, где браузер ведёт себя не так, как наивная сумма advance-ширин, и
каждое замерено линейкой `render/measure.mjs` в том самом Chromium, которым идёт
рендер (числа — в комментариях у соответствующих функций):

1. Кернинг НЕ работает через границу письменности: HarfBuzz режет текст на
   прогоны по скрипту, и пара «б|a» в «Сбербankовский» не кернится. Наивная
   модель занижала это слово на 3.2 px — полтора бюджета DoD (`_script`).
2. Пробелы по краям слова схлопываются: слово в кадре — inline-block, и
   CSS убирает у него краевой пробел, а идущие подряд пробелы сводит в один
   (`_collapse`). Без этого « пробел » мерился на 48 px шире.
3. Эмодзи рисует не наш TTF, а системный цветной шрифт, и весь ZWJ-кластер
   («👨‍👩‍👧») занимает ОДНУ ширину и получает ОДИН letter-spacing, а не по
   одному на кодовую точку (`EMOJI_ADVANCE_EM`, `_emoji_cluster`).

Чего модуль сознательно не умеет: символ, которого НЕТ В ШРИФТЕ и который не
эмодзи, рисуется фолбэком системы, и его ширину этот TTF не знает — для него
берётся `FALLBACK_ADVANCE_EM`, оценка, а не измерение. Допуск 2 px на эту ветку
не распространяется, и распространяться не может: какой шрифт подставит
Chromium, решает система хоста. Замерено на этой машине: «✓» 72.42 px против
оценки 86.68, «▪» 29.89. Оценка сознательно ЗАВЫШЕНА (1 em) — прямоугольник
блока выходит шире настоящего, и картинка отодвигается с запасом, а не
наезжает. Всё, что в шрифте ЕСТЬ, обязано меряться шрифтом: репертуар выгрузки
`tools/font_metrics.py` — это весь cmap, а не список диапазонов (раньше был
список, и 452 символа шрифта уходили в оценку с промахом до 80 px).
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

log = logging.getLogger(__name__)

METRICS_DIR = Path(__file__).resolve().parent / "presets" / "metrics"

# Ширина символа, которого нет в шрифте и который не эмодзи. Точно измерить
# чужой шрифт мы не можем: какой фолбэк подставит Chromium, зависит от системы,
# а не от пресета. Единица — грубая оценка, и DoD «меньше 2 px» на неё не
# распространяется: замер на этой машине дал «✓» 72.42, «α» 47.80, «ஃ» 77.53
# против 86.68 у модели. Оценка выбрана ЗАВЫШЕННОЙ намеренно — промах наружу
# делает блок шире настоящего и отодвигает картинку с запасом, промах внутрь
# поставил бы её на текст. Единственный известный фолбэк шире em — цветной
# эмодзи, и для него есть отдельная ветка (см. EMOJI_ADVANCE_EM).
FALLBACK_ADVANCE_EM = 1.0

# Ширина эмодзи-кластера в em — сам глиф, без letter-spacing. Цветной глиф рисует
# не наш TTF, а системный Noto Color Emoji, метрик у него в пресете нет. Число
# измерено линейкой `render/measure.mjs` в том же Chromium, которым идёт рендер:
# 🔥 на кегле 88 с letter-spacing −0.015 занимает 108.48 px, то есть 1.2478 em
# минус один спейсинг. Эмодзи из одной, двух и пяти кодовых точек (🔥, ❤️, 1️⃣,
# 👨‍👩‍👧) дали ровно ту же ширину — значит меряется КЛАСТЕР целиком и спейсинг
# Chromium добавляет к нему один раз, а не по разу на кодовую точку.
#
# Число зависит от ХОСТА: цветной шрифт ставит система, а не пресет. Другой
# эмодзи-шрифт — другая ширина, и никакой метрикой мы это не поймаем. Пока
# рендер идёт на своей машине, это допущение; при переезде в образ эмодзи-шрифт
# надо закреплять там же, где и Montserrat.
EMOJI_ADVANCE_EM = 1.2478


# --------------------------------------------------------------------- текст глазами браузера


def _round_px(value: float) -> int:
    """Округление «половина вверх» — как `Math.round` в TSX, а не как `round`.

    Питоновский `round` банковский: `round(2.5)` это 2, а `Math.round(2.5)` — 3.
    Числа геометрии считаются по обе стороны (`line_gap`, `safe_margin_y` есть и
    в Captions.tsx), и на ровной половинке они расходились на пиксель — молча и
    только на некоторых высотах кадра после `Preset.scaled`.
    """
    return int(math.floor(value + 0.5))


# Границы письменностей, по которым HarfBuzz режет текст на прогоны. Кернинг
# внутри прогона работает, ЧЕРЕЗ границу — нет: Chromium шейпит прогоны
# независимо. Замерено на «Сбербankовский»: пары «б|a» (−12) и «k|о» (−24) в
# сумме дают 3.2 px на кегле 88, полтора бюджета DoD, и наивная модель ровно на
# столько занижала слово.
_CYRILLIC = ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))
_GREEK = ((0x0370, 0x03FF), (0x1F00, 0x1FFF))
_LATIN = ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF))


def _script(unit: str) -> str:
    """Письменность единицы: `lat`, `cyr`, `grk` или `""` для нейтральных.

    Нейтральные — цифры, пунктуация, пробел, эмодзи: собственной письменности у
    них нет, её они получают от соседей (`_resolve_scripts`).
    """
    code = ord(unit[0])
    for ranges, name in ((_LATIN, "lat"), (_CYRILLIC, "cyr"), (_GREEK, "grk")):
        for lo, hi in ranges:
            if lo <= code <= hi:
                return name
    return ""


def _resolve_scripts(units: Sequence[str]) -> list[str]:
    """Письменность каждой единицы после итемизации.

    Нейтральный символ прилипает к тому, что СЛЕВА (так делает ICU), а в начале
    строки — к тому, что справа. Отсюда и берётся граница в «PDF-файл»: дефис
    уходит в латиницу, и пара «-|ф» кернинг не получает, хотя в шрифте он есть
    (+10 единиц, 0.9 px на кегле 88 — ровно на столько наивная модель и
    расходилась с Chromium).
    """
    out = [_script(u) for u in units]
    previous = ""
    for i, script in enumerate(out):
        if script:
            previous = script
        else:
            out[i] = previous
    following = ""
    for i in range(len(out) - 1, -1, -1):
        if out[i]:
            following = out[i]
        else:
            out[i] = following
    return out


# Эмодзи. Точное определение из UTS #51 сюда не нужно, но одно его правило —
# нужно, и без него правило врало: НАЧЕРТАНИЕ. Диапазоны ниже — это «символ,
# который МОЖЕТ быть эмодзи» (`Emoji=Yes`), и половина из них по умолчанию
# рисуется чёрно-белым текстовым глифом: «▪» в Chromium занимает 29.89 px, «✓» —
# 72.42, а эмодзи-ширина дала бы обоим 108.50. Одного диапазона мало ещё и
# потому, что «есть ли символ в шрифте» — свойство ВЫГРУЗКИ: стоило репертуару
# `tools/font_metrics.py` не покрыть «▲», и он молча становился эмодзи.
#
# Поэтому цветным считается только то, у чего цвет по умолчанию
# (`_EMOJI_PRESENTATION`), либо то, что попросило цвет явно: селектор VS16,
# хвост клавиши, тон кожи, ZWJ, пара региональных индикаторов. Проверено
# линейкой: «☑» 77.58 px против «☑️» (тот же символ с VS16) 108.48 — признак
# начертания и есть недостающее условие.
_EMOJI_BASE = (
    (0x203C, 0x2049), (0x2122, 0x2122), (0x2139, 0x2139), (0x2194, 0x21AA),
    (0x231A, 0x231B), (0x2328, 0x2328), (0x23CF, 0x23FA), (0x24C2, 0x24C2),
    (0x25AA, 0x25FE), (0x2600, 0x27BF), (0x2934, 0x2935), (0x2B00, 0x2B55),
    (0x3030, 0x3030), (0x303D, 0x303D), (0x3297, 0x3299),
    (0x1F000, 0x1FAFF),
)
# `Emoji_Presentation=Yes` — символы, которые Chromium рисует цветным глифом БЕЗ
# всякого селектора. Список для BMP короткий и его проще выписать, чем тащить
# таблицу UCD: всё остальное из `_EMOJI_BASE` в BMP по умолчанию текстовое.
# Выше BMP (`0x1F000+`) исключений почти нет, и все они — знаки вроде «🅰», в
# кадре не встречающиеся, поэтому там диапазон берётся целиком.
_EMOJI_PRESENTATION = (
    (0x231A, 0x231B), (0x23E9, 0x23EC), (0x23F0, 0x23F0), (0x23F3, 0x23F3),
    (0x25FD, 0x25FE), (0x2614, 0x2615), (0x2648, 0x2653), (0x267F, 0x267F),
    (0x2693, 0x2693), (0x26A1, 0x26A1), (0x26AA, 0x26AB), (0x26BD, 0x26BE),
    (0x26C4, 0x26C5), (0x26CE, 0x26CE), (0x26D4, 0x26D4), (0x26EA, 0x26EA),
    (0x26F2, 0x26F3), (0x26F5, 0x26F5), (0x26FA, 0x26FA), (0x26FD, 0x26FD),
    (0x2705, 0x2705), (0x270A, 0x270B), (0x2728, 0x2728), (0x274C, 0x274C),
    (0x274E, 0x274E), (0x2753, 0x2755), (0x2757, 0x2757), (0x2795, 0x2797),
    (0x27B0, 0x27B0), (0x27BF, 0x27BF), (0x2B1B, 0x2B1C), (0x2B50, 0x2B50),
    (0x2B55, 0x2B55),
    (0x1F000, 0x1FAFF),
)
# Что прилипает к базе, не заводя новый кластер. Кодами, а не литералами: все
# эти символы невидимы, и опечатку в литерале не увидит никто.
_ZWJ = "‍"  # соединитель: 👨‍👩‍👧 это пять кодовых точек и одна ширина
_VS = ("︎", "️")  # селекторы начертания: текстовое и эмодзи
_KEYCAP = "⃣"  # хвост клавиши: 1️⃣
_MODIFIER = frozenset((*_VS, _KEYCAP))
# База клавиши — обычный символ, эмодзи из него делает только хвост.
_KEYCAP_BASE = frozenset("#*0123456789")


def _in_ranges(code: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= code <= hi for lo, hi in ranges)


def _is_emoji_base(ch: str) -> bool:
    return _in_ranges(ord(ch), _EMOJI_BASE)


def _is_emoji_by_default(ch: str) -> bool:
    """Рисуется ли символ цветным глифом БЕЗ селектора начертания."""
    return _in_ranges(ord(ch), _EMOJI_PRESENTATION)


def _is_regional(ch: str) -> bool:
    return 0x1F1E6 <= ord(ch) <= 0x1F1FF


def _is_skin_tone(ch: str) -> bool:
    return 0x1F3FB <= ord(ch) <= 0x1F3FF


def _emoji_cluster(text: str, i: int) -> int:
    """Длина эмодзи-кластера, начинающегося в позиции i. 0 — там не эмодзи.

    Кластер, а не кодовая точка: «👨‍👩‍👧» это пять кодовых точек, но в кадре одна
    ширина и один letter-spacing. Два соседних эмодзи — два кластера: склеивать
    их значит занизить строку вдвое, поэтому продолжение кластера разрешено
    только модификаторам и явному соединителю.

    Одинокая база ТЕКСТОВОГО начертания кластером не считается: «✓» и «▪» рисует
    обычный шрифт, и мерить их эмодзи-шириной значит промахнуться на 36-79 px.
    Цвет даёт либо начертание по умолчанию, либо явная просьба — VS16, хвост
    клавиши, тон кожи, ZWJ. Селектор ТЕКСТОВОГО начертания (VS15) отменяет цвет
    даже у эмодзи по умолчанию.
    """
    if _is_regional(text[i]):  # флаг — ровно пара индикаторов
        return 2 if i + 1 < len(text) and _is_regional(text[i + 1]) else 1
    keycap = text[i] in _KEYCAP_BASE and (
        text.startswith(_VS[1] + _KEYCAP, i + 1) or text.startswith(_KEYCAP, i + 1)
    )
    if not keycap and not _is_emoji_base(text[i]):
        return 0
    j = i + 1
    asked = keycap  # цвет попросили явно
    while j < len(text):
        if text[j] == _VS[0]:  # VS15 — «рисуй текстом», кластера здесь нет
            return 0
        if text[j] == _VS[1] or text[j] == _KEYCAP or _is_skin_tone(text[j]):
            asked = True
            j += 1
        elif text[j] == _ZWJ and j + 1 < len(text) and _is_emoji_base(text[j + 1]):
            asked = True
            j += 2  # соединитель тянет за собой следующую базу
        else:
            break
    if not asked and not _is_emoji_by_default(text[i]):
        return 0
    return j - i


# Пробельные символы `white-space: normal` — ровно набор CSS, без питоновского
# U+00A0 и вертикальной табуляции.
_CSS_SPACE = re.compile(r"[ \t\n\r\f]+")


def _collapse(text: str) -> str:
    """Текст так, как его увидит браузер внутри бокса слова.

    Слово в кадре — `display: inline-block`, то есть собственный контекст: CSS
    убирает у него краевые пробелы, а идущие подряд сводит в один. Замерено:
    « пробел » в Chromium 348.92 px против 397.50 у наивной модели — 48 px, два
    десятка бюджетов DoD.

    Схлопывается ВЕСЬ набор `white-space: normal` (пробел, таб, перевод строки,
    возврат каретки, подача листа), а не один U+0020: для CSS они одинаковы, а в
    шрифте таба и перевода строки нет, и «а\tб» уходило в фолбэк — 197.65 px
    против 135.27 в Chromium. `text.strip()` на входе снимает только краевые,
    внутренний перевод строки из ручного EDL доживал до плана.

    Неразрывный пробел не трогаем: он не схлопывается и ширину имеет (проверено
    на «а б»), тонкие U+2009/U+200A — тоже, они есть в шрифте. Питоновский
    `str.split()` здесь не годится ровно поэтому: он считает пробельным и U+00A0.
    """
    if not _CSS_SPACE.search(text):
        return text
    return _CSS_SPACE.sub(" ", text).strip(" ")


# --------------------------------------------------------------------------- прямоугольник


@dataclass(frozen=True, slots=True)
class Rect:
    """Прямоугольник в пикселях кадра. Начало координат — левый верхний угол."""

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    def overlaps(self, other: "Rect", *, gap: float = 0.0) -> bool:
        """Пересекаются ли прямоугольники с зазором `gap`.

        Это то самое «настоящее пересечение вместо половин кадра», ради которого
        прямоугольники и появились: картинка мешает субтитру не тогда, когда она
        в той же половине, а тогда, когда она его накрывает.
        """
        return (
            self.x < other.right + gap
            and other.x < self.right + gap
            and self.y < other.bottom + gap
            and other.y < self.bottom + gap
        )

    def inset(self, dx: float, dy: float | None = None) -> "Rect":
        dy = dx if dy is None else dy
        return Rect(self.x + dx, self.y + dy, self.width - 2 * dx, self.height - 2 * dy)

    def rounded(self, digits: int = 2) -> "Rect":
        return Rect(
            round(self.x, digits),
            round(self.y, digits),
            round(self.width, digits),
            round(self.height, digits),
        )

    def to_dict(self) -> dict[str, float]:
        r = self.rounded()
        return {"x": r.x, "y": r.y, "width": r.width, "height": r.height}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rect":
        return cls(
            float(data["x"]), float(data["y"]), float(data["width"]), float(data["height"])
        )


# --------------------------------------------------------------------------- метрики шрифта


@dataclass(frozen=True, slots=True)
class FontMetrics:
    """Метрики TTF в единицах шрифта. Читается из `presets/metrics/<шрифт>.json`."""

    name: str
    family: str
    units_per_em: int
    ascender: int
    descender: int
    line_gap: int
    cap_height: int | None
    x_height: int | None
    # Ширина .notdef ЭТОГО шрифта. Лежит в выгрузке, но в расчёте не участвует
    # намеренно: пустой квадратик рисует не Montserrat, а тот фолбэк, который
    # Chromium подобрал под символ, и его .notdef другой ширины. Врать точным
    # числом от чужого шрифта хуже, чем честно взять `FALLBACK_ADVANCE_EM`.
    notdef_advance: int
    advances: dict[str, int]
    ligatures: tuple[str, ...]
    kern_left: dict[str, int]
    kern_right: dict[str, int]
    kern_matrix: dict[str, int]

    @classmethod
    def from_dict(cls, data: dict[str, Any], name: str = "") -> "FontMetrics":
        kern = data.get("kern", {})
        return cls(
            name=data.get("font", name),
            family=data.get("family", name),
            units_per_em=int(data["units_per_em"]),
            ascender=int(data["ascender"]),
            descender=int(data["descender"]),
            line_gap=int(data.get("line_gap", 0)),
            cap_height=data.get("cap_height"),
            x_height=data.get("x_height"),
            notdef_advance=int(data.get("notdef_advance", 0)),
            advances=dict(data["advances"]),
            ligatures=tuple(sorted(data.get("ligatures", ()), key=len, reverse=True)),
            kern_left=dict(kern.get("left", {})),
            kern_right=dict(kern.get("right", {})),
            kern_matrix=dict(kern.get("matrix", {})),
        )

    # -- разбор текста ------------------------------------------------------

    def split(self, text: str, *, ligatures: bool = True) -> list[str]:
        """Текст → последовательность «единиц»: то, что ложится в строку одним шагом.

        Единица — символ, лигатура или эмодзи-кластер целиком. Кластер не режется
        на кодовые точки, потому что Chromium его не режет: «👨‍👩‍👧» получает одну
        ширину и один letter-spacing, а не пять.

        Лигатуры отключаются, когда задан letter-spacing: Chromium в этом случае
        не склеивает `fi`, иначе спейсинг было бы некуда вставить.
        """
        out: list[str] = []
        i = 0
        while i < len(text):
            size = _emoji_cluster(text, i)
            # Своя буква эмодзи-правилом не перехватывается — и это свойство
            # НАЧЕРТАНИЯ, а не выгрузки: `_emoji_cluster` не считает кластером
            # одинокий символ текстового начертания, поэтому «▲» и «✓» останутся
            # текстом, даже если завтра пропадут из `advances`. Кластер длиннее
            # одной кодовой точки перехватывается всегда: «1️⃣» начинается обычной
            # единицей, а рисуется цветной клавишей, и «☑️» (тот же «☑», но с
            # VS16) — тоже: 108.48 px против 77.58 у текстового начертания.
            if size and (size > 1 or text[i] not in self.advances):
                out.append(text[i : i + size])
                i += size
                continue
            if ligatures:
                for lig in self.ligatures:  # отсортированы по убыванию длины
                    if text.startswith(lig, i):
                        out.append(lig)
                        i += len(lig)
                        break
                else:
                    out.append(text[i])
                    i += 1
                continue
            out.append(text[i])
            i += 1
        return out

    def advance(self, unit: str) -> int:
        """Ширина единицы в единицах шрифта."""
        found = self.advances.get(unit)
        if found is not None:
            return found
        if _emoji_cluster(unit, 0) == len(unit):
            return round(EMOJI_ADVANCE_EM * self.units_per_em)
        if all(unicodedata.category(c) in ("Mn", "Me", "Cf") for c in unit):
            # Диакритика и невидимые управляющие ширины не занимают ни в одном
            # шрифте: селектор начертания, оставшийся без своей базы, стоил бы
            # иначе целый em (88 px), которого в кадре нет.
            return 0
        return round(FALLBACK_ADVANCE_EM * self.units_per_em)

    def kern(self, left: str, right: str) -> int:
        """Кернинг пары по GPOS, в единицах шрифта (обычно отрицательный).

        Это то, что говорит шрифт. Применит ли его браузер — решает контекст:
        через границу письменности не применит (см. `advance_em`).
        """
        l = self.kern_left.get(left)  # noqa: E741 — это сторона пары, не единица длины
        if l is None:
            return 0
        r = self.kern_right.get(right)
        if r is None:
            return 0
        return self.kern_matrix.get(f"{l}:{r}", 0)

    def advance_em(self, text: str, *, ligatures: bool = True) -> float:
        """Ширина текста в em, с кернингом и лигатурами, без letter-spacing."""
        units = self.split(text, ligatures=ligatures)
        if not units:
            return 0.0
        total = sum(self.advance(u) for u in units)
        scripts = _resolve_scripts(units)
        total += sum(
            self.kern(a, b) if sa == sb else 0
            for a, b, sa, sb in zip(units, units[1:], scripts, scripts[1:])
        )
        return total / self.units_per_em

    def units_count(self, text: str, *, ligatures: bool = True) -> int:
        return len(self.split(text, ligatures=ligatures))

    def spacing_units(self, text: str, *, ligatures: bool = True) -> int:
        """Сколько раз к тексту прибавится letter-spacing.

        Не то же самое, что число единиц: Chromium не даёт спейсинг тому, у чего
        нет собственной ширины. Ударение в «и́мя» и селектор начертания в «→️»
        рисуются поверх соседа, advance у них ноль — и спейсинга они не получают.
        Замерено: «и́мя» на кегле 88 с ls −0.015 это 188.06 px в браузере, а
        модель со счётом «по единицам» давала 186.74 — минус 1.32 px на каждый
        знак, и на bold (ls −0.02, кегль 104) одного знака хватало, чтобы выйти
        за DoD 2 px.
        """
        return sum(1 for u in self.split(text, ligatures=ligatures) if self.advance(u))


@lru_cache(maxsize=8)
def load_metrics(font_file: str) -> FontMetrics:
    """Метрики по имени файла шрифта из пресета (`Montserrat-ExtraBold.ttf`)."""
    stem = Path(font_file).stem
    path = METRICS_DIR / f"{stem}.json"
    if not path.exists():
        have = ", ".join(sorted(p.stem for p in METRICS_DIR.glob("*.json"))) or "—"
        raise FileNotFoundError(
            f"нет метрик шрифта {stem!r} в {METRICS_DIR}; есть: {have}. "
            f"Сгенерировать: python tools/font_metrics.py"
        )
    return FontMetrics.from_dict(json.loads(path.read_text(encoding="utf-8")), stem)


def metrics_for(preset: Any) -> FontMetrics:
    """Метрики шрифта пресета. Без `font.file` мерить нечего — это ошибка данных."""
    file = getattr(preset.font, "file", None)
    if not file:
        raise ValueError(f"в пресете {preset.name!r} не задан font.file — нечем мерить текст")
    return load_metrics(file)


# --------------------------------------------------------------------------- измерение


@dataclass(frozen=True, slots=True)
class SizedWord:
    """Слово со своим кеглем: акцент крупнее соседей и занимает больше места."""

    text: str
    size: float


class _HasEmphasis(Protocol):
    text: str
    emphasis: str
    emoji: str | None


def measure_word(
    text: str,
    font_size: float,
    *,
    metrics: FontMetrics,
    letter_spacing: float = 0.0,
    spacing_size: float | None = None,
    uppercase: bool = False,
) -> float:
    """Ширина слова в пикселях.

    Два поведения Chromium, которые нельзя угадать и которые проверены линейкой
    из `render/measure.mjs`:

    1. Спейсинг добавляется после КАЖДОГО символа, включая последний, поэтому
       множитель — число символов, а не число промежутков между ними. Но только
       тех, у кого есть своя ширина: диакритика и селекторы начертания спейсинга
       не получают (`FontMetrics.spacing_units`).
    2. Ненулевой спейсинг выключает лигатуры: `affine` меряется шестью глифами,
       а не пятью (проверено на 0.1 em, где разница не тонет в округлении).

    `spacing_size` — кегль, ОТ КОТОРОГО считаются em спейсинга. По умолчанию это
    собственный кегль слова, но в субтитре `letter-spacing` объявлен на общем
    контейнере строки и наследуется уже в пикселях: акцентное слово крупнее, а
    спейсинг у него тот же, что у соседей. Считать em от кегля акцента —
    ошибка в 1.6 px на слове из семи букв.

    Пробелы по краям слова НЕ считаются: бокс слова — inline-block, и CSS их
    убирает (`_collapse`). Ширину одиночного пробела между словами берёт
    `space_width`: там пробел лежит между боксами и схлопнуться ему не с чем.
    """
    if not text:
        return 0.0
    if uppercase:
        text = text.upper()
    text = _collapse(text)
    if not text:
        return 0.0
    use_ligatures = letter_spacing == 0.0
    width = metrics.advance_em(text, ligatures=use_ligatures) * font_size
    if letter_spacing:
        spacing_px = letter_spacing * (font_size if spacing_size is None else spacing_size)
        width += spacing_px * metrics.spacing_units(text, ligatures=use_ligatures)
    return width


def space_width(font_size: float, *, metrics: FontMetrics, letter_spacing: float = 0.0) -> float:
    """Ширина пробела МЕЖДУ словами.

    Отдельная функция, а не `measure_word(" ")`: в боксе слова краевой пробел
    схлопывается и стоит ноль, а между двумя inline-block-боксами он настоящий и
    рисуется кеглем строки. Одной функцией эти два случая не описать.
    """
    return (metrics.advance(" ") / metrics.units_per_em + letter_spacing) * font_size


def measure_line(
    words: Sequence[SizedWord | tuple[str, float]],
    *,
    metrics: FontMetrics,
    letter_spacing: float = 0.0,
    base_size: float | None = None,
    uppercase: bool = False,
) -> float:
    """Ширина строки в пикселях с учётом разных кеглей внутри неё.

    Слова разделяет НАСТОЯЩИЙ пробел, а не flex-`gap`: `gap` в строке с разными
    кеглями не масштабируется и врёт тем сильнее, чем крупнее акцент. Пробел
    рисуется кеглем строки (`base_size`), потому что в DOM он лежит в родителе,
    а не внутри спана слова.

    Кернинг через границу слова не считается: слова — отдельные inline-элементы,
    Chromium шейпит их независимо.

    `base_size` — кегль строки. От него считается и пробел, и letter-spacing всех
    слов, включая акцентное (см. `measure_word`). По умолчанию берётся самый
    мелкий кегль в строке: акцент бывает только крупнее базового, никогда мельче.

    Слово с пустым текстом из строки НЕ выбрасывается, хотя ширина у него ноль:
    вместе с ним пропал бы разделяющий пробел, а он в кадре есть. Captions.tsx
    ставит пробел перед каждым словом с индексом больше нуля, не заглядывая в его
    текст, и Chromium это подтверждает: строка «раз»/«»/«два» это 377.81 px, а не
    353.52. Ровно на этот пробел «независимая линейка» тестов расходилась с
    производственным `_Ruler.line_width`.
    """
    items = [w if isinstance(w, SizedWord) else SizedWord(w[0], w[1]) for w in words]
    if not items:
        return 0.0
    if base_size is None:
        base_size = min(w.size for w in items)
    total = sum(
        measure_word(
            w.text,
            w.size,
            metrics=metrics,
            letter_spacing=letter_spacing,
            spacing_size=base_size,
            uppercase=uppercase,
        )
        for w in items
    )
    space = space_width(base_size, metrics=metrics, letter_spacing=letter_spacing)
    return total + space * (len(items) - 1)


def emphasis_size(emphasis: str, preset: Any) -> float:
    """Кегль слова по его выделению.

    Акцент — это РАЗМЕР, а не `transform: scale`. Масштаб не участвует в
    раскладке, поэтому увеличенное слово наезжало на соседа (AUDIT §8.1); кегль
    участвует, и flex разводит слова сам. Здесь Python считает так же, как это
    будет рисовать рендерер после Ф5.
    """
    base = preset.font.size
    if emphasis == "accent":
        return base * preset.animation.accent_scale
    if emphasis == "shout":
        return base * preset.animation.shout_scale
    return float(base)


def sized_words(words: Iterable[_HasEmphasis], preset: Any) -> list[SizedWord]:
    """Слова плана → слова с кеглями пресета.

    Эмодзи входит в текст слова: в кадре он лежит внутри спана слова, за
    пробелом и его кеглем (Captions.tsx). Забыть его — значит занизить строку на
    полторы ширины em, то есть на 130 px при кегле 88.
    """
    return [
        SizedWord(
            f"{w.text} {w.emoji}" if getattr(w, "emoji", None) else w.text,
            emphasis_size(getattr(w, "emphasis", "none"), preset),
        )
        for w in words
    ]


# --------------------------------------------------------------------------- субтитр


def safe_margin_x(preset: Any, frame_width: int) -> float:
    return frame_width * preset.geometry.safe_area_ratio


def safe_margin_y(preset: Any, frame_height: int) -> float:
    """Отступ субтитра от края: пресет может попросить меньше, безопасная зона — нет.

    Прижатый к краю текст обрезает интерфейс ленты, этого не хочет ни один
    пресет, поэтому нижняя граница жёсткая (так же считает Captions.tsx).
    """
    return max(preset.caption.margin, _round_px(frame_height * preset.geometry.safe_area_ratio))


def caption_text_width(preset: Any, frame_width: int) -> float:
    """Сколько пикселей отведено под САМ ТЕКСТ строки — бюджет переноса для Ф5.

    Два ограничения сразу: доля кадра из пресета и безопасная зона за вычетом
    паддингов панели. Второе важнее: без него панель с широким текстом вылезает
    за безопасную зону, хотя текст формально в долю уложился.
    """
    panel = preset.panel
    chrome = 2 * (panel.padding_x + panel.border_width) if panel.enabled else 0.0
    by_ratio = frame_width * preset.geometry.max_line_width_ratio
    by_safe = frame_width - 2 * safe_margin_x(preset, frame_width) - chrome
    return max(0.0, min(by_ratio, by_safe))


def line_gap(preset: Any) -> float:
    """Просвет между строками блока (flex-`gap` в Captions.tsx)."""
    return float(_round_px(preset.font.size * (preset.caption.line_height - 1)))


def line_box_height(sizes: Iterable[float], preset: Any) -> float:
    """Высота строчного бокса по кеглям слов в строке.

    `line-height` задан числом, значит наследуется множителем и считается от
    собственного кегля элемента: бокс поднимает самое крупное слово. Но не
    бывает ниже СТРУТА — кегля самого контейнера строки, то есть кегля пресета:
    как бы сильно перенос ни ужал слова, строка остаётся высотой
    `font.size * line_height` (замерено в Chromium: строка из слова кеглем 35
    занимает 93.27 px, ровно струт, а не 37).

    Единственная реализация на оба пути — `caption_rect` и `_rect_of_lines`.
    Пока их было две, вторая знала про струт, а первая нет.
    """
    biggest = max([float(preset.font.size), *(float(s) for s in sizes)])
    return biggest * preset.caption.line_height


def block_rect(
    position: str,
    text_w: float,
    text_h: float,
    preset: Any,
    frame_size: tuple[int, int],
    *,
    left: float | None = None,
) -> Rect:
    """Коробка блока вокруг текста: паддинги, рамка и привязка к краю кадра.

    Одно место на оба пути расчёта субтитра. Числа обязаны совпадать с
    Captions.tsx: паддинг и рамка панели, отступ от края с нижней границей по
    безопасной зоне, центрирование по горизонтали (`caption.align` управляет
    выключкой ВНУТРИ панели, а не местом панели в кадре).

    `left` — левый край блока в пикселях, когда он прижат к краю кадра, а не
    центрирован: так стоит тихий регистр субтитра (STYLE_PLAN §1.4.1, «влево, у
    левого края»). `None` — центр, то есть единственное поведение, какое было до
    второго регистра. Число приходит готовым (`Quiet.left_x`), и тем же числом
    Captions.tsx ставит `paddingLeft`: иначе прямоугольник плана обещал бы одно
    место, а Chromium рисовал другое.
    """
    width, height = frame_size
    panel = preset.panel
    chrome_x = 2 * (panel.padding_x + panel.border_width) if panel.enabled else 0.0
    chrome_y = 2 * (panel.padding_y + panel.border_width) if panel.enabled else 0.0
    box_w = text_w + chrome_x
    box_h = text_h + chrome_y

    margin = safe_margin_y(preset, height)
    if position == "top":
        y = float(margin)
    elif position == "center":
        y = (height - box_h) / 2
    else:
        y = height - margin - box_h
    return Rect((width - box_w) / 2 if left is None else left, y, box_w, box_h)


class _HasLines(Protocol):
    position: str
    lines: list[list[Any]]


def caption_rect(block: _HasLines, preset: Any, frame_size: tuple[int, int]) -> Rect:
    """Прямоугольник, который займёт блок субтитра, — вместе с панелью.

    Кегли выводятся из ВЫДЕЛЕНИЯ слов (`sized_words`), поэтому функция отвечает
    на вопрос «куда попадёт вот такой блок», а не «куда попал вот этот». В план
    едет прямоугольник от `render_plan._rect_of_lines`: там кегли уже посчитаны
    переносом (длинное слово ужато) и ширина взята из него же. Обе функции
    считают коробку и привязку к краю одним `block_rect` и высоту строки одним
    `line_box_height` — разъехаться им негде.

    Два условия на входе. Пресет уже приведён к высоте кадра (`Preset.scaled`):
    кегли и паддинги берутся как есть. Текст слов уже в экранной форме
    (`display_text`): регистр и пунктуация решены до нас, второй раз их трогать
    нельзя, иначе ширина разойдётся с тем, что нарисует рендерер.

    Ширина текста ограничена бюджетом `caption_text_width`: уложить слова в
    бюджет — задача переноса строк (Ф5), и блок, который в него не уложили,
    здесь не расширяется. `_rect_of_lines` в этом месте ведёт себя иначе и не
    зажимает ничего — сознательно, см. его комментарий.
    """
    width, _ = frame_size
    metrics = metrics_for(preset)
    ls = preset.font.letter_spacing
    lines = [sized_words(line, preset) for line in block.lines if line]

    text_w = 0.0
    text_h = 0.0
    for i, line in enumerate(lines):
        text_w = max(
            text_w,
            measure_line(
                line, metrics=metrics, letter_spacing=ls, base_size=float(preset.font.size)
            ),
        )
        text_h += line_box_height((w.size for w in line), preset)
        if i:
            text_h += line_gap(preset)
    text_w = min(text_w, caption_text_width(preset, width))
    return block_rect(block.position, text_w, text_h, preset, frame_size)


# --------------------------------------------------------------------------- картинки

# Пропорция (ширина/высота), когда файл не измерен. Новые вставки всегда несут
# свою (`Broll.asset_aspect`), так что число нужно только листам, собранным до
# замера: их картинки были фотографиями 2:3, и другое значение обрезало бы их
# при перерендере. Квадратная карточка срезала у такого источника
# `object-fit: cover`-ом треть высоты (AUDIT §8.3); карточка, наследующая
# пропорцию источника, не режет ничего.
FALLBACK_ASPECT = 2 / 3

# Позиции, которые контракт разрешает блоку субтитра (`CaptionBlock.position`
# минус `auto`, который разрешается в одну из них). Резерв обязан считаться по
# худшему случаю: занизить его — значит поставить картинку туда, куда потом
# придёт текст.
CAPTION_POSITIONS = ("top", "center", "bottom")

# Ниже этой доли высоты кадра карточка перестаёт быть иллюстрацией: при 2:3 это
# 154×230 px в кадре 1080×1920 — марка, которую не рассмотреть. Если свободной
# полосы осталось меньше, вставка отдаётся полным кадром: осознанный полный кадр
# честнее карточки, поставленной поверх субтитра.
#
# Это ЗНАЧЕНИЕ ПО УМОЛЧАНИЮ, а не константа модуля: доля решает, будет ли у
# пресета карточка вообще, и подкрутить её обязан уметь пресет. Читается через
# `min_band_ratio(preset)`: поле `geometry.min_band_ratio` перекрывает.
MIN_BAND_RATIO = 0.12


def min_band_ratio(preset: Any) -> float:
    """Порог полосы: из пресета, если он его задал, иначе `MIN_BAND_RATIO`."""
    value = getattr(getattr(preset, "geometry", None), "min_band_ratio", None)
    return float(value) if value else MIN_BAND_RATIO


def caption_positions(preset: Any) -> tuple[str, ...]:
    """Позиции, которые блок может получить в этом пресете БЕЗ явного указания.

    `auto` разрешается в `preset.caption.position` (`bold` — это `center`), явные
    `top` и `bottom` режиссёр ставит в каждом втором ролике. Явный `center`
    контракт тоже разрешает, но предсказать его отсюда нечем: b-roll считается
    раньше субтитров, и блоков в этот момент ещё нет — поэтому он попадает в
    строгий набор `CAPTION_POSITIONS`, а не сюда (см. `broll_free_band`).
    """
    base = getattr(preset.caption, "position", "bottom")
    if base not in CAPTION_POSITIONS:
        base = "bottom"
    if getattr(preset.caption, "pinned", False):
        return (base,)
    return tuple(dict.fromkeys(("top", "bottom", base)))


def caption_zone(preset: Any, frame_size: tuple[int, int], position: str | None = None) -> Rect:
    """Полоса кадра, зарезервированная под субтитр, ещё до того, как известен текст.

    `caption_rect` меряет конкретный блок, а этот резерв нужен там, где текста
    ещё нет вообще: пресет без плана, вызов `broll_free_band` без контекста.
    Поэтому здесь худший случай, и он считается по данным пресета, а не по числам
    в коде: строк — `caption.max_lines` (иначе пресет с четырьмя строками молча
    вылезет из резерва), а КАЖДАЯ строка — по `shout`.

    Крик в каждой строке, а не в одной: формула `(строк−1)·базовая + самая
    высокая` описывала блок, где выделено ровно одно слово, а контракт разрешает
    `shout` в каждой строке — и такой блок у glass 1080×1920 выше резерва на
    63.4 px, то есть картинка вставала внутрь текста. Это стоило DoD Ф7 на
    производственном плане, а тест резерва не видел промаха, потому что клал
    ровно одно выделенное слово.

    Чего резерв НЕ покрывает: блок, который сознательно вылез за `max_lines`.
    `wrap_block` упирается в пол кегля (`caption.min_word_scale`) и отдаёт больше
    строк, чем разрешено, — шесть вместо трёх на шести длинных словах, и это 631
    px против 431. Предсказать число строк без текста нельзя, поэтому сборка
    плана разводит картинку не с этим резервом, а с ИЗМЕРЕННЫМИ прямоугольниками
    блоков (`render_plan.build_plan`, параметр `zones`). Резерв остаётся тем, чем
    он честно является: оценкой для случая, когда текста ещё нет.

    Полоса во всю ширину кадра намеренно. Панель центрирована и в кадре 1080
    доходит до x 119…961, а угловая карточка занимает 54…529 — по горизонтали
    они пересекаются при любом якоре, разводить их можно только по вертикали.
    """
    width, height = frame_size
    lines = max(1, int(getattr(preset.caption, "max_lines", 3) or 3))
    # Потолок роста (С1) — теперь он и есть худший случай: блок поднимается,
    # пока не упрётся либо в бюджет ширины, либо в этот потолок, и упереться в
    # потолок он может в КАЖДОЙ строке (три коротких слова, каждое на своей).
    # Забыть его значило бы вернуть ровно ту ошибку, из-за которой резерв уже
    # промахивался: картинка вставала внутрь текста.
    grow = float(getattr(preset.caption, "fill_max_scale", 1.0) or 1.0)
    biggest = (
        max(
            emphasis_size("accent", preset),
            emphasis_size("shout", preset),
            preset.font.size * max(1.0, grow),
        )
        * preset.caption.line_height
    )
    panel = preset.panel
    chrome_y = 2 * (panel.padding_y + panel.border_width) if panel.enabled else 0.0
    h = lines * biggest + (lines - 1) * line_gap(preset) + chrome_y

    pos = position or preset.caption.position
    margin = safe_margin_y(preset, height)
    if pos == "top":
        y = float(margin)
    elif pos == "center":
        y = (height - h) / 2
    else:
        y = height - margin - h
    return Rect(0.0, y, float(width), h)


def blocks_caption(
    rect: Rect, preset: Any, frame_size: tuple[int, int], position: str | None = None
) -> bool:
    """Мешает ли картинка субтитру в этой позиции.

    Настоящее пересечение прямоугольников вместо «той же половины кадра»:
    половинами `center` и `fullscreen` описывались как «занято всё», из-за чего
    уклонение субтитра выключалось именно там, где было нужнее всего
    (AUDIT §8.3, логическая инверсия в `occupied()`).
    """
    return rect.overlaps(caption_zone(preset, frame_size, position))


def free_band(
    preset: Any, frame_size: tuple[int, int], zones: Iterable[Rect]
) -> tuple[float, float]:
    """Кадр минус безопасная зона минус занятые полосы минус зазор.

    Из оставшихся кусков берётся самый высокий. Безопасная зона считается от
    `geometry.safe_area_ratio`, а не от `safe_margin_y`: в последнем сидит
    `caption.margin` (380 px — высота интерфейса ленты под субтитром), к картинке
    он отношения не имеет.

    `zones` — прямоугольники, которые может занять текст. Прямоугольники, а не
    названия позиций: сборка плана меряет блоки НАСТОЯЩИМИ и передаёт их сюда,
    и только так гарантия покрывает блок, который вылез за `caption.max_lines`.
    Из прямоугольника берётся вертикаль: по горизонтали панель и карточка
    пересекаются при любом якоре (см. `caption_zone`).
    """
    width, height = frame_size
    safe = height * preset.geometry.safe_area_ratio
    gap = width * preset.geometry.broll_margin_ratio
    pieces: list[tuple[float, float]] = [(safe, height - safe)]
    for zone in zones:
        lo, hi = zone.y - gap, zone.bottom + gap
        rest: list[tuple[float, float]] = []
        for a, b in pieces:
            if hi <= a or lo >= b:
                rest.append((a, b))
                continue
            if lo > a:
                rest.append((a, lo))
            if hi < b:
                rest.append((hi, b))
        pieces = rest
    if not pieces:
        return (safe, safe)
    return max(pieces, key=lambda p: p[1] - p[0])


def _band_without(
    preset: Any, frame_size: tuple[int, int], positions: Iterable[str]
) -> tuple[float, float]:
    """Полоса, свободная от РЕЗЕРВОВ субтитра в перечисленных позициях."""
    return free_band(
        preset, frame_size, [caption_zone(preset, frame_size, p) for p in positions]
    )


def broll_free_band(
    preset: Any,
    frame_size: tuple[int, int],
    positions: Iterable[str] | None = None,
    *,
    zones: Iterable[Rect] | None = None,
) -> tuple[float, float]:
    """Полоса по вертикали (y_top, y_bottom), где картинке можно стоять.

    Резервируется НЕ ТА полоса, которую займёт субтитр, а все, которые он может
    занять: позицию блока выбирает не картинка.

    Что именно уступает картинка, знает вызывающий, и знает по-разному:

    * `zones` — измеренные прямоугольники блоков. Так делает сборка плана: она
      раскладывает субтитры ДО картинок и знает про них всё, включая блок,
      вылезший за `caption.max_lines`. Это единственный путь, на котором DoD Ф7
      «картинка ни разу не пересекает субтитр» — гарантия, а не оценка.
    * `positions` — названия позиций, если текста ещё нет: берутся резервы
      `caption_zone`.
    * ни того, ни другого — строгий набор `CAPTION_POSITIONS`, то есть всё, что
      контракт вообще разрешает блоку.

    Ступени «а если строго не влезает, уступим только позиции пресета» больше
    нет. Она давала полосу, которая у glass совпадала с полосой ЦЕНТРАЛЬНОГО
    субтитра: гарантия молча отключалась ровно там, где режиссёр ставил блоку
    явный `center`. Лучше честный полный кадр: пустая полоса (a, a) означает
    «места нет», и `broll_rect` отдаёт в этом случае полный кадр, а не ставит
    карточку поверх текста.
    """
    if zones is not None:
        return free_band(preset, frame_size, tuple(zones))
    if positions is not None:
        return _band_without(preset, frame_size, tuple(positions))
    strict = _band_without(preset, frame_size, CAPTION_POSITIONS)
    if strict[1] - strict[0] >= frame_size[1] * min_band_ratio(preset):
        return strict
    return (strict[0], strict[0])


def resting_anchor(anchor: str, anim: str) -> str:
    """Где картинка окажется в покое.

    `fullscreen_to_inset` стартует с полного кадра, но живёт в углу, и разводить
    с субтитром надо именно угол. Якорь `fullscreen` для этой анимации не имеет
    смысла — уезжать некуда, поэтому он трактуется как `top_right` (так же
    делает Broll.tsx).
    """
    if anim == "fullscreen_to_inset":
        return "top_right" if anchor == "fullscreen" else anchor
    return anchor


def broll_rect(
    anchor: str,
    preset: Any,
    frame_size: tuple[int, int],
    *,
    aspect: float | None = None,
    positions: Iterable[str] | None = None,
    zones: Iterable[Rect] | None = None,
) -> Rect:
    """Прямоугольник вставной картинки.

    `aspect` — отношение ширины к высоте ИСТОЧНИКА. Карточка наследует его, а не
    остаётся квадратной: при квадрате `object-fit: cover` срезал у картинки 2:3
    33 % высоты (AUDIT §8.3). Ширина карточки от пропорции не зависит, меняется
    высота. `None` — источник не измерен, берём `FALLBACK_ASPECT`.

    Якорь задаёт положение, но не имеет права поставить карточку туда, где будет
    текст: по вертикали прямоугольник зажимается в свободную полосу
    (`broll_free_band`). Это и есть DoD Ф7 «картинка ни разу не пересекает
    caption_rect» — гарантия, а не удачно подобранные доли в пресете.

    `zones` — измеренные прямоугольники блоков; так вызывает сборка плана, и
    только на этом пути гарантия безусловна (резерв `caption_zone` не знает про
    блок, вылезший за `caption.max_lines`). `positions` — позиции, если текста
    ещё нет. Ни того, ни другого — уступается всё, что разрешает контракт.

    Свободной полосы может не остаться совсем (плотный пресет; субтитр по центру,
    как в `bold`). Тогда карточки не будет: возвращается полный кадр. Молча
    выдать карточку полного размера прямо в полосу субтитра — худшее из
    возможных: гарантия отключалась бы ровно там, где она нужна.
    """
    width, height = frame_size
    geometry = preset.geometry
    if anchor == "fullscreen":
        return Rect(0.0, 0.0, float(width), float(height))

    aspect = aspect if aspect and aspect > 0 else FALLBACK_ASPECT
    band_top, band_bottom = broll_free_band(preset, frame_size, positions, zones=zones)
    band_h = max(0.0, band_bottom - band_top)
    if band_h < height * min_band_ratio(preset):
        log.info(
            "пресет %r: свободной полосы под карточку нет (%.0f px) — вставка полноэкранная",
            getattr(preset, "name", "?"), band_h,
        )
        return Rect(0.0, 0.0, float(width), float(height))

    # Карточка поверх размытого кадра — та же середина свободной полосы, что и
    # `center`, только крупная: лицо под ней размыто, уступать ему нечего.
    ratio = geometry.broll_card_ratio if anchor == "card" else geometry.broll_inset_ratio
    w = width * ratio
    h = w / aspect
    if h > band_h:
        # Не влезает в свободную полосу — уменьшаем карточку целиком, пропорция
        # важнее заданной пресетом ширины: растянуть её значит снова резать.
        h = band_h
        w = h * aspect
    margin = width * geometry.broll_margin_ratio

    if anchor in ("center", "card"):
        return Rect((width - w) / 2, band_top + (band_h - h) / 2, w, h)

    top = anchor.startswith("top")
    left = anchor.endswith("left")
    y = height * (geometry.broll_top_ratio if top else geometry.broll_bottom_ratio)
    y = min(max(y, band_top), max(band_top, band_bottom - h))
    x = margin if left else width - w - margin
    return Rect(x, y, w, h)
