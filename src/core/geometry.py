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

Чего модуль сознательно не умеет: эмодзи и любой символ вне репертуара шрифта
рисуются фолбэком системы, их ширину этот TTF не знает. Для них берётся
`FALLBACK_ADVANCE_EM` — оценка, а не измерение (см. `FontMetrics.advance_em`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

log = logging.getLogger(__name__)

METRICS_DIR = Path(__file__).resolve().parent / "presets" / "metrics"

# Ширина символа, которого нет в шрифте. Эмодзи почти во всех цветных шрифтах
# квадратные и занимают ровно em, поэтому единица — честная оценка сверху для
# них и грубая для всего остального. Точно измерить чужой шрифт мы не можем:
# какой фолбэк подставит Chromium, зависит от системы, а не от пресета.
FALLBACK_ADVANCE_EM = 1.0


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
        """Текст → последовательность «единиц»: символ или лигатура.

        Лигатуры отключаются, когда задан letter-spacing: Chromium в этом случае
        не склеивает `fi`, иначе спейсинг было бы некуда вставить.
        """
        if not ligatures or not self.ligatures:
            return list(text)
        out: list[str] = []
        i = 0
        while i < len(text):
            for lig in self.ligatures:  # отсортированы по убыванию длины
                if text.startswith(lig, i):
                    out.append(lig)
                    i += len(lig)
                    break
            else:
                out.append(text[i])
                i += 1
        return out

    def advance(self, unit: str) -> int:
        """Ширина единицы в единицах шрифта."""
        found = self.advances.get(unit)
        if found is not None:
            return found
        return round(FALLBACK_ADVANCE_EM * self.units_per_em)

    def kern(self, left: str, right: str) -> int:
        """Кернинг пары в единицах шрифта (обычно отрицательный)."""
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
        total += sum(self.kern(a, b) for a, b in zip(units, units[1:]))
        return total / self.units_per_em

    def units_count(self, text: str, *, ligatures: bool = True) -> int:
        return len(self.split(text, ligatures=ligatures))


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
       множитель — число символов, а не число промежутков между ними.
    2. Ненулевой спейсинг выключает лигатуры: `affine` меряется шестью глифами,
       а не пятью (проверено на 0.1 em, где разница не тонет в округлении).

    `spacing_size` — кегль, ОТ КОТОРОГО считаются em спейсинга. По умолчанию это
    собственный кегль слова, но в субтитре `letter-spacing` объявлен на общем
    контейнере строки и наследуется уже в пикселях: акцентное слово крупнее, а
    спейсинг у него тот же, что у соседей. Считать em от кегля акцента —
    ошибка в 1.6 px на слове из семи букв.
    """
    if not text:
        return 0.0
    if uppercase:
        text = text.upper()
    use_ligatures = letter_spacing == 0.0
    width = metrics.advance_em(text, ligatures=use_ligatures) * font_size
    if letter_spacing:
        spacing_px = letter_spacing * (font_size if spacing_size is None else spacing_size)
        width += spacing_px * metrics.units_count(text, ligatures=use_ligatures)
    return width


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
    """
    items = [w if isinstance(w, SizedWord) else SizedWord(w[0], w[1]) for w in words]
    items = [w for w in items if w.text]
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
    space = measure_word(" ", base_size, metrics=metrics, letter_spacing=letter_spacing)
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
    """Слова плана → слова с кеглями пресета."""
    return [SizedWord(w.text, emphasis_size(getattr(w, "emphasis", "none"), preset)) for w in words]


# --------------------------------------------------------------------------- субтитр


def safe_margin_x(preset: Any, frame_width: int) -> float:
    return frame_width * preset.geometry.safe_area_ratio


def safe_margin_y(preset: Any, frame_height: int) -> float:
    """Отступ субтитра от края: пресет может попросить меньше, безопасная зона — нет.

    Прижатый к краю текст обрезает интерфейс ленты, этого не хочет ни один
    пресет, поэтому нижняя граница жёсткая (так же считает Captions.tsx).
    """
    return max(preset.caption.margin, round(frame_height * preset.geometry.safe_area_ratio))


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
    return round(preset.font.size * (preset.caption.line_height - 1))


def _line_height(line: Sequence[SizedWord], preset: Any) -> float:
    """Высота строчного бокса: `line-height` пресета от самого крупного слова.

    `line-height` задан числом, значит наследуется множителем и считается от
    собственного кегля элемента. Строчный бокс поднимает самое высокое слово —
    акцентное.
    """
    biggest = max((w.size for w in line), default=preset.font.size)
    return biggest * preset.caption.line_height


class _HasLines(Protocol):
    position: str
    lines: list[list[Any]]


def caption_rect(block: _HasLines, preset: Any, frame_size: tuple[int, int]) -> Rect:
    """Прямоугольник, который займёт блок субтитра, — вместе с панелью.

    Два условия на входе. Пресет уже приведён к высоте кадра (`Preset.scaled`):
    кегли и паддинги берутся как есть. Текст слов уже в экранной форме
    (`display_text`): регистр и пунктуация решены до нас, второй раз их трогать
    нельзя, иначе ширина разойдётся с тем, что нарисует рендерер.

    Ширина текста ограничена бюджетом `caption_text_width`. Это не подгонка, а
    контракт: в TSX на панели стоит `max-width`, поэтому шире она не станет в
    любом случае. Уложить слова в бюджет — задача переноса строк (Ф5).
    """
    width, height = frame_size
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
        text_h += _line_height(line, preset)
        if i:
            text_h += line_gap(preset)
    text_w = min(text_w, caption_text_width(preset, width))

    panel = preset.panel
    chrome_x = 2 * (panel.padding_x + panel.border_width) if panel.enabled else 0.0
    chrome_y = 2 * (panel.padding_y + panel.border_width) if panel.enabled else 0.0
    box_w = text_w + chrome_x
    box_h = text_h + chrome_y

    margin = safe_margin_y(preset, height)
    if block.position == "top":
        y = float(margin)
    elif block.position == "center":
        y = (height - box_h) / 2
    else:
        y = height - margin - box_h

    # Блок центрируется по кадру всегда: `caption.align` управляет выключкой
    # текста внутри панели, а не положением самой панели.
    return Rect((width - box_w) / 2, y, box_w, box_h)


# --------------------------------------------------------------------------- картинки

# Пропорция (ширина/высота) стоковой картинки. Pexels отдаёт нарезку `portrait`
# 800×1200, то есть 2:3 (AUDIT §8.3). Квадратная карточка при таком источнике
# срезала `object-fit: cover`-ом треть высоты — по 200 px исходника сверху и
# снизу, а сток кладёт объект как раз в верхнюю треть. Карточка, наследующая
# пропорцию источника, не режет ничего.
STOCK_ASPECT = 2 / 3

# Позиции, которые контракт разрешает блоку субтитра (`CaptionBlock.position`
# минус `auto`, который разрешается в одну из них). Резерв обязан считаться по
# худшему случаю: занизить его — значит поставить картинку туда, куда потом
# придёт текст.
CAPTION_POSITIONS = ("top", "center", "bottom")

# Ниже этой доли высоты кадра карточка перестаёт быть иллюстрацией: при 2:3 это
# 154×230 px в кадре 1080×1920 — марка, которую не рассмотреть. Если свободной
# полосы осталось меньше, вставка отдаётся полным кадром: осознанный полный кадр
# честнее карточки, поставленной поверх субтитра.
MIN_BAND_RATIO = 0.12


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
    return tuple(dict.fromkeys(("top", "bottom", base)))


def caption_zone(preset: Any, frame_size: tuple[int, int], position: str | None = None) -> Rect:
    """Полоса кадра, зарезервированная под субтитр, ещё до того, как известен текст.

    `caption_rect` меряет конкретный блок, а картинку надо разводить с блоком,
    которого в этот момент ещё нет: b-roll считается раньше субтитров (блокам
    нужно знать, какая часть кадра занята). Поэтому здесь худший случай, и он
    считается по данным пресета, а не по числам в коде: строк —
    `caption.max_lines` (иначе пресет с четырьмя строками молча вылезет из
    резерва), а самая высокая строка — по `shout`, а не по `accent`. Крик крупнее
    акцента (1.34 против 1.18 в glass), контракт разрешает его в любом блоке, и
    трёхстрочный блок с криком выше «худшего случая» по акценту на 15 px.

    Полоса во всю ширину кадра намеренно. Панель центрирована и в кадре 1080
    доходит до x 119…961, а угловая карточка занимает 54…529 — по горизонтали
    они пересекаются при любом якоре, разводить их можно только по вертикали.
    """
    width, height = frame_size
    lines = max(1, int(getattr(preset.caption, "max_lines", 3) or 3))
    base = float(preset.font.size) * preset.caption.line_height
    biggest = (
        max(emphasis_size("accent", preset), emphasis_size("shout", preset))
        * preset.caption.line_height
    )
    panel = preset.panel
    chrome_y = 2 * (panel.padding_y + panel.border_width) if panel.enabled else 0.0
    h = (lines - 1) * base + biggest + (lines - 1) * line_gap(preset) + chrome_y

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


def _band_without(
    preset: Any, frame_size: tuple[int, int], positions: Iterable[str]
) -> tuple[float, float]:
    """Кадр минус безопасная зона минус перечисленные полосы субтитра минус зазор.

    Из оставшихся кусков берётся самый высокий. Безопасная зона считается от
    `geometry.safe_area_ratio`, а не от `safe_margin_y`: в последнем сидит
    `caption.margin` (380 px — высота интерфейса ленты под субтитром), к картинке
    он отношения не имеет.
    """
    width, height = frame_size
    safe = height * preset.geometry.safe_area_ratio
    gap = width * preset.geometry.broll_margin_ratio
    pieces: list[tuple[float, float]] = [(safe, height - safe)]
    for position in positions:
        zone = caption_zone(preset, frame_size, position)
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


def broll_free_band(
    preset: Any, frame_size: tuple[int, int], positions: Iterable[str] | None = None
) -> tuple[float, float]:
    """Полоса по вертикали (y_top, y_bottom), где картинке можно стоять.

    Резервируется НЕ ТА полоса, которую займёт субтитр, а все, которые он может
    занять: позицию блока выбирает не картинка. Картинка считается раньше
    субтитров и поэтому уступает первой.

    `positions` — набор позиций, которые блок реально может получить. Знает его
    только сборка плана (явные позиции блоков видны в EDL); отсюда он не виден,
    поэтому по умолчанию берётся лестница из двух ступеней:

    1. Все три позиции контракта (`CAPTION_POSITIONS`). Тогда гарантия
       безусловная: карточка не пересечёт субтитр даже при явном `center`.
    2. Если после этого не осталось полосы разумной высоты (`MIN_BAND_RATIO`) —
       только те позиции, которые пресет даёт сам (`caption_positions`). Три
       трёхстрочные полосы, две безопасные зоны и шесть зазоров занимают в кадре
       1080×1920 больше, чем в нём есть: у glass остаётся 192 px, у bold — 184.

    Пустая полоса (a, a) означает «места нет»; `broll_rect` отдаёт в этом случае
    полный кадр, а не ставит карточку поверх текста.
    """
    if positions is not None:
        return _band_without(preset, frame_size, tuple(positions))

    floor = frame_size[1] * MIN_BAND_RATIO
    strict = _band_without(preset, frame_size, CAPTION_POSITIONS)
    if strict[1] - strict[0] >= floor:
        return strict
    relaxed = _band_without(preset, frame_size, caption_positions(preset))
    if relaxed[1] - relaxed[0] >= floor:
        return relaxed
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
) -> Rect:
    """Прямоугольник вставной картинки.

    `aspect` — отношение ширины к высоте ИСТОЧНИКА. Карточка наследует его, а не
    остаётся квадратной: при квадрате `object-fit: cover` срезал у стоковой
    картинки 33 % высоты (AUDIT §8.3). Ширина карточки от пропорции не зависит,
    меняется высота. `None` — источник ещё не измерен, берём стоковые 2:3.

    Якорь задаёт положение, но не имеет права поставить карточку туда, где будет
    текст: по вертикали прямоугольник зажимается в свободную полосу
    (`broll_free_band`). Это и есть DoD Ф7 «картинка ни разу не пересекает
    caption_rect» — гарантия, а не удачно подобранные доли в пресете. Гарантия
    покрывает те позиции субтитра, которые вернул `broll_free_band` (см. его
    лестницу): все три, когда в кадре есть место, и позиции пресета, когда нет.

    `positions` — позиции, которые блоки реально получат. Их видит только сборка
    плана (явные позиции лежат в EDL); передать их сюда значит сделать гарантию
    безусловной и там, где режиссёр поставил блоку явный `center`.

    Свободной полосы может не остаться совсем (плотный пресет; субтитр по центру,
    как в `bold`). Тогда карточки не будет: возвращается полный кадр. Молча
    выдать карточку полного размера прямо в полосу субтитра — худшее из
    возможных: гарантия отключалась бы ровно там, где она нужна.
    """
    width, height = frame_size
    geometry = preset.geometry
    if anchor == "fullscreen":
        return Rect(0.0, 0.0, float(width), float(height))

    aspect = aspect if aspect and aspect > 0 else STOCK_ASPECT
    band_top, band_bottom = broll_free_band(preset, frame_size, positions)
    band_h = max(0.0, band_bottom - band_top)
    if band_h < height * MIN_BAND_RATIO:
        log.info(
            "пресет %r: свободной полосы под карточку нет (%.0f px) — вставка полноэкранная",
            getattr(preset, "name", "?"), band_h,
        )
        return Rect(0.0, 0.0, float(width), float(height))

    w = width * geometry.broll_inset_ratio
    h = w / aspect
    if h > band_h:
        # Не влезает в свободную полосу — уменьшаем карточку целиком, пропорция
        # важнее заданной пресетом ширины: растянуть её значит снова резать.
        h = band_h
        w = h * aspect
    margin = width * geometry.broll_margin_ratio

    if anchor == "center":
        return Rect((width - w) / 2, band_top + (band_h - h) / 2, w, h)

    top = anchor.startswith("top")
    left = anchor.endswith("left")
    y = height * (geometry.broll_top_ratio if top else geometry.broll_bottom_ratio)
    y = min(max(y, band_top), max(band_top, band_bottom - h))
    x = margin if left else width - w - margin
    return Rect(x, y, w, h)
