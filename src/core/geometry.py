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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

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
    anchor: str, preset: Any, frame_size: tuple[int, int], *, aspect: float = 1.0
) -> Rect:
    """Прямоугольник вставной картинки.

    `aspect` — отношение ширины к высоте карточки. Единица (квадрат) осталась от
    старого TSX; Ф7 передаст сюда пропорцию источника, чтобы `cover` не срезал
    половину картинки. Ширина карточки от пропорции не зависит.
    """
    width, height = frame_size
    geometry = preset.geometry
    if anchor == "fullscreen":
        return Rect(0.0, 0.0, float(width), float(height))

    w = width * geometry.broll_inset_ratio
    h = w / aspect if aspect > 0 else w
    margin = width * geometry.broll_margin_ratio

    if anchor == "center":
        return Rect((width - w) / 2, (height - h) / 2, w, h)

    top = anchor.startswith("top")
    left = anchor.endswith("left")
    y = height * (geometry.broll_top_ratio if top else geometry.broll_bottom_ratio)
    x = margin if left else width - w - margin
    return Rect(x, y, w, h)
