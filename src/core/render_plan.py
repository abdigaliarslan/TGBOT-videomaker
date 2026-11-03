"""EDL → план рендера. Единственное место, где считается время и геометрия.

Рендерер (хоть ASS, хоть Remotion) не принимает решений и не считает арифметику:
он получает готовые секунды, готовые строки, готовые прямоугольники и готовый
пресет и рисует буквально (§0). Поэтому обе реализации рендера едят одну и ту же
структуру.

Прямоугольники (`PlanBlock.rect`, `PlanBroll.rect`) появились в Ф4: до них
геометрия жила константами в TSX, а Python оперировал половинами кадра и не мог
ни предсказать положение строки, ни развести субтитр с картинкой (AUDIT §8.4).
Считает их `src/core/geometry.py` в пикселях кадра.

В Ф5 сюда переехала и раскладка строки. Перенос считается в пикселях
(`wrap_block`), а не в символах, и каждому слову проставляется собственный
кегль: акцент — это РАЗМЕР, а не `transform: scale`, иначе увеличенное слово
не участвует в раскладке и накрывает соседа (AUDIT §8.1). Рендерер получает
готовые строки и готовые кегли и не выводит из выделения ничего.

Регистров субтитра ДВА (STYLE_PLAN §1.4.1): «плакат» и «тихий», и отличаются они
не оттенком, а кеглем вчетверо, отсутствием золота, выключкой влево у левого края
и четырьмя строками вместо двух. Регистр ОБЪЯВЛЯЕТ режиссёр
(`CaptionBlock.quiet`) — рендерер его не выводит (§0). Выводится он только у
листов, сохранённых до появления поля (`quiet=None`): там его по-прежнему решает
позиция (`caption.quiet.position`).

Регистр задаёт всю раскладку, а у блока с `position="auto"` и необъявленным
регистром раскладка поэтому зависит от того, какую позицию блок займёт. Значит
мерить его приходится под каждую возможную позицию (`_Place`, `lay_places`), а не
один раз на все.

Третий регистр — ТИТР первых секунд (STYLE_PLAN §5, `title_card_block`). Он тоже
плакат, только наверху и своим кеглем, и от двух других отличается не раскладкой,
а происхождением: его текст приезжает заголовком листа (`EDL.title`), а не
словами транскрипта, поэтому блока для него в EDL нет и собирается он здесь.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Sequence

from src.core.anchor import fallbacks, follow_offsets
from src.core.config import Settings, settings as default_settings
from src.core.edl import EDL, Emphasis, Timeline
from src.core.geometry import (
    CAPTION_POSITIONS,
    FontMetrics,
    Rect,
    block_rect,
    blocks_caption,
    broll_rect,
    caption_positions,
    caption_text_width,
    caption_zone,
    emphasis_size,
    line_box_height,
    line_gap,
    measure_word,
    metrics_for,
    resting_anchor,
    space_width,
)
from src.core.preset import Preset, TitleCard, load_preset
from src.core.text import display_text
from src.core.track import FaceTrack

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PlanWord:
    text: str
    start: float  # секунды на ВЫХОДНОМ таймлайне
    end: float
    emphasis: str = "none"
    emoji: str | None = None
    word_idx: int = -1
    # Кегль слова в пикселях кадра. Акцент — это РАЗМЕР, а не `transform: scale`:
    # масштаб не участвует в раскладке, поэтому увеличенное слово накрывало
    # соседа (AUDIT §8.1). Здесь размер считает Python: он же им и мерил строку,
    # и рендерер обязан поставить ровно его. `None` — метрик шрифта не было,
    # рендерер берёт кегль из пресета сам, как до Ф5.
    size: float | None = None


@dataclass(slots=True)
class PlanBlock:
    start: float
    end: float
    position: Literal["top", "center", "bottom"]
    lines: list[list[PlanWord]]
    # Куда блок попадёт в кадре, вместе с панелью. `None` — метрик шрифта нет,
    # рендерер раскладывает по-старому.
    rect: Rect | None = None
    # Средняя яркость кадра под блоком, 0..1. Меряет препроцесс; рендерер по ней
    # ведёт прозрачность панели и цвет текста, чтобы контраст взял порог (Ф9).
    # `None` — никто не мерил, пресет играет как играл.
    bg_luma: float | None = None
    # Носит ли ЭТОТ блок матовое стекло. Решение приезжает из EDL, потому что
    # оно режиссёрское, а не вычислимое: см. `CaptionBlock.panel`. Место под
    # плашку (паддинги и рамка) зарезервировано у ВСЕХ блоков одинаково, иначе
    # текст прыгал бы на её высоту, а Python и Chromium разошлись бы в замере.
    panel: bool = False
    # Путь блока, привязанного к голове (М2): смещение от прямоугольника покоя в
    # пикселях кадра, по кадрам от `start`. Считает Python — вместе с
    # предохранителями и плавным откатом, потому что рендерер решений не
    # принимает (§0). `None` — блок статичен, то есть поведение до М2.
    anchor_path: list[list[float]] | None = None
    # В каком РЕГИСТРЕ нарисован блок: `True` — тихий (STYLE_PLAN §1.4.1),
    # `False` — плакат. Регистров два, и это не оттенок одного стиля: кегль
    # отличается вчетверо, золота у тихого нет вовсе, выключка влево у левого
    # края, строк до четырёх.
    #
    # Решение приезжает готовым, как `position` и `panel`: регистр объявляет
    # режиссёр (`CaptionBlock.quiet`), а рендереру выводить его заново
    # запрещено (§0). Он только подменяет числа темы на числа регистра
    # (`caption.quiet`), которые уже лежат в плане. `False` у всего плана — тема
    # без второго регистра (glass, bold) или план, собранный до его появления.
    quiet: bool = False
    # ТИТР ролика (STYLE_PLAN §5) — третий регистр. Блок, которого нет в речи:
    # текст ему даёт заголовок (`EDL.title`), а не слова транскрипта, и собирает
    # его `build_plan`, а не режиссёр.
    #
    # Флаг, а не вывод по позиции или по кеглю: числа регистра лежат в
    # `preset.caption.title_card`, и рендерер обязан подменить ими числа темы —
    # отступ от верха у титра свой (332 px против 380 у плаката) и межстрочие
    # своё, иначе полоса титра уедет мимо замера. Регистры взаимоисключающи:
    # титр — это плакат наверху, и `quiet` у него всегда `False`.
    title_card: bool = False

    @property
    def words(self) -> list[PlanWord]:
        return [w for line in self.lines for w in line]


@dataclass(slots=True)
class PlanBroll:
    start: float
    end: float
    anim: str
    anchor: str
    asset: str | None
    query: str
    # Место картинки В ПОКОЕ. `fullscreen_to_inset` стартует с полного кадра,
    # но живёт здесь, и разводить с субтитром надо именно это место.
    rect: Rect | None = None
    # Направление медленного наезда, −1 или 1 (М3). Приезжает из EDL, чтобы
    # перерендер того же листа дал ровно то же видео.
    kb_dir: int = 1
    # Пропорция файла (ширина/высота). Геометрии она нужна была и раньше — по
    # ней считается `rect`, — но у полнокадровой вставки `rect` всегда кадр
    # целиком, и по нему уже не понять, что там за источник. Рендереру это
    # решает один вопрос: класть ли под картинку размытую копию. Картинка чужой
    # пропорции без неё обрезается, нарисованная вертикаль 9:16 с ней — только
    # портится.
    aspect: float | None = None


@dataclass(slots=True)
class RenderPlan:
    version: int
    fps: int
    width: int
    height: int
    duration: float
    video: str
    preset: dict[str, Any]
    blocks: list[PlanBlock] = field(default_factory=list)
    broll: list[PlanBroll] = field(default_factory=list)
    audio: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RenderPlan":
        return cls(
            version=data.get("version", 1),
            fps=data["fps"],
            width=data["width"],
            height=data["height"],
            duration=data["duration"],
            video=data["video"],
            audio=data.get("audio"),
            preset=data["preset"],
            blocks=[
                PlanBlock(
                    start=b["start"],
                    end=b["end"],
                    position=b["position"],
                    lines=[[PlanWord(**w) for w in line] for line in b["lines"]],
                    rect=_rect_from(b.get("rect")),
                    bg_luma=b.get("bg_luma"),
                    panel=bool(b.get("panel", False)),
                    anchor_path=b.get("anchor_path"),
                    quiet=bool(b.get("quiet", False)),
                    title_card=bool(b.get("title_card", False)),
                )
                for b in data.get("blocks", [])
            ],
            broll=[
                PlanBroll(
                    start=x["start"],
                    end=x["end"],
                    anim=x["anim"],
                    anchor=x["anchor"],
                    asset=x.get("asset"),
                    query=x.get("query", ""),
                    rect=_rect_from(x.get("rect")),
                    kb_dir=int(x.get("kb_dir", 1) or 1),
                    aspect=(float(x["aspect"]) if x.get("aspect") else None),
                )
                for x in data.get("broll", [])
            ],
            meta=data.get("meta", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration": round(self.duration, 3),
            "video": self.video,
            "audio": self.audio,
            "preset": self.preset,
            "blocks": [
                {
                    "start": round(b.start, 3),
                    "end": round(b.end, 3),
                    "position": b.position,
                    "lines": [[asdict(w) for w in line] for line in b.lines],
                    "rect": b.rect.to_dict() if b.rect else None,
                    "bg_luma": b.bg_luma,
                    "panel": b.panel,
                    "anchor_path": b.anchor_path,
                    "quiet": b.quiet,
                    "title_card": b.title_card,
                }
                for b in self.blocks
            ],
            "broll": [
                {
                    "start": round(x.start, 3),
                    "end": round(x.end, 3),
                    "anim": x.anim,
                    "anchor": x.anchor,
                    "asset": x.asset,
                    "query": x.query,
                    "rect": x.rect.to_dict() if x.rect else None,
                    "kb_dir": x.kb_dir,
                    "aspect": (round(x.aspect, 4) if x.aspect else None),
                }
                for x in self.broll
            ],
            "meta": self.meta,
        }


def _rect_from(data: dict[str, Any] | None) -> Rect | None:
    """План без прямоугольников (старый или от чужого инструмента) читается как есть."""
    return Rect.from_dict(data) if data else None


# --------------------------------------------------------------------------- перенос строк


def wrap_words(words: list[PlanWord], max_chars: int, max_lines: int = 3) -> list[list[PlanWord]]:
    """Жадный перенос по символам — запасной путь, когда мерить нечем.

    Символ не единица ширины: `Ш` шире `і` вдвое, эмодзи не считается вовсе,
    пробел в бюджете стоит один символ, а рисуется другой ширины (AUDIT §8.1).
    Поэтому основной путь — `wrap_block` в пикселях, а эта функция остаётся на
    случай пресета без метрик шрифта, где ширину взять негде.
    """
    if not words:
        return []
    lines: list[list[PlanWord]] = [[]]
    length = 0
    for w in words:
        wl = len(w.text)
        if lines[-1] and length + 1 + wl > max_chars and len(lines) < max_lines:
            lines.append([])
            length = 0
        lines[-1].append(w)
        length += wl + (1 if length else 0)
    return [line for line in lines if line]


# Средняя ширина буквы в em на запасном пути — там, где метрик шрифта нет и
# ширину строки посчитать физически нечем. Числа замерены по текстам настоящего
# прогона (`samples/plan-baseline.json`, Montserrat ExtraBold): строчная
# кириллица 0.654 em, прописная 0.761 em. Одной константой на все пресеты
# обойтись нельзя: bold рисует UPPERCASE кеглем 104, и «15 символов», выведенные
# под glass, дают у него строку в 1217 px при бюджете 842 (AUDIT §8.1 замерил
# ровно это: 11.9 символа в строку).
FALLBACK_CHAR_EM = 0.654
FALLBACK_CHAR_EM_UPPER = 0.761

def fallback_line_chars(preset: Preset, frame_width: int) -> int:
    """Сколько символов класть в строку, когда мерить нечем.

    Считается от кегля и регистра ПРЕСЕТА, а не константой: см. `FALLBACK_CHAR_EM`.
    Спейсинг входит в ширину символа (он отрицательный и заметный: −0.02 em в
    bold — это 2 px на букву).
    """
    em = FALLBACK_CHAR_EM_UPPER if preset.font.uppercase else FALLBACK_CHAR_EM
    per_char = (em + preset.font.letter_spacing) * float(preset.font.size)
    if per_char <= 0:
        return 15
    return max(4, int(caption_text_width(preset, frame_width) // per_char))


@dataclass(frozen=True, slots=True)
class _Ruler:
    """Линейка строки: те же числа, которыми Chromium нарисует субтитр.

    Пробел между словами — настоящий пробел кеглем строки, а не flex-`gap`:
    `gap` не масштабируется вместе с акцентом и врёт тем сильнее, чем крупнее
    выделенное слово. `render/measure.mjs` меряет ровно эту модель DOM.
    """

    metrics: FontMetrics
    letter_spacing: float
    base_size: float
    budget: float

    def word(self, word: PlanWord, size: float) -> float:
        """Ширина слова вместе с эмодзи.

        Эмодзи лежит ВНУТРИ спана слова, за настоящим пробелом и кеглем слова
        (Captions.tsx), поэтому меряется тем же вызовом, что и текст: пробел
        считает наш шрифт, кластер эмодзи — `geometry.EMOJI_ADVANCE_EM`, и
        letter-spacing достаётся кластеру ровно один раз.
        """
        text = f"{word.text} {word.emoji}" if word.emoji else word.text
        return measure_word(
            text,
            size,
            metrics=self.metrics,
            letter_spacing=self.letter_spacing,
            # Спейсинг объявлен на контейнере строки и наследуется уже в
            # пикселях: у акцента он тот же, что у соседей.
            spacing_size=self.base_size,
        )

    @property
    def space(self) -> float:
        return space_width(
            self.base_size, metrics=self.metrics, letter_spacing=self.letter_spacing
        )

    def line_width(self, line: Sequence[PlanWord]) -> float:
        if not line:
            return 0.0
        total = sum(self.word(w, w.size or self.base_size) for w in line)
        return total + self.space * (len(line) - 1)

    def fit_size(self, word: PlanWord, size: float, floor: float) -> float:
        """Кегль, при котором слово влезает в строку в одиночку.

        Длинное слово раньше просто переполняло контейнер: перенос его не резал
        никогда. Ширина по кеглю аффинна (`a·s + b`, где `b` — спейсинг, он
        считается от кегля строки и от кегля слова не зависит), поэтому нужный
        кегль решается точно, без подбора. `floor` — пол из пресета: ниже него
        текст перестаёт быть субтитром, и переполнение честнее микроскопии.
        """
        wide = self.word(word, size)
        if wide <= self.budget:
            return size
        half = self.word(word, size / 2)
        a = (wide - half) / (size / 2)
        if a <= 0:
            return size
        b = wide - a * size
        return max(floor, min(size, (self.budget - b) / a))


def _greedy(words: Sequence[PlanWord], ruler: _Ruler) -> list[list[PlanWord]]:
    """Жадная укладка по уже проставленным кеглям. Порядок слов не меняется."""
    lines: list[list[PlanWord]] = [[]]
    width = 0.0
    for w in words:
        own = ruler.word(w, w.size or ruler.base_size)
        if lines[-1] and width + ruler.space + own > ruler.budget:
            lines.append([])
            width = 0.0
        lines[-1].append(w)
        width += own + (ruler.space if len(lines[-1]) > 1 else 0.0)
    return [line for line in lines if line]


def wrap_block(
    words: list[PlanWord], preset: Preset, frame_width: int
) -> tuple[list[list[PlanWord]], list[float]]:
    """Перенос строк в пикселях. Возвращает строки и их ширины.

    Побочный эффект намеренный: функция проставляет словам `size`. Кегль — это и
    есть раскладка (акцент крупнее соседей), поэтому мерить одним числом, а
    рисовать другим нельзя: рендерер получает ровно тот кегль, которым здесь
    посчитана ширина.

    Два способа не переполнить контейнер, в порядке применения:
    1. Слово шире всей строки — ужимаем это слово (`fit_size`).
    2. Блок не уложился в `caption.max_lines` — ужимаем блок целиком и кладём
       заново. Шаг мелкий, потому что перенос ступенчатый: одно слово,
       перескочившее на строку выше, убирает целую строку.
    """
    if not words:
        return [], []
    metrics = metrics_for(preset)
    ruler = _Ruler(
        metrics=metrics,
        letter_spacing=preset.font.letter_spacing,
        base_size=float(preset.font.size),
        budget=caption_text_width(preset, frame_width),
    )
    floor_ratio = max(0.05, min(1.0, preset.caption.min_word_scale))
    max_lines = max(1, preset.caption.max_lines)

    # Пол общий для всего блока и считается от кегля строки, а не от кегля
    # самого слова: иначе крику с его 1.42 разрешено остаться в полтора раза
    # крупнее соседей и не влезть.
    floor = ruler.base_size * floor_ratio
    grow_max = max(1.0, float(preset.caption.fill_max_scale))
    ceiling = ruler.base_size * grow_max

    roles = [emphasis_size(w.emphasis, preset) for w in words]
    biggest_role = max(roles)

    def lay(scale: float) -> list[list[PlanWord]]:
        """Проставить кегли этим масштабом и разложить. Побочный эффект намеренный."""
        for w, role in zip(words, roles):
            w.size = round(ruler.fit_size(w, role * scale, floor), 2)
        return _greedy(words, ruler)

    scale = 1.0
    lines = lay(scale)
    while len(lines) > max_lines and scale > floor_ratio:
        scale = max(floor_ratio, scale * 0.94)
        lines = lay(scale)

    # Рост (С1). Блок поднимается, пока строка не упрётся в бюджет ширины или
    # пока следующий шаг не выдавит лишнюю строку. Этим одним правилом
    # получается вся наблюдаемая у эталона картина: короткая строка набрана
    # крупнее длинной, потому что упирается она позже.
    #
    # Ищем двоичным поиском, а не шагом вверх: шаг пришлось бы делать мелким
    # (перенос ступенчатый — одно слово, перескочившее строкой выше, убирает
    # целую строку), и на блок уходило бы под сотню раскладок. Поиск
    # детерминирован, иначе поехала бы сверка с Chromium.
    #
    # Потолок ограничивает МАСШТАБ блока, а не каждое слово: общий потолок на
    # слово схлопывает иерархию — акцент и сосед упираются в одно число, после
    # чего более длинный акцент ужимается `fit_size` ПОД соседа (замерено:
    # «громче» 212.1 против «тише» 246.4, то есть выделенное слово оказалось
    # мельче обычного).
    hi = max(scale, ceiling / biggest_role) if biggest_role > 0 else scale
    if grow_max > 1.0 and hi > scale and len(lines) <= max_lines:
        lo = scale
        for _ in range(18):
            mid = (lo + hi) / 2
            if len(lay(mid)) <= max_lines:
                lo = mid
            else:
                hi = mid
        scale = lo
        lines = lay(scale)

    return lines, [ruler.line_width(line) for line in lines]


def _rect_of_lines(
    position: str,
    lines: list[list[PlanWord]],
    widths: list[float],
    preset: Preset,
    frame: tuple[int, int],
    *,
    left: float | None = None,
) -> Rect:
    """Прямоугольник блока по уже измеренным строкам.

    Отличается от `geometry.caption_rect` одним: ширина берётся из переноса, а
    не пересчитывается по выделению слов. Пересчёт не знает про длинное слово,
    ужатое `fit_size`, и поэтому даёт другую панель. Коробка, привязка к краю
    кадра и высота строки считаются общим кодом (`block_rect`,
    `line_box_height`), чтобы две реализации не разъехались молча.

    Слово, которое сознательно переполняет бюджет (пол кегля, см. `fit_size`),
    даёт прямоугольник шире кадра — с отрицательным `x`. Это не ошибка расчёта,
    а правда о кадре: панель действительно вылезет, и зажимать её здесь нельзя.
    Прямоугольник обязан НАКРЫВАТЬ текст (по нему меряется яркость фона и
    разводится картинка); зажатый бюджетом, он бы врал про место текста.

    Накрывает он, пока межстрочие не меньше единицы: это КОРОБКА строк, а глифы
    при тесном межстрочии торчат из своего строчного бокса. У титра (§5,
    межстрочие 0.5) они выходят за коробку на 6 px сверху и снизу — замерено на
    кадре. Для развода с картинкой это в пределах зазора `broll_margin_ratio`
    (54 px), а занижать коробку ради них значило бы разойтись с Chromium,
    который рисует именно её.

    `left` уезжает в `block_rect` как есть: тихий регистр стоит у левого края
    кадра, а не по центру (§1.4.1). `None` — центр.
    """
    text_w = max(widths, default=0.0)
    gap = line_gap(preset)
    # Строчный бокс не бывает ниже СТРУТА — это знает `geometry.line_box_height`.
    # Пока высота считалась только по самому крупному слову, ужатая строка
    # занижала блок на 45-164 px, и весь промах уходил вверх: блок прибит к низу
    # кадра.
    text_h = sum(
        line_box_height((w.size for w in line if w.size), preset) for line in lines
    ) + gap * max(0, len(lines) - 1)
    return block_rect(position, text_w, text_h, preset, frame, left=left)


# --------------------------------------------------------------------------- титр


def title_card_words(
    title: str, card: TitleCard, window: tuple[float, float], *, uppercase: bool = False
) -> list[PlanWord]:
    """Заголовок ролика → слова титра (STYLE_PLAN §5).

    Крупным идёт ПОСЛЕДНЕЕ слово заголовка, остальные — подписью над ним. Два
    довода, и оба про раскладку, а не про вкус:

    * Так устроен титр эталона: подпись сверху, одно огромное слово под ней.
      Перенос у нас жадный (`_greedy`) и порядок слов не меняет, поэтому на
      отдельную строку слово уходит только с КРАЯ последовательности. Слово
      покрупнее из середины (так выбирает `director.lead_word` для плаката, С3)
      разорвало бы подпись пополам, и структура регистра рассыпалась бы.
    * Заголовок пишет `make_metadata` как обещание ролика, и его конец —
      именно та точка, ради которой он написан.

    Слова приходят по одному, как в обычном блоке: место под ещё не пришедшее
    зарезервировано заранее (замер §1.3 — левый край подписи не шевелится по
    мере набора), это обычный механизм ghost-слов. Шаг — `word_step`, но
    последнее слово обязано встать не позже середины жизни титра: заголовок из
    восьми слов иначе дорисовывался бы ровно к своему исчезновению.

    `word_idx` отрицательный: в транскрипте этих слов нет вовсе, ссылаться им
    не на что, а рендереру нужен только уникальный в блоке ключ.
    """
    start, end = window
    parts = [display_text(p, uppercase=uppercase) for p in title.split()]
    parts = [p for p in parts if p]
    if not parts:
        return []
    half = max(0.0, (end - start) / 2)
    step = min(card.word_step, half / (len(parts) - 1)) if len(parts) > 1 else 0.0
    return [
        PlanWord(
            text=text,
            start=round(start + i * step, 3),
            end=round(end, 3),
            emphasis="accent" if i == len(parts) - 1 else "none",
            word_idx=-1 - i,
        )
        for i, text in enumerate(parts)
    ]


# --------------------------------------------------------------------------- сборка


def _resolve_position(
    requested: str,
    preset_position: str,
    *,
    collides_bottom: bool = False,
    collides_top: bool = False,
    collides_center: bool = False,
) -> Literal["top", "center", "bottom"]:
    """Явную позицию уважаем, «auto» решаем сами.

    Единственное настоящее решение здесь — разъехаться с b-roll: если картинка
    накрывает полосу субтитра, блок уходит в свободную. Занято везде — уходить
    некуда, остаёмся на месте: прыгать в такой ситуации значит только менять
    один перехлёст на другой.

    Центр — такая же полоса, как остальные две, и уклонение ему нужно не меньше:
    `center` — позиция по умолчанию у пресета bold, то есть у половины
    поставляемых тем. Из центра уходим вниз (привычное место субтитра), а если
    занят и низ — наверх.
    """
    if requested in ("top", "center", "bottom"):
        return requested  # type: ignore[return-value]
    base = preset_position if preset_position in ("top", "center", "bottom") else "bottom"
    if base == "bottom" and collides_bottom and not collides_top:
        return "top"
    if base == "top" and collides_top and not collides_bottom:
        return "bottom"
    if base == "center" and collides_center:
        if not collides_bottom:
            return "bottom"
        if not collides_top:
            return "top"
    return base  # type: ignore[return-value]


# Ритм позиции у блоков «auto». Субтитр, приклеенный к одному месту на весь
# ролик, глаз перестаёт замечать, а прыгающий каждую фразу — утомляет. Поэтому
# у плаката есть дом (позиция темы, обычно низ), и изредка одна фраза уходит на
# другую сторону кадра и возвращается. Числа — про «изредка»: дом держится не
# меньше `RHYTHM_HOME_MIN`, вылазка длится от `RHYTHM_AWAY_MIN` до
# `RHYTHM_AWAY_MAX` и начинается только с новой фразы, а не посреди мысли.
RHYTHM_HOME_MIN = 12.0
RHYTHM_AWAY_MIN = 1.5
RHYTHM_AWAY_MAX = 5.0
# Пауза между блоками, после которой следующий считается новой фразой, даже если
# предыдущий не закончился знаком.
RHYTHM_PHRASE_GAP = 0.25
# Граница фразы — не только точка: предложение живой речи идёт по 6-8 с и в
# вылазку целиком не влезает, а запятая — уже место, где мысль переводит дух.
RHYTHM_PHRASE_MARKS = (".", "!", "?", "…", ",", ";", ":", "—", "–")
RHYTHM_SIDES = ("top", "bottom")
# Лицо, которое вылазка не имеет права закрыть: от глаз до подбородка в долях
# высоты лица из трека (`TrackPoint.h`) и ширина в тех же долях. Лоб под текстом
# допустим — так ставят титры над головой и в эталоне.
FACE_GUARD_UP = 0.25
FACE_GUARD_DOWN = 0.5
FACE_WIDTH_RATIO = 0.8
FACE_MIN_CONF = 0.5


@dataclass(slots=True)
class RhythmSpan:
    """Блок глазами ритма: когда звучит и волен ли он сменить место.

    `fixed` — позиция, которую блок займёт сам: явная от режиссёра или дом у
    блока, которому уходить нельзя. `None` — блок «auto», ритм решает за него.
    """

    start: float
    end: float
    phrase: bool
    fixed: str | None = None


def caption_rhythm(
    spans: Sequence[RhythmSpan],
    home: str,
    *,
    can_leave: Any = None,
) -> list[str]:
    """Позиция каждого блока: дом или редкая вылазка на другую сторону кадра.

    Вылазка — это фраза целиком: блоки подряд с начала фразы до её конца или до
    `RHYTHM_AWAY_MAX`. Короче `RHYTHM_AWAY_MIN` она не ставится вовсе: блок,
    мелькнувший наверху на полсекунды, читается как сбой, а не как приём.
    `can_leave(i)` — может ли блок `i` встать на другой стороне (там лицо —
    не может); вылазка, которую он обрывает, тоже не ставится.

    Блок с явной позицией не на доме — та же вылазка, только режиссёрская
    (тихий блок наверху): после него дом снова отсчитывается заново.
    """
    if home not in ("top", "bottom"):
        return [s.fixed or home for s in spans]
    away = "top" if home == "bottom" else "bottom"
    leave = can_leave or (lambda i: True)
    out = [s.fixed or home for s in spans]
    home_since = spans[0].start if spans else 0.0
    i = 0
    while i < len(spans):
        span = spans[i]
        if span.fixed is not None:
            if span.fixed != home:
                home_since = span.end
            i += 1
            continue
        if not span.phrase or span.start - home_since < RHYTHM_HOME_MIN:
            i += 1
            continue
        run = [i]
        j = i + 1
        while (
            j < len(spans)
            and spans[j].fixed is None
            and not spans[j].phrase
            and spans[j].end - span.start <= RHYTHM_AWAY_MAX
        ):
            run.append(j)
            j += 1
        if spans[run[-1]].end - span.start > RHYTHM_AWAY_MAX:
            run.pop()  # фраза из одного длинного блока: дом
        # Возвращается вылазка тоже на границе фразы: уйти наверх с началом
        # мысли и спуститься посреди неё — тот же разрыв, только в конце.
        after = run[-1] + 1 if run else i
        if (
            not run
            or spans[run[-1]].end - span.start < RHYTHM_AWAY_MIN
            or (after < len(spans) and not spans[after].phrase and spans[after].fixed is None)
            or not all(leave(k) for k in run)
        ):
            i += 1
            continue
        for k in run:
            out[k] = away
        home_since = spans[run[-1]].end
        i = run[-1] + 1
    return out


@dataclass(slots=True)
class _Place:
    """Блок, разложенный под ОДНУ позицию — вместе с регистром, в котором он там играет.

    Регистров у субтитра два (STYLE_PLAN §1.4.1). Регистр задаёт кегль, бюджет
    колонки, число строк, выключку и наличие выделения — то есть ВСЮ раскладку,
    поэтому «одно измерение на все позиции» больше не держится: у каждой позиции
    своя раскладка и свой прямоугольник. У блока с объявленным регистром они
    совпадут, у листа без поля (`quiet=None`) — нет, там регистр решает
    позиция.
    """

    quiet: bool
    lines: list[list[PlanWord]]
    rect: Rect | None


@dataclass(slots=True)
class _Laid:
    """Блок, уже разложенный в строки, но ещё не выбравший позицию.

    Между раскладкой и позицией стоит картинка: «auto» уклоняется от неё, а сама
    она уступает тексту. Разорвать этот круг можно только так — сначала измерить
    блок во ВСЕХ позициях, какие он может занять (`places`), потом развести
    картинку со всеми ними, потом выбрать одну.
    """

    block: Any
    start: float
    end: float
    places: dict[str, _Place]


def build_plan(
    edl: EDL,
    video_path: str,
    *,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    preset: Preset | None = None,
    apply_cuts: bool = True,
    audio_path: str | None = None,
    cfg: Settings | None = None,
    meta: dict[str, Any] | None = None,
    track: FaceTrack | None = None,
) -> RenderPlan:
    """EDL → RenderPlan.

    `apply_cuts=True` означает, что видео УЖЕ нарезано по EDL.cuts и время надо
    пересчитать на выходной таймлайн. `False` — рисуем поверх исходника.

    `track` — трек лица из `crop.project_track` (М1). Нужен только блокам с
    `anchor_track`; без него они рисуются статично, как до М2. Трек приходит
    отдельным входом, а не полем EDL, потому что минута при 30 fps — это 1800
    точек, и EDL перестал бы быть документом, который правят руками.
    """
    cfg = cfg or default_settings
    fps = fps or cfg.render_fps
    width = width or cfg.render_width
    height = height or cfg.render_height
    ps = (preset or load_preset(edl.preset)).scaled(height)

    tl = edl.timeline() if apply_cuts else Timeline([], edl.source_duration)
    duration = tl.duration
    by_idx = {w.idx: w for w in edl.words}
    frame = (width, height)

    # Пресет тихого регистра считается один раз на весь план: это та же тема с
    # подменёнными числами (`Quiet.preset`), и дальше весь общий код — перенос,
    # бюджет строки, высота строчного бокса, коробка блока, резерв под картинку —
    # работает с ним без единой ветки «а если тихий».
    quiet = ps.caption.quiet
    quiet_ps = quiet.preset(ps) if quiet.enabled else ps

    def register_preset(silent: bool) -> Preset:
        """Пресет регистра: та же тема с подменёнными числами (`Quiet.preset`)."""
        return quiet_ps if silent else ps

    def plays_quiet(block: Any, place: str) -> bool:
        """Играет ли блок тихо, встав в эту позицию.

        Регистр объявляет режиссёр (`CaptionBlock.quiet`) — читаем заявку, а не
        выводим её заново (§0). Выводить приходится только у листов, сохранённых
        до появления поля (`quiet=None`): там регистр решала позиция, и такой
        лист обязан рендериться как рендерился.

        Гейт темы остаётся на месте и стоит ПЕРЕД заявкой: второй регистр — это
        данные темы, у glass и bold его нет, и `quiet=true` там не меняет ничего
        — верх играет плакатом, как играл. `plays_at` проверяет тот же
        выключатель, поэтому ветка необъявленного регистра его не дублирует.
        """
        if not quiet.enabled:
            return False
        declared = getattr(block, "quiet", None)
        if declared is not None:
            return bool(declared)
        return quiet.plays_at(place)

    def layout(
        words: list[PlanWord], preset_of: Preset
    ) -> tuple[list[list[PlanWord]], list[float]]:
        """Перенос строк блока в одном регистре. Ширины пустые — мерить было нечем.

        Метрик шрифта может не быть (свой TTF без выгрузки, пресет без
        `font.file`). Это не повод не отрендерить ролик: переносим по символам
        (лимит считается от кегля пресета), кегли не проставляем, а без
        прямоугольника рендерер и сам переносит строку по правилам CSS — это его
        последняя защита от строки во весь кадр. Картинку в этом случае разводит
        резерв `caption_zone`, как раньше.
        """
        try:
            return wrap_block(words, preset_of, width)
        except (FileNotFoundError, ValueError) as err:
            log.warning("геометрия субтитра не посчитана (%s), рендерер разложит сам", err)
            chars = fallback_line_chars(preset_of, width)
            return wrap_words(words, chars, preset_of.caption.max_lines), []

    def lay_places(
        places: Sequence[str], words: list[PlanWord], block: Any
    ) -> dict[str, _Place]:
        """Раскладка и прямоугольник блока в каждой позиции, какую он может занять.

        Раньше здесь стояло одно измерение на все позиции: `wrap_block` позицию
        не видел, и от неё зависела только вертикальная привязка готовой коробки.
        Со вторым регистром (§1.4.1) это неверно — регистр решает кегль, бюджет
        колонки, число строк и выделение, то есть тот же блок в двух регистрах
        разложен РАЗНЫМ текстом. У блока с объявленным регистром он один на все
        позиции, у листа без поля зависит от позиции (`plays_quiet`), поэтому
        раскладка считается по регистрам (их два, а позиций три, и общий регистр
        считается один раз), а прямоугольник — по каждой позиции отдельно, своим
        регистром.

        Инвариант, который обязан пережить правку: картинка разводится со ВСЕМИ
        этими прямоугольниками (DoD Ф7), и в какой бы позиции блок ни оказался,
        это настоящий прямоугольник именно его регистра, а не оценка.
        """
        by_register: dict[bool, tuple[Preset, list[list[PlanWord]], list[float]]] = {}
        out: dict[str, _Place] = {}
        for place in places:
            silent = plays_quiet(block, place)
            if silent not in by_register:
                pr = register_preset(silent)
                # Свои слова на каждый регистр, а не общие. Два повода: кегль
                # `wrap_block` проставляет словам НА МЕСТЕ (он же им мерил
                # строку), и у тихого блока выделения нет вовсе — ни золота, ни
                # крика, ни увеличенного слова. Гасится оно здесь, в самих
                # словах, а не подменой цвета в рендерере: тогда и кегль, и
                # поворот крика, и цвет отпадают разом и у обоих рендереров.
                own = [replace(w, emphasis="none") if silent else replace(w) for w in words]
                lines, widths = layout(own, pr)
                by_register[silent] = (pr, lines, widths)
            pr, lines, widths = by_register[silent]
            out[place] = _Place(
                quiet=silent,
                lines=lines,
                rect=(
                    _rect_of_lines(
                        place,
                        lines,
                        widths,
                        pr,
                        frame,
                        left=quiet.left_x(width) if silent else None,
                    )
                    if widths
                    else None
                ),
            )
        return out

    # Раскладка субтитров идёт ПЕРВОЙ, картинки — после неё. Раньше было
    # наоборот, и картинка разводилась не с блоком, а с ЗАРЕЗЕРВИРОВАННОЙ под
    # него полосой `caption_zone`. Резерв считается без текста и потому обязан
    # быть оценкой сверху, а быть ею не может: `wrap_block` при упоре в пол кегля
    # сознательно отдаёт больше строк, чем `caption.max_lines`, и такой блок выше
    # резерва (631 px против 431 у glass) — карточка вставала внутрь текста.
    # Здесь блоки уже измерены, и картинка разводится с их НАСТОЯЩИМИ
    # прямоугольниками. Это и есть DoD Ф7 «картинка ни разу не пересекает
    # субтитр»: гарантия, а не оценка.
    #
    # Позиция при этом ещё не выбрана: «auto» уклоняется от картинки, а картинка
    # считается позже. Поэтому у блока их несколько — все, какие он может
    # получить, — и картинка уступает КАЖДОЙ. Раскладка считается под каждую из
    # них (§1.4.1: позиция решает регистр, а регистр — всю раскладку), см.
    # `lay_places`.
    home = ps.caption.position

    def roams(block: Any) -> bool:
        """Волен ли блок уйти из дома по ритму (`caption_rhythm`).

        Только «auto» и только плакат. Регистр обязан быть объявлен: у листа без
        поля (`quiet=None`) его решает позиция, и вылазка наверх молча
        перекрасила бы плакат в тихий. Блок, привязанный к голове, ведёт голова.
        """
        if block.position in CAPTION_POSITIONS or home not in RHYTHM_SIDES:
            return False
        if getattr(block, "anchor_track", None) is not None:
            return False
        return not quiet.enabled or getattr(block, "quiet", None) is False

    def covers_face(start: float, end: float, rect: Rect) -> bool:
        """Закрывает ли блок лицо — от глаз до подбородка — хоть на одном кадре.

        Лоб и волосы под текстом допустимы, глаза и рот нет. Нет трека — кроп
        лица не нашёл, и закрывать нечего.
        """
        if track is None:
            return False
        for f in range(round(start * track.fps), round(end * track.fps) + 1):
            p = track.at(f)
            if p is None or p.conf < FACE_MIN_CONF:
                continue
            face_h = p.h * height
            top_y = p.cy * height - FACE_GUARD_UP * face_h
            low_y = p.cy * height + FACE_GUARD_DOWN * face_h
            half_w = FACE_WIDTH_RATIO * face_h / 2
            if (
                rect.y < low_y
                and rect.y + rect.height > top_y
                and rect.x < p.cx * width + half_w
                and rect.x + rect.width > p.cx * width - half_w
            ):
                return True
        return False

    def settle_rhythm(items: list[_Laid], phrases: list[bool]) -> list[str]:
        """Решает ритм и оставляет пришпиленному блоку одну измеренную позицию.

        Отбросить лишнее нужно ДО картинок: картинка уступает каждой позиции из
        `places`, и пришпиленная тема держит одну полосу затем, чтобы вставке
        хватило места (`Caption.pinned`).
        """
        away = "top" if home == "bottom" else "bottom"

        def can_leave(i: int) -> bool:
            there = items[i].places.get(away)
            if there is None or there.rect is None:
                return False  # не измерен — нечем проверить ни лицо, ни картинку
            return not covers_face(items[i].start, items[i].end, there.rect)

        spans = [
            RhythmSpan(
                start=item.start,
                end=item.end,
                phrase=phrase,
                fixed=None if roams(item.block) else (
                    item.block.position if item.block.position in CAPTION_POSITIONS else home
                ),
            )
            for item, phrase in zip(items, phrases)
        ]
        chosen = caption_rhythm(spans, home, can_leave=can_leave)
        for item, pos in zip(items, chosen):
            if ps.caption.pinned and roams(item.block):
                item.places = {pos: item.places[pos]}
        return chosen

    laid: list[_Laid] = []
    phrases: list[bool] = []
    prev_word: Any = None
    prev_end = 0.0
    for block in edl.captions:
        pwords: list[PlanWord] = []
        for cw in block.words:
            w = by_idx.get(cw.word_idx)
            if w is None:
                continue
            window = tl.word_window(w)
            if window is None:
                continue
            text = display_text(w.text, uppercase=ps.font.uppercase)
            pwords.append(
                PlanWord(
                    text=text,
                    start=round(window[0], 3),
                    end=round(window[1], 3),
                    emphasis=(
                        cw.emphasis.value
                        if isinstance(cw.emphasis, Emphasis)
                        else str(cw.emphasis)
                    ),
                    emoji=cw.emoji,
                    word_idx=cw.word_idx,
                )
            )
        if not pwords:
            continue

        # Новая фраза — после знака или после паузы. Текст берётся из
        # транскрипта, а не из показанного: у темы в верхнем регистре
        # `display_text` может знаки и снимать.
        phrases.append(
            prev_word is None
            or pwords[0].start - prev_end >= RHYTHM_PHRASE_GAP
            or prev_word.text.rstrip().endswith(RHYTHM_PHRASE_MARKS)
        )
        prev_word, prev_end = by_idx[pwords[-1].word_idx], pwords[-1].end

        places: tuple[str, ...] = (
            (block.position,)
            if block.position in CAPTION_POSITIONS
            else caption_positions(ps)
        )
        if roams(block):
            # Ритм решает позицию позже, по замеру: мерим блок и на другой
            # стороне, а лишнее отбросим, как только решение будет принято.
            places = tuple(dict.fromkeys((*places, *RHYTHM_SIDES)))
        laid.append(
            _Laid(
                block=block,
                start=pwords[0].start,
                end=pwords[-1].end + cfg.sub_tail,
                places=lay_places(places, pwords, block),
            )
        )

    homes = settle_rhythm(laid, phrases)

    def text_zones(start: float, end: float) -> list[Rect]:
        """Прямоугольники, которые текст может занять в этом промежутке.

        По ОКНУ вставки, а не по всему ролику: у каждой картинки своё время, и
        блок с явным `center` в начале ролика к вставке в конце отношения не
        имеет. Общий на весь план набор позиций схлопывал полосу ниже порога и
        делал полноэкранными все карточки ролика разом.

        Блок, который посчитать не удалось (нет метрик шрифта), отдаёт резерв
        `caption_zone` в каждой своей позиции: без измерения это единственное,
        что можно уступить. Резерв считается ПРЕСЕТОМ РЕГИСТРА, в котором блок
        там играет, — иначе тихому блоку уступалась бы полоса под плакатный
        кегль, вчетверо выше настоящей, и карточка уезжала бы в полный кадр без
        причины.
        """
        out: list[Rect] = []
        for item in laid:
            if item.end <= start or item.start >= end:
                continue
            for place, laid_place in item.places.items():
                out.append(
                    laid_place.rect
                    if laid_place.rect is not None
                    else caption_zone(register_preset(laid_place.quiet), frame, place)
                )
        return out

    broll: list[PlanBroll] = []
    for b in edl.broll:
        w = by_idx.get(b.word_idx)
        if w is None:
            continue
        window = tl.word_window(w)
        if window is None:
            continue
        anim = b.anim.value if hasattr(b.anim, "value") else str(b.anim)
        start = round(window[0], 3)
        end = round(min(window[0] + b.duration, duration), 3)
        broll.append(
            PlanBroll(
                start=start,
                end=end,
                anim=anim,
                anchor=b.anchor,
                asset=b.asset,
                query=b.query,
                rect=broll_rect(
                    resting_anchor(b.anchor, anim),
                    ps,
                    frame,
                    # Пропорция нарисованного ФАЙЛА. Без неё `broll_rect` берёт
                    # пропорцию по умолчанию, и карточка выходит другой высоты,
                    # чем содержимое: `object-fit: cover` режет источник (§8.3).
                    aspect=getattr(b, "asset_aspect", None),
                    zones=text_zones(start, end),
                ),
                kb_dir=1 if getattr(b, "kb_dir", 1) >= 0 else -1,
                aspect=getattr(b, "asset_aspect", None),
            )
        )

    def occupied(item_laid: _Laid) -> dict[str, bool]:
        """Какие позиции блока картинка перекрывает за время его жизни.

        Настоящее пересечение прямоугольников вместо «той же половины кадра»:
        половинами `center` и `fullscreen` описывались как «занято всё», из-за
        чего уклонение выключалось именно там, где было нужнее всего
        (AUDIT §8.3). Прямоугольник блока здесь уже измерен — в каждой позиции,
        какую он может занять; без метрик остаётся резерв `caption_zone`, и он
        считается регистром ЭТОГО блока, а не угаданным по позиции.
        """
        start, end, places = item_laid.start, item_laid.end, item_laid.places
        hits = dict.fromkeys(CAPTION_POSITIONS, False)
        for item in broll:
            if item.end <= start or item.start >= end or item.rect is None:
                continue
            for place in CAPTION_POSITIONS:
                laid_place = places.get(place)
                rect = laid_place.rect if laid_place is not None else None
                hits[place] = hits[place] or (
                    item.rect.overlaps(rect)
                    if rect is not None
                    else blocks_caption(
                        item.rect,
                        register_preset(plays_quiet(item_laid.block, place)),
                        frame,
                        place,
                    )
                )
        return hits

    def anchor_path_of(item: _Laid, rect: Rect | None) -> list[list[float]] | None:
        """Путь блока, привязанного к голове (М2), или `None` для статичного.

        Считается ПОСЛЕДНИМ: нужен прямоугольник в уже выбранной позиции, а
        позиция зависит от того, куда встала картинка.

        Без прямоугольника привязки не будет, и это не деградация «на всякий
        случай»: путь — смещение ОТ ПОКОЯ, а покой без измерения неизвестен.
        Двигать блок относительно неизвестной точки значит гадать, где он
        окажется, — статичный блок тут честнее.
        """
        anchor = getattr(item.block, "anchor_track", None)
        if anchor is None or track is None or rect is None:
            return None
        start_frame = round(item.start * fps)
        frames = max(1, round(item.end * fps) - start_frame + 1)
        offsets = follow_offsets(
            track,
            start_frame=start_frame,
            frames=frames,
            rest_center=(rect.x + rect.width / 2, rect.y + rect.height / 2),
            frame_size=frame,
            offset_x=anchor.offset_x,
            offset_y=anchor.offset_y,
            follow_x=anchor.follow_x,
            follow_y=anchor.follow_y,
            # Высота блока и безопасная зона: без них блок на крупном плане
            # садится на глаза, а на сверхкрупном уезжает за край кадра. Обе
            # величины известны только здесь — блок уже измерен.
            #
            # Граница берётся из `geometry.safe_area_ratio`, а не из
            # `safe_margin_y`: тот держит ещё и `caption.margin` — у glass это
            # 380 px, отступ ПОКОЯ, а не физический предел. Привязанный блок
            # покой сознательно покидает, и мерить его отступом покоя значит
            # запретить приём ровно там, где он нужен.
            block_height=rect.height,
            bounds=(
                height * ps.geometry.safe_area_ratio,
                height * (1.0 - ps.geometry.safe_area_ratio),
            ),
        )
        if not offsets:
            return None
        drops = fallbacks(offsets)
        if drops:
            log.info(
                "блок %.2f-%.2f с: привязка к голове отпускала её %d раз",
                item.start, item.end, drops,
            )
        return [[dx, dy] for dx, dy in offsets]

    blocks: list[PlanBlock] = []
    for item, item_home in zip(laid, homes):
        hits = occupied(item)
        position = _resolve_position(
            item.block.position,
            item_home,
            collides_bottom=hits["bottom"],
            collides_top=hits["top"],
            collides_center=hits["center"],
        )
        chosen = item.places.get(position)
        if chosen is None:
            # Уклонение увело блок в позицию, которой у него в наборе не было.
            # Так выходит только у пришпиленного пресета (`caption.pinned`:
            # набор из одной позиции) под полнокадровой вставкой, то есть когда
            # занято и так всё. Прямоугольника у такой позиции нет — значит нет
            # и гарантии Ф7, — поэтому отдаём раскладку, которую посчитали, без
            # прямоугольника: ровно так этот угол вёл себя и до второго регистра.
            log.warning(
                "блок %.2f-%.2f с ушёл в позицию %r, не измеренную в наборе %s",
                item.start, item.end, position, tuple(item.places),
            )
            chosen = next(iter(item.places.values()))
            rect = None
        else:
            rect = chosen.rect
        blocks.append(
            PlanBlock(
                start=item.start,
                end=item.end,
                position=position,
                lines=chosen.lines,
                rect=rect,
                bg_luma=getattr(item.block, "bg_luma", None),
                panel=bool(getattr(item.block, "panel", False)),
                anchor_path=anchor_path_of(item, rect),
                quiet=chosen.quiet,
            )
        )

    # Блоки не должны накладываться: следующий начинается — предыдущий уходит.
    for a, b in zip(blocks, blocks[1:]):
        if a.end > b.start:
            a.end = b.start
    if blocks and duration > 0:
        blocks[-1].end = min(blocks[-1].end, duration)

    def title_card_block() -> PlanBlock | None:
        """Титр ролика или `None`, если ему в этом ролике не место (STYLE_PLAN §5).

        Текст приезжает заголовком листа (`EDL.title`), а не из речи: титр — это
        обещание ролика, а не реплика. Числа — из темы (`caption.title_card`),
        которая выражена пресетом и проходит через тот же общий код, что и два
        других регистра (`TitleCard.preset`).

        Титра НЕ БУДЕТ, и каждая причина — правило, а не осторожность:

        1. Тема титра не просит или заголовка нет (лист до появления поля, прогон
           без LLM) — рисовать нечего.
        2. Ролик кончается раньше титра. Обрезанный титр читается как сбой.
        3. Заголовок не уложился в свои две строки даже после ужатия. Три строки
           наверху кадра — это абзац, а не титр.
        4. После ужатия большое слово не крупнее потолка ПЛАКАТА этой темы. Весь
           смысл титра в том, что он крупнее любого субтитра ролика (эталон: 10.1 %
           высоты кадра против потолка плаката 7.5 %); ужатый до плакатного кегля,
           он перестаёт быть титром и становится вторым субтитром наверху.
        5. Полосу титра занимает субтитр или картинка.

        Про последнее — правило разведения. Титр УСТУПАЕТ всему и не двигает
        ничего: он не попадает в `text_zones`, то есть картинка про него не знает
        и место под него не освобождает (иначе необязательный блок выталкивал бы
        вставку в полный кадр), а занятость проверяется теми же прямоугольниками,
        что и везде в этом модуле. Уступает он целиком, а не частично: ужать
        титр значит отнять у него единственное, чем он является (пункт 4), а
        обрезать по времени — показать половину заголовка (слова приходят по
        одному), и обе половинчатости хуже честного отсутствия.

        Что при этом НЕ считается занятостью: субтитр, который просто звучит в
        это же время. У нас речь начинается с первых кадров (`cut_pad_before`
        0.15 с), и запрет «титр только в тишине» означал бы, что титра не бывает
        никогда. Плакат живёт внизу кадра, титр — наверху, и мешают они друг
        другу тогда же, когда любые два блока: когда пересекаются
        прямоугольники. В эталоне вопрос не возникает вовсе — там первый субтитр
        приходит на 2.4 с, через секунду после титра.
        """
        card = ps.caption.title_card
        text = (edl.title or "").strip()
        if not card.enabled or not text:
            return None
        start = card.start
        end = start + card.duration
        if duration > 0 and end > duration:
            log.info("титр не поставлен: ролик (%.2f с) короче титра", duration)
            return None

        card_ps = card.preset(ps)
        words = title_card_words(
            text, card, (start, end), uppercase=card_ps.font.uppercase
        )
        if not words:
            return None
        lines, widths = layout(words, card_ps)
        if not widths:
            # Метрик шрифта нет, строка не измерена. Субтитр в этом случае
            # отдаёт раскладку браузеру, а титр — нет: без прямоугольника его
            # нечем развести с остальным кадром, а стоит он поверх лица.
            log.info("титр не поставлен: строку нечем измерить")
            return None
        if len(lines) > card_ps.caption.max_lines:
            log.info("титр не поставлен: заголовок не уложился в %d строки", card.max_lines)
            return None
        biggest = max((w.size or 0.0) for w in words)
        poster_ceiling = ps.font.size * max(1.0, ps.caption.fill_max_scale)
        if biggest <= poster_ceiling:
            log.info(
                "титр не поставлен: большое слово ужато до %.0f px при потолке плаката %.0f px",
                biggest, poster_ceiling,
            )
            return None

        rect = _rect_of_lines("top", lines, widths, card_ps, frame)
        if any(rect.overlaps(zone) for zone in text_zones(start, end)):
            log.info("титр не поставлен: его полосу занимает субтитр")
            return None
        if any(
            pic.rect is not None
            and pic.start < end
            and pic.end > start
            and pic.rect.overlaps(rect)
            for pic in broll
        ):
            log.info("титр не поставлен: его полосу занимает картинка")
            return None
        return PlanBlock(
            start=start,
            end=end,
            position="top",
            lines=lines,
            rect=rect,
            title_card=True,
        )

    card_block = title_card_block()
    if card_block is not None:
        # Титр идёт ПЕРВЫМ и в обрезку соседей выше не попадает намеренно: она
        # разводит блоки, стоящие в одном месте кадра друг за другом, а титр
        # стоит в своей полосе — пустой, иначе его бы тут не было, — и обрывать
        # его началом нижнего субтитра не за что.
        blocks.insert(0, card_block)

    return RenderPlan(
        version=1,
        fps=fps,
        width=width,
        height=height,
        duration=duration,
        video=video_path,
        audio=audio_path,
        preset=ps.model_dump(),
        blocks=blocks,
        broll=broll,
        meta=meta or {},
    )
