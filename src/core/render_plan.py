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
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
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
from src.core.preset import Preset, load_preset
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
    # Фото или видео (М4) и с какой секунды играть клип. У фото `asset_start`
    # всегда 0: рендерер не выбирает кадр сам, ему говорят.
    kind: str = "photo"
    asset_start: float = 0.0
    # Пропорция файла (ширина/высота). Геометрии она нужна была и раньше — по
    # ней считается `rect`, — но у полнокадровой вставки `rect` всегда кадр
    # целиком, и по нему уже не понять, что там за источник. Рендереру это
    # решает один вопрос: класть ли под картинку размытую копию. Сток 2:3 без
    # неё обрезается, нарисованная вертикаль 9:16 с ней — только портится.
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
                    kind=x.get("kind") or "photo",
                    asset_start=float(x.get("asset_start") or 0.0),
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
                    "kind": x.kind,
                    "asset_start": round(x.asset_start, 3),
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
    return block_rect(position, text_w, text_h, preset, frame)


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


@dataclass(slots=True)
class _Laid:
    """Блок, уже разложенный в строки, но ещё не выбравший позицию.

    Между раскладкой и позицией стоит картинка: «auto» уклоняется от неё, а сама
    она уступает тексту. Разорвать этот круг можно только так — сначала измерить
    блок во ВСЕХ позициях, какие он может занять (`rects`), потом развести
    картинку со всеми ними, потом выбрать одну.
    """

    block: Any
    start: float
    end: float
    places: tuple[str, ...]
    lines: list[list[PlanWord]]
    rects: dict[str, Rect]


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

    def layout(
        places: Sequence[str], words: list[PlanWord]
    ) -> tuple[list[list[PlanWord]], dict[str, Rect]]:
        """Перенос строк и прямоугольники блока во всех возможных позициях.

        Одно измерение на все позиции: `wrap_block` позицию не видит, от неё
        зависит только вертикальная привязка готовой коробки.

        Метрик шрифта может не быть (свой TTF без выгрузки, пресет без
        `font.file`). Это не повод не отрендерить ролик: переносим по символам
        (лимит считается от кегля пресета), кегли не проставляем, а без
        прямоугольника рендерер и сам переносит строку по правилам CSS — это его
        последняя защита от строки во весь кадр. Картинку в этом случае разводит
        резерв `caption_zone`, как раньше.
        """
        try:
            lines, widths = wrap_block(words, ps, width)
        except (FileNotFoundError, ValueError) as err:
            log.warning("геометрия субтитра не посчитана (%s), рендерер разложит сам", err)
            chars = fallback_line_chars(ps, width)
            return wrap_words(words, chars, ps.caption.max_lines), {}
        return lines, {p: _rect_of_lines(p, lines, widths, ps, frame) for p in places}

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
    # получить, — и картинка уступает КАЖДОЙ. Раскладка от позиции не зависит,
    # поэтому меряется один раз на все.
    laid: list[_Laid] = []
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

        places: tuple[str, ...] = (
            (block.position,)
            if block.position in CAPTION_POSITIONS
            else caption_positions(ps)
        )
        lines, rects = layout(places, pwords)
        laid.append(
            _Laid(
                block=block,
                start=pwords[0].start,
                end=pwords[-1].end + cfg.sub_tail,
                places=places,
                lines=lines,
                rects=rects,
            )
        )

    def text_zones(start: float, end: float) -> list[Rect]:
        """Прямоугольники, которые текст может занять в этом промежутке.

        По ОКНУ вставки, а не по всему ролику: у каждой картинки своё время, и
        блок с явным `center` в начале ролика к вставке в конце отношения не
        имеет. Общий на весь план набор позиций схлопывал полосу ниже порога и
        делал полноэкранными все карточки ролика разом.

        Блок, который посчитать не удалось (нет метрик шрифта), отдаёт резерв
        `caption_zone` в каждой своей позиции: без измерения это единственное,
        что можно уступить.
        """
        out: list[Rect] = []
        for item in laid:
            if item.end <= start or item.start >= end:
                continue
            if item.rects:
                out.extend(item.rects.values())
            else:
                out.extend(caption_zone(ps, frame, p) for p in item.places)
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
                    # Пропорция скачанного ФАЙЛА (М4). Без неё `broll_rect` берёт
                    # стоковые 2:3 по умолчанию, и для генерации 3:4 или для
                    # видеоклипа 9:16 карточка выходит другой высоты, чем
                    # содержимое: `object-fit: cover` режет источник (§8.3).
                    aspect=getattr(b, "asset_aspect", None),
                    zones=text_zones(start, end),
                ),
                kb_dir=1 if getattr(b, "kb_dir", 1) >= 0 else -1,
                kind=getattr(b, "kind", "photo"),
                asset_start=float(getattr(b, "asset_start", 0.0) or 0.0),
                aspect=getattr(b, "asset_aspect", None),
            )
        )

    def occupied(start: float, end: float, rects: dict[str, Rect]) -> dict[str, bool]:
        """Какие позиции блока картинка перекрывает в этом промежутке.

        Настоящее пересечение прямоугольников вместо «той же половины кадра»:
        половинами `center` и `fullscreen` описывались как «занято всё», из-за
        чего уклонение выключалось именно там, где было нужнее всего
        (AUDIT §8.3). Прямоугольник блока здесь уже измерен — в каждой позиции,
        какую он может занять; без метрик остаётся резерв `caption_zone`.
        """
        hits = dict.fromkeys(CAPTION_POSITIONS, False)
        for item in broll:
            if item.end <= start or item.start >= end or item.rect is None:
                continue
            for place in CAPTION_POSITIONS:
                rect = rects.get(place)
                hits[place] = hits[place] or (
                    item.rect.overlaps(rect)
                    if rect is not None
                    else blocks_caption(item.rect, ps, frame, place)
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
    for item in laid:
        hits = occupied(item.start, item.end, item.rects)
        position = _resolve_position(
            item.block.position,
            ps.caption.position,
            collides_bottom=hits["bottom"],
            collides_top=hits["top"],
            collides_center=hits["center"],
        )
        rect = item.rects.get(position)
        blocks.append(
            PlanBlock(
                start=item.start,
                end=item.end,
                position=position,
                lines=item.lines,
                rect=rect,
                bg_luma=getattr(item.block, "bg_luma", None),
                panel=bool(getattr(item.block, "panel", False)),
                anchor_path=anchor_path_of(item, rect),
            )
        )

    # Блоки не должны накладываться: следующий начинается — предыдущий уходит.
    for a, b in zip(blocks, blocks[1:]):
        if a.end > b.start:
            a.end = b.start
    if blocks and duration > 0:
        blocks[-1].end = min(blocks[-1].end, duration)

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
