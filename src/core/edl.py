"""EDL (Edit Decision List) — контракт системы.

Восприятие (детерминированное) → LLM-режиссёр (JSON) → Рендерер (тупой).

LLM возвращает ТОЛЬКО ссылки на индексы слов. Таймкоды подставляет Python из
вывода ASR (§0.1). Все enum строго закрытые — иначе модель изобретёт эффект,
которого нет в рендерере.

Здесь же живёт семантический валидатор (§3) — отдельно от pydantic, потому что
он проверяет связи между полями, а не форму каждого поля.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.config import settings as default_settings

# Потолок b-roll: не чаще одной картинки на BROLL_WINDOW секунд и не больше
# BROLL_MAX на ролик. Одна картинка на 8 с — это мельтешение: на 20-секундном
# ролике их выходило три, и ролик переставал быть про говорящего.
BROLL_WINDOW = 12.0
BROLL_MAX = 2
# Сколько слов допустимо в одном блоке субтитров. Регистров ДВА (STYLE_PLAN
# §1.4.1), и потолок у них разный.
BLOCK_MIN_WORDS = 1
# Плакат — нижний регистр: 1-2 строки огромным кеглем, одно золотое слово.
# Потолок прежний (5 = `sub_max_words` + 1): больше пяти слов в две строки при
# таком кегле не влезает.
BLOCK_POSTER_MAX_WORDS = 5
# Тихий — верхний регистр: 4 строки мелким кеглем у левого края, золота нет
# вообще. В эталоне это «Я лишь отправляю ему сырое видео» (3.6-5.2 с) — шесть
# слов в четыре строки, то есть ~1.5 слова на строку.
BLOCK_QUIET_MIN_WORDS = 6
# Потолок поднят с 5 до 8 ровно под тихий регистр: при ~2 словах на строку
# четыре строки держат восемь слов, а на пяти словах тихий блок невозможен в
# принципе — он вырождается в плакат. Причина правки — замер золотой маски: у
# эталона размер главного слова гуляет 2.7-7.5 % высоты кадра (медиана 4.5 %), у
# нас до этой работы 5.4-7.6 % (медиана 6.5 %) — разброса не было вовсе, потому
# что все блоки выходили одной длины. Даёт его разная длина блока, а не мелкий
# шрифт: с двумя регистрами замер стал 3.0-6.8 % при медиане 5.0 %.
BLOCK_MAX_WORDS = 8
# Сколько блоков за ролик могут ездить за головой (М2). Приём для ОДНОГО
# смыслового пика: когда за головой ездят все субтитры, ролик невозможно
# смотреть. Промпт `direct.md` это объясняет, валидатор — обеспечивает.
ANCHOR_TRACK_MAX = 2
# Сколько блоков за ролик носят матовое стекло. В эталоне 23.6 с плашка есть у
# двух блоков из тринадцати; поставить её всем — значит вернуть тот самый фон,
# от которого мы уходили. Число взято с эталона, а не выбрано.
PANEL_MAX = 2


# --------------------------------------------------------------------------- модели


class Word(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idx: int  # позиция в исходном транскрипте
    text: str
    start: float  # сек, из ASR
    end: float
    prob: float = 1.0  # уверенность ASR, нужна для фильтра галлюцинаций

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0


class Cut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src_in: float
    src_out: float
    reason: Literal["hook", "keep", "drop_silence", "drop_filler"]
    # Крупность куска. Движение в кадре делается сменой крупности на резе, а не
    # зумом внутри куска: оно мотивировано монтажом, не спорит с субтитрами и не
    # требует физики по номеру кадра. Ставит Python по готовым резам, не LLM.
    framing: Literal["wide", "tight"] = "wide"

    @property
    def kept(self) -> bool:
        return self.reason in ("hook", "keep")

    @property
    def duration(self) -> float:
        return max(0.0, self.src_out - self.src_in)


class Emphasis(str, Enum):
    NONE = "none"
    ACCENT = "accent"  # крупно + акцентный цвет
    SHOUT = "shout"  # ещё крупнее + лёгкий поворот


class CaptionWord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word_idx: int
    emphasis: Emphasis = Emphasis.NONE
    emoji: str | None = None


class AnchorTrack(BaseModel):
    """Привязка блока к треку лица (М2).

    Смещение — доли кадра, а не пиксели: трек нормализованный, и привязка обязана
    пережить смену разрешения вместе с ним. Минус по Y — вверх, то есть блок
    висит НАД головой, а не на ней.

    Числа — краткая часть решения, и её принимает не LLM: модель решает, нужен ли
    приём в этом месте ролика, а насколько выше головы стоит текст — это ремесло
    (§0). Поэтому в схему `direct` уезжает один булев флаг.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["face"] = "face"
    offset_y: float = Field(default=-0.14, ge=-0.5, le=0.5)
    offset_x: float = Field(default=0.0, ge=-0.5, le=0.5)
    follow_x: bool = True
    follow_y: bool = True


class CaptionBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    words: list[CaptionWord] = Field(min_length=1)  # 3-5 слов
    position: Literal["top", "center", "bottom", "auto"] = "auto"
    # РЕГИСТР субтитра (STYLE_PLAN §1.4.1): `False` — плакат (крупно, по центру,
    # одно золотое слово), `True` — тихий (мелко у левого края, 4 строки, золота
    # нет вовсе). Имя поля то же, что у регистра дальше по конвейеру
    # (`RenderPlan` → `Captions.tsx`), чтобы решение звалось одним словом на всём
    # пути.
    #
    # Объявляет РЕЖИССЁР, как `position` и `panel`. До этого поля регистр
    # выводился из позиции — «наверху значит тихий», — и из-за этого второй
    # регистр верха был невыразим в принципе. А верх в эталоне носит ОБА: тихий
    # блок на 3.6-5.2 с и ЗОЛОТОЙ плакат на 9.2-12.0 с, поднятый наверх потому,
    # что центр занят вставкой (§1.4.1 и §1.5). Под старым правилом такой блок
    # терял золото, которое на кадре есть.
    #
    # `None` — регистр не объявлен: лист, сохранённый до появления поля. Такой
    # лист обязан и рендериться, и проходить валидацию ровно как раньше, а «как
    # раньше» у двух потребителей РАЗНОЕ, потому что примета была разная:
    # рендерер смотрел на позицию (он знает тему, см. `Quiet.plays_at`), а
    # валидатор — на длину (темы он не знает вовсе, а длиннее плаката блок бывает
    # только тихий). Поэтому `None` каждый разрешает своей старой приметой:
    # контрактную сторону считает `quiet_register` здесь, рендерную —
    # `render_plan.plays_quiet`.
    quiet: bool | None = None
    layout: Literal["stack", "ragged", "single_line"] = "stack"
    # `None` — статичная позиция, то есть поведение до М2. Приём редкий: если за
    # головой ездят все субтитры, смотреть невозможно, поэтому валидатор держит
    # потолок ANCHOR_TRACK_MAX.
    anchor_track: AnchorTrack | None = None
    # Средняя яркость кадра под блоком, 0..1. Заполняет Python замером, не LLM.
    # Единственное место, где стиль уступает измерению: панель фиксированной
    # прозрачности не читается над светлым фоном.
    bg_luma: float | None = Field(default=None, ge=0.0, le=1.0)
    # Матовое стекло под ЭТИМ блоком. В эталоне плашка есть у 2 блоков из 13, и
    # правило, по которому монтажёр её ставит, измерить не удалось: ни яркость
    # фона, ни его суетливость плашечные блоки не выделяют (STYLE_PLAN §1.4).
    # Значит это решение, а не следствие, и принимает его режиссёр — как
    # `position` и `anchor_track`. Потолок держит PANEL_MAX.
    #
    # Читаемость от флага НЕ зависит: правило контраста (Ф9) поднимает заливку
    # на любом блоке, где порог не взят. Флаг добавляет стекло — размытие и
    # рамку, — то есть стиль поверх закона.
    #
    # У тихого регистра плашки не бывает: замер §1.4.1 — «плашка: иногда» у
    # плаката и «нет» у тихого. Тему это и так держит (`Quiet.preset` гасит
    # панель), а заявку «тихий блок со стеклом» валидатор отклоняет.
    panel: bool = False

    def quiet_register(self) -> bool:
        """Тихий ли это регистр по КОНТРАКТУ — то есть без оглядки на тему.

        Заявку режиссёра берём как есть. Необъявленный регистр (`None`, лист до
        появления поля) разрешаем ДЛИНОЙ, ровно как разрешал прежний потолок:
        больше `BLOCK_POSTER_MAX_WORDS` слов кеглем плаката — это строка,
        уехавшая за кадр, значит такой блок бывает только тихим. Позицией здесь
        разрешать нельзя: тему валидатор не знает, а у glass и bold второго
        регистра нет вовсе — верхний блок у них плакат, и золото ему нужно.
        """
        if self.quiet is None:
            return len(self.words) > BLOCK_POSTER_MAX_WORDS
        return self.quiet


class BrollAnim(str, Enum):
    POP_IN = "pop_in"
    SLIDE_UP = "slide_up"
    FULLSCREEN_TO_INSET = "fullscreen_to_inset"


class Broll(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word_idx: int  # к какому слову привязана
    duration: float = Field(ge=0.8, le=3.0)
    query: str  # короткое имя предмета по-английски: ключ кеша и проверки картинки
    prompt: str | None = None  # что рисовать; пусто — рисуется `query`
    anim: BrollAnim
    anchor: Literal[
        "top_left", "top_right", "bottom_left", "bottom_right", "center", "fullscreen", "card"
    ]
    # Всё ниже заполняет Python после генерации, не LLM.
    asset: str | None = None
    # Пропорция ФАЙЛА (ширина/высота). Карточка наследует её, иначе `cover`
    # срезает картинку (AUDIT §8.3). Меряется файл, а не заказ: провайдер отдаёт
    # вертикаль на пиксель мимо заказанной.
    asset_aspect: float | None = Field(default=None, gt=0.0)
    # Направление медленного наезда (Ken Burns), М3. Неподвижная картинка поверх
    # видео читается как наклейка, поэтому вставке дают еле заметное движение.
    # Знак берётся из ПОЗИЦИИ вставки в списке, а не из `random`: рендерер обязан
    # быть воспроизводимым (инвариант 4), иначе перерендер сохранённого листа даёт
    # другое видео. Ставит Python, не LLM.
    kb_dir: Literal[-1, 1] = 1


class Sfx(BaseModel):
    """Звуковой акцент на резе или входе картинки. Расставляет Python, не LLM."""

    model_config = ConfigDict(extra="forbid")

    at_word_idx: int
    name: str  # имя файла в assets/sfx без расширения
    gain_db: float = -6.0


class Music(BaseModel):
    """Подложка на весь ролик с дакингом под голос."""

    model_config = ConfigDict(extra="forbid")

    track: str  # имя файла в assets/music
    gain_db: float = -20.0
    duck_db: float = -12.0


class EDL(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    source_duration: float
    words: list[Word]  # заполняет Python, не LLM
    cuts: list[Cut]
    captions: list[CaptionBlock]
    broll: list[Broll] = []
    sfx: list[Sfx] = []
    music: Music | None = None
    # Умолчание из конфига, а не зашитое «glass» (дефект №5): лист, собранный без
    # явной темы, обязан получить ту же тему, что и новый пользователь.
    # `default_factory` — чтобы значение читалось при создании листа, а не при
    # импорте модуля.
    preset: str = Field(default_factory=lambda: default_settings.default_preset)

    @model_validator(mode="before")
    @classmethod
    def _drop_stock_leftovers(cls, data: Any) -> Any:
        """Листы, сохранённые до снятия стока, обязаны читаться и сейчас.

        У вставок тех времён были поля `kind` и `asset_start`, а вставка с
        `kind: "video"` была клипом Pexels. Клипы рендерер больше не играет,
        поэтому такая вставка выпадает из листа целиком; у картинок лишние поля
        просто срезаются. Без этого `extra="forbid"` ронял бы перерендер любого
        старого ролика с вставкой.
        """
        if not isinstance(data, dict) or not isinstance(data.get("broll"), list):
            return data
        broll = []
        for item in data["broll"]:
            if isinstance(item, dict):
                if item.get("kind") == "video":
                    continue
                item = {k: v for k, v in item.items() if k not in ("kind", "asset_start")}
            broll.append(item)
        return {**data, "broll": broll}

    # ---------------------------------------------------------------- удобные срезы
    @property
    def kept_cuts(self) -> list[Cut]:
        return [c for c in self.cuts if c.kept]

    @property
    def out_duration(self) -> float:
        """Длительность результата: сумма оставленных кусков (или весь исходник)."""
        kept = self.kept_cuts
        if not kept:
            return self.source_duration
        return sum(c.duration for c in kept)

    def word_by_idx(self, idx: int) -> Word | None:
        if 0 <= idx < len(self.words) and self.words[idx].idx == idx:
            return self.words[idx]
        for w in self.words:
            if w.idx == idx:
                return w
        return None

    def timeline(self) -> "Timeline":
        return Timeline(self.kept_cuts, self.source_duration)


# --------------------------------------------------------------------------- таймлайн


class Timeline:
    """Пересчёт времени исходника во время результата после вырезания кусков.

    Рендерер получает уже пересчитанные тайминги: он ничего не решает (§0).
    """

    def __init__(self, kept: Iterable[Cut], source_duration: float) -> None:
        segs = sorted((c for c in kept), key=lambda c: c.src_in)
        self.segments: list[tuple[float, float, float]] = []  # (src_in, src_out, out_start)
        acc = 0.0
        for c in segs:
            if c.duration <= 0:
                continue
            self.segments.append((c.src_in, c.src_out, acc))
            acc += c.duration
        if not self.segments:
            self.segments = [(0.0, source_duration, 0.0)]
            acc = source_duration
        self.duration = acc

    def map(self, t: float, *, clamp: bool = True) -> float | None:
        """src-время → out-время. None, если точка попала в вырезанный кусок."""
        for src_in, src_out, out_start in self.segments:
            if src_in <= t <= src_out:
                return out_start + (t - src_in)
        if not clamp:
            return None
        # Точка в дырке: прижимаем к ближайшей границе, чтобы субтитр не пропал.
        best: tuple[float, float] | None = None
        for src_in, src_out, out_start in self.segments:
            for edge, out_edge in ((src_in, out_start), (src_out, out_start + (src_out - src_in))):
                d = abs(t - edge)
                if best is None or d < best[0]:
                    best = (d, out_edge)
        return best[1] if best else None

    def keeps(self, t: float) -> bool:
        return any(src_in <= t <= src_out for src_in, src_out, _ in self.segments)

    def word_window(self, w: Word) -> tuple[float, float] | None:
        """Окно слова на выходном таймлайне. None — слово вырезано."""
        if not self.keeps(w.mid):
            return None
        a = self.map(w.start)
        b = self.map(w.end)
        if a is None or b is None:
            return None
        if b <= a:
            b = a + 0.04
        return (a, b)


# --------------------------------------------------------------------------- валидатор


class EDLValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def kept_word_indices(edl: EDL) -> list[int]:
    """Слова, которые остались в ролике после нарезки, в порядке следования."""
    tl_cuts = edl.kept_cuts
    if not tl_cuts:
        return [w.idx for w in edl.words]
    segs = sorted(((c.src_in, c.src_out) for c in tl_cuts))
    out: list[int] = []
    for w in edl.words:
        m = w.mid
        if any(a <= m <= b for a, b in segs):
            out.append(w.idx)
    return out


def validate_edl(edl: EDL, *, strict_coverage: bool = True) -> list[str]:
    """Семантическая валидация (§3). Возвращает список ошибок; пустой список = ок."""
    errors: list[str] = []

    # --- words: индексы должны совпадать с позицией, иначе ссылки LLM разъедутся
    for pos, w in enumerate(edl.words):
        if w.idx != pos:
            errors.append(f"words[{pos}].idx={w.idx}, ожидалось {pos}")
            break
        if w.end < w.start:
            errors.append(f"word {w.idx}: end < start")
            break
    known = {w.idx for w in edl.words}

    # --- cuts: отсортированы, не пересекаются, внутри исходника
    prev_out = -1e9
    for i, c in enumerate(edl.cuts):
        if c.src_out <= c.src_in:
            errors.append(f"cuts[{i}]: src_out <= src_in ({c.src_in}..{c.src_out})")
        if c.src_in < -1e-6 or c.src_out > edl.source_duration + 0.5:
            errors.append(
                f"cuts[{i}]: выход за пределы исходника "
                f"({c.src_in:.2f}..{c.src_out:.2f} vs 0..{edl.source_duration:.2f})"
            )
        if c.src_in < prev_out - 1e-6:
            errors.append(f"cuts[{i}]: пересекается с предыдущим (src_in={c.src_in:.2f})")
        prev_out = max(prev_out, c.src_out)

    if edl.cuts and not edl.kept_cuts:
        errors.append("cuts: не осталось ни одного kept-куска, рендерить нечего")

    # --- captions: каждый word_idx существует, порядок строго возрастает, без дублей
    flat: list[int] = []
    for bi, block in enumerate(edl.captions):
        if not (BLOCK_MIN_WORDS <= len(block.words) <= BLOCK_MAX_WORDS):
            errors.append(
                f"captions[{bi}]: {len(block.words)} слов, допустимо "
                f"{BLOCK_MIN_WORDS}..{BLOCK_MAX_WORDS}"
            )
        accents = [cw for cw in block.words if cw.emphasis != Emphasis.NONE]
        if len(accents) > 1:
            errors.append(f"captions[{bi}]: {len(accents)} акцентных слов, допустимо не больше 1")
        # Правила регистра (§1.4.1). Раньше они были записаны через ДЛИНУ, и из
        # этого выходило, что золото теряет любой верхний блок. Теперь длина
        # связана с регистром только в одну сторону — длинный блок бывает лишь
        # тихим, потому что кеглем плаката такая строка уезжает за кадр, — а
        # обратное неверно: наверху стоят оба регистра, и плакат наверху остаётся
        # плакатом со своим золотым словом (эталон, 9.2-12.0 с).
        quiet = block.quiet_register()
        if len(block.words) > BLOCK_POSTER_MAX_WORDS and not quiet:
            errors.append(
                f"captions[{bi}]: {len(block.words)} слов в плакате — больше "
                f"{BLOCK_POSTER_MAX_WORDS} слов бывает только у тихого регистра "
                f'(quiet=true, position="top")'
            )
        if quiet:
            # Две приметы тихого регистра, обе замерены: он висит наверху и в
            # нём нет ни одного выделенного слова. Мелкое золото у левого края —
            # не иерархия, а пятно, и в эталоне его нет ни на одном кадре.
            if block.position != "top":
                # Причину называем ту, по которой регистр и оказался тихим:
                # заявку режиссёра или длину. Текст ошибки уезжает модели на
                # ретрае, и «блок объявлен тихим» с «блок длиннее плаката» —
                # разные правки с её стороны.
                why = (
                    "объявлен тихим"
                    if block.quiet
                    else (
                        f"длиннее {BLOCK_POSTER_MAX_WORDS} слов, а столько бывает "
                        f"только у тихого регистра"
                    )
                )
                errors.append(
                    f"captions[{bi}]: блок {why}, но стоит в position="
                    f'{block.position!r} — тихого блока вне верха (position="top") '
                    f"в эталоне нет"
                )
            if accents:
                errors.append(
                    f"captions[{bi}]: тихий регистр и акцент — у тихого блока "
                    f"выделенного слова нет вообще, акцент живёт в плакате "
                    f"до {BLOCK_POSTER_MAX_WORDS} слов"
                )
        # Стекло на тихом блоке — противоречие в самой заявке (§1.4.1: «плашка:
        # нет»). Спрашиваем только ОБЪЯВЛЕННЫЙ регистр: у необъявленного он
        # угадан, а не заявлен, и стекло на нём всё равно не рисуется —
        # `Quiet.preset` гасит панель вместе с её хромом.
        if block.quiet and block.panel:
            errors.append(
                f"captions[{bi}]: тихий регистр с плашкой — матовое стекло носит "
                f"только плакат"
            )
        for cw in block.words:
            if cw.word_idx not in known:
                errors.append(f"captions[{bi}]: word_idx={cw.word_idx} отсутствует в words")
            flat.append(cw.word_idx)

    anchored = sum(1 for b in edl.captions if b.anchor_track is not None)
    if anchored > ANCHOR_TRACK_MAX:
        errors.append(
            f"captions: {anchored} блоков привязано к треку лица, "
            f"допустимо не больше {ANCHOR_TRACK_MAX}"
        )

    panelled = sum(1 for b in edl.captions if b.panel)
    if panelled > PANEL_MAX:
        errors.append(
            f"captions: {panelled} блоков с плашкой, допустимо не больше {PANEL_MAX}"
        )

    if len(flat) != len(set(flat)):
        dupes = sorted({i for i in flat if flat.count(i) > 1})[:5]
        errors.append(f"captions: слова повторяются в нескольких блоках: {dupes}")
    if any(b <= a for a, b in zip(flat, flat[1:])):
        errors.append("captions: порядок слов нарушен (индексы должны строго возрастать)")

    if strict_coverage:
        expected = kept_word_indices(edl)
        missing = [i for i in expected if i not in set(flat)]
        extra = [i for i in flat if i not in set(expected)]
        if missing:
            errors.append(f"captions: слова не попали ни в один блок: {missing[:8]}")
        if extra:
            errors.append(f"captions: слова из вырезанных кусков попали в блоки: {extra[:8]}")

    # --- broll: жёсткий потолок по количеству и по плотности
    cap = min(BROLL_MAX, max(0, math.ceil(edl.out_duration / BROLL_WINDOW)))
    if len(edl.broll) > cap:
        errors.append(
            f"broll: {len(edl.broll)} штук при потолке {cap} "
            f"(1 на {BROLL_WINDOW:.0f} с, максимум {BROLL_MAX})"
        )
    tl = edl.timeline()
    stamps: list[float] = []
    for i, b in enumerate(edl.broll):
        if b.word_idx not in known:
            errors.append(f"broll[{i}]: word_idx={b.word_idx} отсутствует в words")
            continue
        w = edl.word_by_idx(b.word_idx)
        assert w is not None
        win = tl.word_window(w)
        if win is None:
            errors.append(f"broll[{i}]: привязан к вырезанному слову {b.word_idx}")
            continue
        stamps.append(win[0])
    stamps.sort()
    for a, b_ in zip(stamps, stamps[1:]):
        if b_ - a < BROLL_WINDOW - 1e-6:
            errors.append(f"broll: две картинки ближе {BROLL_WINDOW:.0f} с ({a:.1f} и {b_:.1f})")
            break

    # --- sfx
    for i, s in enumerate(edl.sfx):
        if s.at_word_idx not in known:
            errors.append(f"sfx[{i}]: word_idx={s.at_word_idx} отсутствует в words")

    # --- монотонность слов: тайминги — единственный источник правды о времени,
    # и если они врут, подсветка уезжает на слово. Проверка здесь, а не только в
    # тесте: сломанный ASR должен ронять валидацию, а не тихо ехать в рендер.
    for i in range(len(edl.words) - 1):
        cur, nxt = edl.words[i], edl.words[i + 1]
        if cur.end > nxt.start + 1e-6:
            errors.append(
                f"words[{i}]: конец {cur.end:.3f} позже начала следующего {nxt.start:.3f}"
            )
            break

    return errors


def assert_valid(edl: EDL, *, strict_coverage: bool = True) -> None:
    errs = validate_edl(edl, strict_coverage=strict_coverage)
    if errs:
        raise EDLValidationError(errs)
