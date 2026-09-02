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
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

# Потолок b-roll: не чаще одной картинки на BROLL_WINDOW секунд (§3, M7 DoD).
BROLL_WINDOW = 8.0
# Сколько слов допустимо в одном блоке субтитров.
BLOCK_MIN_WORDS = 1
BLOCK_MAX_WORDS = 5


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


class CaptionBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    words: list[CaptionWord] = Field(min_length=1)  # 3-5 слов
    position: Literal["top", "center", "bottom", "auto"] = "auto"
    layout: Literal["stack", "ragged", "single_line"] = "stack"


class BrollAnim(str, Enum):
    POP_IN = "pop_in"
    SLIDE_UP = "slide_up"
    FULLSCREEN_TO_INSET = "fullscreen_to_inset"


class Broll(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word_idx: int  # к какому слову привязана
    duration: float = Field(ge=0.8, le=3.0)
    query: str  # англ. поисковый запрос
    prompt: str | None = None  # для генерации, если стока нет
    anim: BrollAnim
    anchor: Literal[
        "top_left", "top_right", "bottom_left", "bottom_right", "center", "fullscreen"
    ]
    # Заполняет Python после поиска/генерации, не LLM.
    asset: str | None = None


class CameraMove(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word_idx: int
    type: Literal["zoom_punch", "shake"]
    scale: float = Field(ge=1.0, le=1.25)


class EDL(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    source_duration: float
    words: list[Word]  # заполняет Python, не LLM
    cuts: list[Cut]
    captions: list[CaptionBlock]
    broll: list[Broll] = []
    camera: list[CameraMove] = []
    preset: str = "glass"

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
        for cw in block.words:
            if cw.word_idx not in known:
                errors.append(f"captions[{bi}]: word_idx={cw.word_idx} отсутствует в words")
            flat.append(cw.word_idx)

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
    cap = max(0, math.ceil(edl.out_duration / BROLL_WINDOW))
    if len(edl.broll) > cap:
        errors.append(f"broll: {len(edl.broll)} штук при потолке {cap} (1 на {BROLL_WINDOW:.0f} с)")
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

    # --- camera
    for i, cm in enumerate(edl.camera):
        if cm.word_idx not in known:
            errors.append(f"camera[{i}]: word_idx={cm.word_idx} отсутствует в words")

    return errors


def assert_valid(edl: EDL, *, strict_coverage: bool = True) -> None:
    errs = validate_edl(edl, strict_coverage=strict_coverage)
    if errs:
        raise EDLValidationError(errs)
