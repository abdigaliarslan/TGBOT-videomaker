"""M2: words.json → edl.json.

Разделение ответственности жёсткое:

- Python считает время: паузы, границы кусков, пересчёт таймлайна. Всё
  детерминированно и тестируемо.
- LLM решает смысл: где хук, что паразит, как разбить на блоки, что выделить.
  Возвращает только индексы слов (§0.1).
- Если LLM недоступна или дважды соврала — детерминированный режиссёр собирает
  EDL без акцентов и b-roll. Ролик всё равно соберётся (§3).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.core.config import Settings, settings as default_settings
from src.core.edl import (
    BLOCK_MAX_WORDS,
    EDL,
    CaptionBlock,
    CaptionWord,
    Cut,
    Emphasis,
    Word,
    validate_edl,
)
from src.llm.client import LLMBadOutput, LLMBudgetExceeded, LLMClient, LLMUnavailable
from src.llm.schemas import DIRECT_SCHEMA

log = logging.getLogger(__name__)

EPS = 1e-3
# Зазор вокруг выброшенного слова. Больше шага округления границ (3 знака),
# иначе после round() середина слова может снова оказаться внутри kept-куска.
DROP_MARGIN = 5e-3
SENTENCE_END = re.compile(r"[.!?…]+[\"»)]*$")

# Ярус-1: заминки, режутся всегда, даже без LLM.
HESITATIONS = {
    "ээ", "эээ", "эм", "эмм", "мм", "ммм", "аа", "ааа", "мгм", "эээм",
    "uh", "uhh", "um", "umm", "erm", "hmm", "mmm",
}


# --------------------------------------------------------------------------- нарезка


@dataclass(slots=True)
class Span:
    start: float
    end: float
    hook: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


def _normalize(text: str) -> str:
    return re.sub(r"[^\w']+", "", text.lower(), flags=re.UNICODE)


def hesitation_indices(words: Sequence[Word]) -> set[int]:
    """Заминки — их режем без всякой LLM."""
    return {w.idx for w in words if _normalize(w.text) in HESITATIONS}


def build_cuts(
    words: Sequence[Word],
    source_duration: float,
    *,
    drop: Iterable[int] = (),
    hook_end_word: int | None = None,
    cfg: Settings | None = None,
) -> list[Cut]:
    """Слова + индексы на выброс → непересекающийся отсортированный список cuts.

    Инвариант, на который опирается валидатор: середина КАЖДОГО оставленного
    слова лежит внутри kept-куска, а середина каждого выброшенного — снаружи.
    """
    cfg = cfg or default_settings
    drop_set = set(drop)
    kept = [w for w in words if w.idx not in drop_set]
    if not kept:
        return [Cut(src_in=0.0, src_out=source_duration, reason="keep")]

    dropped = [w for w in words if w.idx in drop_set]

    # 1. Стартовые окна вокруг каждого оставленного слова.
    #    Вместе с окном запоминаем «ядро» — от начала первого слова до конца
    #    последнего. Ядро трогать нельзя, паддинг вокруг него — можно.
    spans: list[Span] = []
    cores: list[tuple[float, float]] = []
    for i, w in enumerate(kept):
        s = w.start - cfg.cut_pad_before
        e = w.end + cfg.cut_pad_after
        merge = False
        if spans:
            prev = kept[i - 1]
            gap = w.start - prev.end
            # Не склеиваем через выброшенное слово и через длинную паузу.
            between_dropped = any(prev.end <= d.mid <= w.start for d in dropped)
            merge = gap <= cfg.cut_max_gap and not between_dropped
        if merge:
            spans[-1].end = e
            cores[-1] = (cores[-1][0], w.end)
        else:
            spans.append(Span(start=s, end=e))
            cores.append((w.start, w.end))

    # 2. Паддинг не должен проглатывать выброшенные слова и заезжать на соседей.
    #    Сравниваем с ЯДРОМ, а не с текущей границей: граница ещё поедет.
    for i, sp in enumerate(spans):
        core_start, core_end = cores[i]
        left_limit = 0.0 if i == 0 else cores[i - 1][1]
        right_limit = source_duration if i == len(spans) - 1 else cores[i + 1][0]
        for d in dropped:
            if d.mid < core_start:
                left_limit = max(left_limit, d.mid + DROP_MARGIN)
            elif d.mid > core_end:
                right_limit = min(right_limit, d.mid - DROP_MARGIN)
        sp.start = min(max(sp.start, left_limit, 0.0), core_start)
        sp.end = max(min(sp.end, right_limit, source_duration), core_end)

    # 2b. После клипа соседи могут пересечься паддингом — разводим по середине.
    for i in range(len(spans) - 1):
        if spans[i].end > spans[i + 1].start:
            mid = (cores[i][1] + cores[i + 1][0]) / 2.0
            spans[i].end = min(spans[i].end, max(cores[i][1], mid))
            spans[i + 1].start = max(spans[i + 1].start, min(cores[i + 1][0], mid))

    # 3. Схлопнувшиеся окна расширяем ровно до нужного минимума, но не за границы
    #    соседей — терять слово нельзя, оно уже в субтитрах.
    for i, sp in enumerate(spans):
        if sp.duration >= cfg.cut_min_segment:
            continue
        need = cfg.cut_min_segment - sp.duration
        core_start, core_end = cores[i]
        low = 0.0 if i == 0 else spans[i - 1].end
        high = source_duration if i == len(spans) - 1 else spans[i + 1].start
        for d in dropped:
            if d.mid < core_start:
                low = max(low, d.mid + DROP_MARGIN)
            elif d.mid > core_end:
                high = min(high, d.mid - DROP_MARGIN)
        grow_left = min(need / 2, max(0.0, sp.start - low))
        sp.start -= grow_left
        sp.end += min(need - grow_left, max(0.0, high - sp.end))

    spans = [sp for sp in spans if sp.duration > EPS]
    if not spans:
        return [Cut(src_in=0.0, src_out=source_duration, reason="keep")]

    # 4. Речи слишком мало — не режем вообще (§ MIN_SPEECH_RATIO).
    speech = sum(sp.duration for sp in spans)
    if source_duration > 0 and speech / source_duration < cfg.min_speech_ratio:
        log.info("речи %.0f%% от длительности — нарезку не применяем", 100 * speech / source_duration)
        return [Cut(src_in=0.0, src_out=source_duration, reason="keep")]

    # 5. Помечаем хук.
    if hook_end_word is not None:
        hook_word = next((w for w in kept if w.idx == hook_end_word), None)
        if hook_word is not None:
            for sp in spans:
                if sp.start <= hook_word.mid <= sp.end:
                    sp.hook = True
                    break
            else:
                spans[0].hook = True
        else:
            spans[0].hook = True

    # 6. Собираем полное покрытие: kept-куски + дырки между ними.
    cuts: list[Cut] = []
    cursor = 0.0
    for sp in spans:
        if sp.start - cursor > EPS:
            has_dropped = any(cursor <= d.mid <= sp.start for d in dropped)
            cuts.append(
                Cut(
                    src_in=round(cursor, 3),
                    src_out=round(sp.start, 3),
                    reason="drop_filler" if has_dropped else "drop_silence",
                )
            )
        cuts.append(
            Cut(
                src_in=round(sp.start, 3),
                src_out=round(sp.end, 3),
                reason="hook" if sp.hook else "keep",
            )
        )
        cursor = sp.end
    if source_duration - cursor > EPS:
        has_dropped = any(cursor <= d.mid <= source_duration for d in dropped)
        cuts.append(
            Cut(
                src_in=round(cursor, 3),
                src_out=round(source_duration, 3),
                reason="drop_filler" if has_dropped else "drop_silence",
            )
        )
    return cuts


# --------------------------------------------------------------------------- блоки


def split_blocks(
    words: Sequence[Word],
    kept_idx: Sequence[int],
    *,
    cfg: Settings | None = None,
) -> list[list[int]]:
    """Детерминированная разбивка на блоки субтитров.

    Рвём по: концу предложения, заметной паузе, лимиту слов, лимиту символов.
    Это же используется как фолбэк, если LLM недоступна.
    """
    cfg = cfg or default_settings
    by_idx = {w.idx: w for w in words}
    blocks: list[list[int]] = []
    current: list[int] = []
    chars = 0
    # Почему блок был закрыт: склеивать обратно можно только счётный разрыв.
    hard_break: list[bool] = []

    def close(hard: bool) -> None:
        nonlocal current, chars
        blocks.append(current)
        hard_break.append(hard)
        current, chars = [], 0

    for idx in kept_idx:
        w = by_idx[idx]
        wlen = len(w.text)
        if current:
            prev = by_idx[current[-1]]
            gap_break = (w.start - prev.end) > cfg.sub_gap_split
            char_break = chars + wlen + 1 > cfg.sub_max_chars
            count_break = len(current) >= cfg.sub_max_words
            if gap_break or char_break or count_break:
                close(hard=gap_break or char_break)

        current.append(idx)
        chars += wlen + 1

        if SENTENCE_END.search(w.text) and len(current) >= 2:
            close(hard=True)

    if current:
        blocks.append(current)
        hard_break.append(False)

    # Одинокое слово в хвосте выглядит как обрубок — приклеиваем к предыдущему
    # блоку, но только если разрыв был счётным и склейка не ломает лимиты.
    if len(blocks) >= 2 and len(blocks[-1]) == 1 and not hard_break[-2]:
        prev_block, tail = blocks[-2], blocks[-1]
        limit = min(cfg.sub_max_words + 1, BLOCK_MAX_WORDS)
        joined_chars = sum(len(by_idx[i].text) + 1 for i in (*prev_block, *tail))
        if len(prev_block) + 1 <= limit and joined_chars <= cfg.sub_max_chars * 1.35:
            blocks[-2].extend(blocks.pop())
    return blocks


# --------------------------------------------------------------------------- сборка EDL


def deterministic_edl(
    words: Sequence[Word],
    source_duration: float,
    *,
    preset: str | None = None,
    cut_pauses: bool = True,
    cfg: Settings | None = None,
) -> EDL:
    """Фолбэк-режиссёр: без акцентов и b-roll, но ролик собирается всегда (§3)."""
    cfg = cfg or default_settings
    drop = hesitation_indices(words) if cut_pauses else set()
    cuts = (
        build_cuts(words, source_duration, drop=drop, cfg=cfg)
        if cut_pauses
        else [Cut(src_in=0.0, src_out=source_duration, reason="keep")]
    )
    edl = EDL(
        source_duration=source_duration,
        words=list(words),
        cuts=cuts,
        captions=[],
        preset=preset or cfg.default_preset,
    )
    kept = [w.idx for w in words if _is_kept(w, cuts)]
    edl.captions = [
        CaptionBlock(words=[CaptionWord(word_idx=i) for i in group])
        for group in split_blocks(words, kept, cfg=cfg)
    ]
    return edl


def _is_kept(w: Word, cuts: Sequence[Cut]) -> bool:
    kept = [c for c in cuts if c.kept]
    if not kept:
        return True
    return any(c.src_in <= w.mid <= c.src_out for c in kept)


def edl_from_plan(
    words: Sequence[Word],
    source_duration: float,
    plan: dict[str, Any],
    *,
    preset: str | None = None,
    cfg: Settings | None = None,
) -> EDL:
    """Собирает EDL из ответа LLM. Всё, что связано со временем, считаем тут."""
    cfg = cfg or default_settings
    known = {w.idx for w in words}

    drop = {int(i) for i in plan.get("drop_words") or [] if int(i) in known}
    drop |= hesitation_indices(words)  # заминки режем в любом случае

    hook_end = plan.get("hook_end_word")
    hook_end = int(hook_end) if isinstance(hook_end, int) and hook_end in known else None

    cuts = build_cuts(words, source_duration, drop=drop, hook_end_word=hook_end, cfg=cfg)

    captions: list[CaptionBlock] = []
    for raw in plan.get("blocks") or []:
        idxs = [int(i) for i in raw.get("words") or [] if int(i) in known and int(i) not in drop]
        if not idxs:
            continue
        accent_word = raw.get("accent_word")
        accent_kind = raw.get("accent")
        emphasis_map: dict[int, Emphasis] = {}
        if isinstance(accent_word, int) and accent_word in idxs and accent_kind in ("accent", "shout"):
            emphasis_map[accent_word] = Emphasis(accent_kind)
        captions.append(
            CaptionBlock(
                words=[
                    CaptionWord(word_idx=i, emphasis=emphasis_map.get(i, Emphasis.NONE))
                    for i in idxs
                ],
                position=raw.get("position") or "auto",
            )
        )

    return EDL(
        source_duration=source_duration,
        words=list(words),
        cuts=cuts,
        captions=captions,
        preset=preset or cfg.default_preset,
    )


# --------------------------------------------------------------------------- вход


def words_table(words: Sequence[Word], *, gap_threshold: float = 0.35) -> str:
    """Транскрипт для промпта: индекс, слово, заметная пауза после него."""
    lines: list[str] = []
    for i, w in enumerate(words):
        gap = ""
        if i + 1 < len(words):
            d = words[i + 1].start - w.end
            if d >= gap_threshold:
                gap = f"   ‹пауза {d:.1f}с›"
        lines.append(f"{w.idx:4d} {w.text}{gap}")
    return "\n".join(lines)


async def clean_transcript(
    words: Sequence[Word],
    *,
    llm: LLMClient | None,
    language: str = "",
    cfg: Settings | None = None,
) -> tuple[list[Word], int]:
    """Точечные правки распознавания по индексу (§4, тир `cheap`).

    Меняется ТОЛЬКО текст слова. Тайминги трогать нельзя: они из ASR, и это
    единственный источник правды о времени.
    """
    cfg = cfg or default_settings
    out = [w.model_copy() for w in words]
    if llm is None or not llm.ready or not out:
        return out, 0

    from src.llm.schemas import CLEAN_SCHEMA

    try:
        result = await llm.complete(
            "clean_transcript",
            {"language": language or cfg.whisper_language or "auto", "words_table": words_table(words)},
            CLEAN_SCHEMA,
            "cheap",
            max_tokens=8000,
        )
    except (LLMUnavailable, LLMBadOutput) as e:
        log.warning("правка транскрипта пропущена: %s", e)
        return out, 0

    applied = 0
    for fix in result.data.get("fixes") or []:
        i = fix.get("i")
        text = (fix.get("fix") or "").strip()
        if not isinstance(i, int) or not (0 <= i < len(out)) or not text:
            continue
        # Замена одного слова одним словом: пробелы внутри правки сломали бы
        # соответствие индексов, на котором стоит весь EDL.
        if " " in text:
            continue
        if text == out[i].text:
            continue
        log.debug("правка %d: %r → %r", i, out[i].text, text)
        out[i] = out[i].model_copy(update={"text": text})
        applied += 1

    if applied:
        log.info("транскрипт: применено правок %d", applied)
    return out, applied


async def make_metadata(
    text: str,
    duration: float,
    *,
    llm: LLMClient | None,
    language: str = "",
    cfg: Settings | None = None,
) -> dict[str, Any] | None:
    """Заголовок, описание, хештеги (§4, тир `cheap`). Необязательный шаг."""
    cfg = cfg or default_settings
    if llm is None or not llm.ready or not text.strip():
        return None

    from src.llm.schemas import METADATA_SCHEMA

    try:
        result = await llm.complete(
            "metadata",
            {
                "language": language or cfg.whisper_language or "auto",
                "duration": f"{duration:.0f}",
                "transcript": text,
            },
            METADATA_SCHEMA,
            "cheap",
            max_tokens=6000,
        )
    except (LLMUnavailable, LLMBadOutput) as e:
        log.warning("метаданные не сгенерированы: %s", e)
        return None
    return result.data


async def direct(
    words: Sequence[Word],
    source_duration: float,
    *,
    llm: LLMClient | None = None,
    language: str = "",
    preset: str | None = None,
    cfg: Settings | None = None,
) -> EDL:
    """Главный вход M2: транскрипт → EDL.

    Порядок: вызов LLM → валидация → один ретрай с текстом ошибки → фолбэк (§3).
    """
    cfg = cfg or default_settings
    if not words:
        raise ValueError("пустой транскрипт: нечего монтировать")

    if llm is None or not llm.ready:
        log.info("режиссёр: LLM недоступна, детерминированный монтаж")
        edl = deterministic_edl(words, source_duration, preset=preset, cfg=cfg)
        _assert_or_die(edl)
        return edl

    variables = {
        "language": language or cfg.whisper_language or "auto",
        "duration": f"{source_duration:.1f}",
        "word_count": str(len(words)),
        "max_words_per_block": str(cfg.sub_max_words + 1),
        "words_table": words_table(words),
    }

    correction: str | None = None
    for attempt in (1, 2):
        try:
            result = await llm.complete(
                "direct",
                variables,
                DIRECT_SCHEMA,
                "smart",
                correction=correction,
                effort="high",
            )
        except LLMUnavailable as e:
            # Провайдер недоступен — повторять нечего, сразу фолбэк.
            log.warning("режиссёр: LLM недоступна (%s), фолбэк", e)
            break
        except LLMBudgetExceeded as e:
            # Повтор с теми же настройками даст тот же обрыв и вторую оплату.
            log.warning("режиссёр: %s — фолбэк без повтора", e)
            break
        except LLMBadOutput as e:
            # Ответ не той формы. Это ровно тот случай, ради которого §4
            # разрешает один ретрай — тратим его так же, как на семантику.
            log.warning("режиссёр: ответ не по схеме (попытка %d): %s", attempt, e)
            if attempt == 2:
                break
            correction = str(e)
            continue

        edl = edl_from_plan(words, source_duration, result.data, preset=preset, cfg=cfg)
        errors = validate_edl(edl)
        if not errors:
            log.info(
                "режиссёр: EDL от LLM принят с попытки %d (блоков=%d, вырезано кусков=%d)",
                attempt,
                len(edl.captions),
                sum(1 for c in edl.cuts if not c.kept),
            )
            return edl

        log.warning("режиссёр: EDL не прошёл валидацию (попытка %d): %s", attempt, "; ".join(errors))
        if attempt == 2:
            break
        correction = "\n".join(f"- {e}" for e in errors[:10])

    log.info("режиссёр: фолбэк на детерминированный EDL")
    edl = deterministic_edl(words, source_duration, preset=preset, cfg=cfg)
    _assert_or_die(edl)
    return edl


def _assert_or_die(edl: EDL) -> None:
    errors = validate_edl(edl)
    if errors:
        # Детерминированный путь обязан быть валидным — это баг в нас, не в модели.
        raise AssertionError("детерминированный EDL невалиден: " + "; ".join(errors))


def broll_cap(out_duration: float) -> int:
    return max(0, math.ceil(out_duration / 8.0))


if __name__ == "__main__":  # python -m src.pipeline.director words.json > edl.json
    import asyncio
    import json
    import sys

    from src.core.logging import setup_logging
    from src.pipeline.asr import words_from_json

    setup_logging(default_settings.log_level)
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
    ws = words_from_json(payload)
    dur = float(payload.get("duration") or (ws[-1].end if ws else 0.0))
    client = LLMClient()
    out = asyncio.run(direct(ws, dur, llm=client, language=payload.get("language", "")))
    print(out.model_dump_json(indent=2))
