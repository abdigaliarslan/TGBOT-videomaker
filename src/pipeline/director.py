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

import itertools
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from src.core.config import Settings, settings as default_settings
from src.core.edl import (
    ANCHOR_TRACK_MAX,
    BLOCK_MAX_WORDS,
    EDL,
    AnchorTrack,
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

# Крупность (Ф6). Смена происходит на резе и только на нём, поэтому единственное,
# что здесь нужно решить, — на каком именно резе.
#
# FRAMING_MIN_HOLD: сколько секунд крупность обязана простоять, прежде чем её
# можно менять. Русская речь даёт рез каждые 2-4 секунды; без выдержки крупность
# скакала бы почти на каждом, и монтаж превратился бы в мигание.
#
# Величина связана с `crop.MIN_SECONDS_PER_FRAMING` и раньше ей противоречила.
# Выдержка 2.2 с давала смену почти на каждом резе: замер на смоделированной
# нарезке — 4 смены на 20 секундах, 13 на шестидесяти, 41 на трёх минутах. Всё,
# что длиннее полуминуты, упиралось в потолок кусков и МОЛЧА схлопывалось в одно
# окно на весь ролик, то есть Ф6 выключалась ровно там, где ролик длинный.
# 4.5 с — это не вкус, а согласование: кроп считает мельтешением чаще одной смены
# на 4 секунды, и план режиссёра обязан в это влезать по построению. Считается это
# по ИНТЕРВАЛАМ, а не по кускам: n крупностей занимают n−1 интервал, поэтому
# потолок в кропе и написан как `covered / 4.0 + 1`. Раньше «+1» там не было, и на
# роликах 10-20 секунд план режиссёра пробивал собственный потолок кропа.
FRAMING_MIN_HOLD = 4.5
# FRAMING_MIN_SEGMENT: короткому куску своя крупность не достаётся — зритель
# успеет заметить только скачок, но не сам план. Это касается и хука: хук,
# который короче, чем нужно, чтобы его прочитали, даёт не крупный план, а рывок.
FRAMING_MIN_SEGMENT = 0.8

# Ярус-1: заминки, режутся всегда, даже без LLM.
HESITATIONS = {
    "ээ", "эээ", "эм", "эмм", "мм", "ммм", "аа", "ааа", "мгм", "эээм",
    "ии", "иии", "ыы", "ыыы", "ээм", "ааэ",
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


def hesitation_indices(words: Sequence[Word], cfg: Settings | None = None) -> set[int]:
    """Заминки — их режем без всякой LLM, но только ЗАТЯЖНЫЕ.

    Каждое выброшенное слово — отдельный рез: спаны через выброс не склеиваются
    (`build_cuts`, `between_dropped`). Значит короткое «эм» на 0.15 с стоит
    зрителю видимого скачка, а экономит полкадра. Порог `cut_min_filler`
    оставляет только те заминки, которые слышны как затяжка.
    """
    cfg = cfg or default_settings
    floor = cfg.cut_min_filler
    return {
        w.idx
        for w in words
        if _normalize(w.text) in HESITATIONS and (w.end - w.start) >= floor
    }


def assign_framing(cuts: Sequence[Cut]) -> None:
    """Проставляет крупность оставленным кускам. На месте, по готовым резам.

    LLM здесь не участвует и в схему `framing` не входит: чередование планов —
    это ремесло монтажёра, полностью выводимое из уже принятых решений о резах.
    Модели тут нечего добавить, а возможность ошибиться — есть.

    Правило: хук снимаем крупно (это единственный кусок, который обязан удержать
    зрителя), дальше крупность чередуется, но не чаще FRAMING_MIN_HOLD секунд и
    только на кусках длиннее FRAMING_MIN_SEGMENT. Выброшенные куски остаются со
    своим значением по умолчанию: их никто не увидит.

    Порог длины действует и на хук. Хук в полсекунды — это не крупный план, а
    вспышка: зритель увидит только скачок кадра, а сам план прочитать не успеет,
    и лучше отдать такой кусок общей крупности.

    Выдержка действует и на ПЕРВУЮ крупность. Хук не обязан быть первым kept-куском
    (его выбирает `hook_end_word`), и тогда всё, что было до него, оставалось со
    значением по умолчанию: ролик открывался планом на 0.4 секунды и сразу менял
    крупность — та же вспышка, только в начале. Заодно это единственный случай,
    когда план режиссёра нарушал арифметику потолка кропа
    (`crop.MIN_SECONDS_PER_FRAMING`): все интервалы, кроме первого, по построению
    не короче FRAMING_MIN_HOLD.
    """
    current = "wide"
    hold = 0.0
    kept = sorted((c for c in cuts if c.kept), key=lambda c: c.src_in)
    for cut in kept:
        if cut.reason == "hook" and cut.duration >= FRAMING_MIN_SEGMENT:
            current, hold = "tight", 0.0
        elif hold >= FRAMING_MIN_HOLD and cut.duration >= FRAMING_MIN_SEGMENT:
            current, hold = ("wide" if current == "tight" else "tight"), 0.0
        cut.framing = current  # type: ignore[assignment]
        hold += cut.duration

    # Первая крупность короче выдержки — отдаём эти куски следующей: короткий
    # «разгон» перед хуком не читается как план, только как рывок.
    head = list(itertools.takewhile(lambda c: c.framing == kept[0].framing, kept)) if kept else []
    if 0 < len(head) < len(kept) and sum(c.duration for c in head) < FRAMING_MIN_HOLD:
        nxt = kept[len(head)].framing
        for cut in head:
            cut.framing = nxt


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
    # Крупность — по готовым резам (Ф6). Вырожденные выходы выше отдают один
    # kept-кусок без хука, и `wide` по умолчанию для них уже верно.
    assign_framing(cuts)
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
    """Фолбэк-режиссёр: без b-roll и без разбора смысла, но ролик собирается всегда (§3).

    Выделение здесь всё же расставляется — детерминированно (`lead_word`). Это
    не разбор смысла, а защита от стены ровного крупного текста: после С1 кегль
    внутри блока стал иерархией, и блок без выделенного слова читается хуже,
    чем блок с приблизительно выбранным.
    """
    cfg = cfg or default_settings
    drop = hesitation_indices(words, cfg) if cut_pauses else set()
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
    by_idx = {w.idx: w for w in words}
    blocks: list[CaptionBlock] = []
    for group in split_blocks(words, kept, cfg=cfg):
        # Иерархию кегля держим и здесь: без неё запасной режиссёр отдаёт стену
        # ровного крупного текста, а это заметнее, чем отсутствие b-roll.
        lead = lead_word(group, by_idx)
        blocks.append(
            CaptionBlock(
                words=[
                    CaptionWord(
                        word_idx=i,
                        emphasis=Emphasis.ACCENT if i == lead else Emphasis.NONE,
                    )
                    for i in group
                ]
            )
        )
    edl.captions = blocks
    return edl


def cover_missing(
    blocks: list[CaptionBlock], kept: Sequence[int], by_idx: Mapping[int, Word]
) -> list[CaptionBlock]:
    """Досыпать в субтитры слова, которые модель исключила, а мы оставили.

    Модель строит блоки и список на выброс ВМЕСТЕ: раз слово выброшено, в блок
    она его не кладёт. Её выброс мы больше не принимаем (режем только затяжные
    заминки), и такое слово осталось бы звучать без субтитра — валидатор ловит
    это как «слова не попали ни в один блок».

    Слово подсаживается к соседу по индексу: после `i-1`, иначе перед `i+1`,
    иначе отдельным блоком. Переполненный блок режется на куски, и каждому
    куску, оставшемуся без выделения, `lead_word` возвращает главное слово —
    иначе после разреза половина блока вышла бы стеной ровного текста (С3).
    """
    present = {cw.word_idx for b in blocks for cw in b.words}
    for i in [i for i in kept if i not in present]:
        placed = False
        for b in blocks:
            idxs = [cw.word_idx for cw in b.words]
            if i - 1 in idxs:
                b.words.insert(idxs.index(i - 1) + 1, CaptionWord(word_idx=i))
                placed = True
                break
            if i + 1 in idxs:
                b.words.insert(idxs.index(i + 1), CaptionWord(word_idx=i))
                placed = True
                break
        if not placed:
            blocks.append(CaptionBlock(words=[CaptionWord(word_idx=i)]))

    blocks.sort(key=lambda b: b.words[0].word_idx if b.words else 0)

    out: list[CaptionBlock] = []
    for b in blocks:
        if len(b.words) <= BLOCK_MAX_WORDS:
            out.append(b)
            continue
        for k in range(0, len(b.words), BLOCK_MAX_WORDS):
            out.append(
                CaptionBlock(
                    words=b.words[k : k + BLOCK_MAX_WORDS],
                    position=b.position,
                    anchor_track=b.anchor_track,
                )
            )
    for b in out:
        if any(cw.emphasis != Emphasis.NONE for cw in b.words):
            continue
        lead = lead_word([cw.word_idx for cw in b.words], by_idx)
        for cw in b.words:
            if cw.word_idx == lead:
                cw.emphasis = Emphasis.ACCENT
    return out


def lead_word(idxs: Sequence[int], by_idx: Mapping[int, Word]) -> int | None:
    """Какое слово поднять, если режиссёр не назвал ни одного (С3).

    После С1 кегль внутри блока — это иерархия: выделенное слово набирается в
    2.3 раза крупнее соседей. Блок, в котором не выделено НИЧЕГО, выходит ровной
    стеной крупного текста — замерено на эталонном плане, где таких блоков 17 %
    (2 из 12), и на кадре это «Я думаю что» тремя словами по 185 px.

    Выбор — самое длинное слово, при равенстве самое раннее. Это не разбор
    смысла, а его дешёвый заменитель: служебные слова русского языка коротки
    («я», «что», «в», «и»), а смысловое обычно длиннее. На эталонном плане
    правило даёт «думаю» из «я думаю что» и «истину» из «одну истину».

    Выбор кладётся В EDL, а не считается при рендере: EDL остаётся единственным
    источником правды (§0), сохранённый EDL пересобирается кадр в кадр
    (инвариант 4), и режиссёр волен его переписать.
    """
    best: tuple[int, int] | None = None
    for i in idxs:
        w = by_idx.get(i)
        if w is None:
            continue
        n = len(w.text.strip())
        if best is None or n > best[0]:
            best = (n, i)
    return best[1] if best else None


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
    by_idx = {w.idx: w for w in words}

    # Что вырезать — решает НЕ модель. Раньше ей было позволено выбросить до
    # четверти слов (паразиты, повторы, самоисправления), а каждое выброшенное
    # слово даёт отдельный рез: спаны через выброс не склеиваются, и ролик
    # выходил рваным до неразборчивости. Теперь режутся только затяжные
    # заминки, а список модели сужается до них же.
    drop = hesitation_indices(words, cfg)
    asked = {int(i) for i in plan.get("drop_words") or [] if int(i) in known}
    ignored = asked - drop
    if ignored:
        log.info(
            "режиссёр: %d слов на выброс не приняты — режем только затяжные заминки",
            len(ignored),
        )

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
        else:
            # Режиссёр не назвал слово — поднимаем сами (С3, см. `lead_word`).
            fallback = lead_word(idxs, by_idx)
            if fallback is not None:
                emphasis_map[fallback] = Emphasis.ACCENT
        captions.append(
            CaptionBlock(
                words=[
                    CaptionWord(word_idx=i, emphasis=emphasis_map.get(i, Emphasis.NONE))
                    for i in idxs
                ],
                position=raw.get("position") or "auto",
                # Модель отвечает булевым флагом, а не структурой: НУЖЕН ли приём
                # здесь — вопрос смысла и её работа, а насколько выше головы
                # висит текст — ремесло, и оно живёт в значениях по умолчанию
                # (§0). Заодно модели нечего сломать: диапазонов, которые она
                # может нарушить, у неё просто нет.
                anchor_track=AnchorTrack() if raw.get("follow_face") else None,
            )
        )

    # Слова, которые модель исключила из блоков ради СВОЕГО выброса, а мы
    # оставили в кадре. Без этого они звучали бы без субтитра.
    #
    # Именно эти, а не все пропущенные: слово, которое модель просто забыла, —
    # это битый ответ, и его чинит ретрай (`direct`), а не мы. Молча дописав
    # забытое, мы бы выключили единственную проверку, которая ловит модель на
    # потере текста.
    restored = sorted(i for i in ignored if _is_kept(by_idx[i], cuts))
    if restored:
        captions = cover_missing(captions, restored, by_idx)

    # Потолок держим здесь, а не ретраем валидатора: лишняя привязка — не
    # ошибка формы, а перебор с приёмом, и снять её дешевле, чем гонять модель
    # по кругу. Остаются ПЕРВЫЕ: у ролика один смысловой пик, и он в начале
    # чаще, чем в конце.
    anchored = [b for b in captions if b.anchor_track is not None]
    for extra in anchored[ANCHOR_TRACK_MAX:]:
        extra.anchor_track = None
    if len(anchored) > ANCHOR_TRACK_MAX:
        log.info(
            "привязка к голове: модель попросила %d блоков, оставлено %d",
            len(anchored), ANCHOR_TRACK_MAX,
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
