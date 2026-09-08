"""Тесты детерминированной части режиссёра: нарезка и разбивка на блоки.

Инвариант, который здесь проверяется главным: EDL, собранный из любого разумного
плана, проходит семантический валидатор. Если он не проходит — виноват не LLM,
а мы.
"""

from __future__ import annotations

import pytest

from src.core.config import Settings
from src.core.edl import Cut, Word, validate_edl
from src.pipeline.director import (
    FRAMING_MIN_HOLD,
    FRAMING_MIN_SEGMENT,
    assign_framing,
    build_cuts,
    deterministic_edl,
    edl_from_plan,
    hesitation_indices,
    split_blocks,
    words_table,
)


def cfg(**kw) -> Settings:
    base = dict(
        cut_max_gap=0.8,
        cut_pad_before=0.15,
        cut_pad_after=0.25,
        cut_min_segment=0.3,
        min_speech_ratio=0.05,
        sub_max_words=4,
        sub_max_chars=22,
        sub_gap_split=0.6,
    )
    base.update(kw)
    return Settings(**base)


def mk(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(idx=i, text=t, start=s, end=e) for i, (t, s, e) in enumerate(spec)]


# ------------------------------------------------------------------ нарезка пауз


def test_long_pause_is_cut():
    ws = mk([("раз", 0.5, 0.9), ("два", 1.1, 1.5), ("три", 5.0, 5.4), ("четыре", 5.6, 6.0)])
    cuts = build_cuts(ws, 7.0, cfg=cfg())
    kept = [c for c in cuts if c.kept]
    dropped = [c for c in cuts if not c.kept]
    assert len(kept) == 2, cuts
    assert any(c.reason == "drop_silence" for c in dropped)
    # покрытие исходника полное и без пересечений
    assert cuts[0].src_in == pytest.approx(0.0)
    assert cuts[-1].src_out == pytest.approx(7.0)
    for a, b in zip(cuts, cuts[1:]):
        assert a.src_out == pytest.approx(b.src_in)


def test_short_pause_is_kept():
    ws = mk([("раз", 0.5, 0.9), ("два", 1.3, 1.7)])  # пауза 0.4 < 0.8
    cuts = build_cuts(ws, 3.0, cfg=cfg())
    assert len([c for c in cuts if c.kept]) == 1


def test_leading_and_trailing_silence_dropped():
    ws = mk([("слово", 4.0, 4.5)])
    cuts = build_cuts(ws, 10.0, cfg=cfg())
    assert cuts[0].reason == "drop_silence" and cuts[0].src_in == 0.0
    assert cuts[-1].reason == "drop_silence" and cuts[-1].src_out == pytest.approx(10.0)


def test_dropped_word_is_excluded_from_kept_region():
    ws = mk([("я", 0.5, 0.8), ("ээ", 0.85, 1.05), ("сделал", 1.1, 1.6)])
    cuts = build_cuts(ws, 3.0, drop=[1], cfg=cfg())
    kept = [c for c in cuts if c.kept]
    mid = ws[1].mid
    assert not any(c.src_in <= mid <= c.src_out for c in kept), cuts
    # оставленные слова обязаны остаться внутри
    for i in (0, 2):
        assert any(c.src_in <= ws[i].mid <= c.src_out for c in kept)


def test_dropped_word_marks_drop_filler():
    ws = mk([("я", 0.5, 0.8), ("ээ", 0.85, 1.05), ("сделал", 1.1, 1.6)])
    cuts = build_cuts(ws, 3.0, drop=[1], cfg=cfg())
    assert any(c.reason == "drop_filler" for c in cuts)


def test_no_speech_means_no_cutting():
    ws = mk([("а", 0.1, 0.15)])
    cuts = build_cuts(ws, 60.0, cfg=cfg())
    assert len(cuts) == 1 and cuts[0].reason == "keep"


def test_hook_marks_first_segment():
    ws = mk([("раз", 0.5, 0.9), ("два", 1.1, 1.5), ("три", 5.0, 5.4)])
    cuts = build_cuts(ws, 7.0, hook_end_word=1, cfg=cfg())
    assert cuts[1].reason == "hook"  # cuts[0] — тишина в начале
    assert [c.reason for c in cuts].count("hook") == 1


def test_cuts_never_exceed_source():
    ws = mk([("конец", 9.6, 10.0)])
    cuts = build_cuts(ws, 10.0, cfg=cfg())
    assert cuts[-1].src_out <= 10.0 + 1e-6


def test_hesitation_detection():
    ws = mk([("ну", 0.0, 0.2), ("ээ", 0.3, 0.5), ("Мм,", 0.6, 0.8), ("работа", 1.0, 1.5)])
    assert hesitation_indices(ws) == {1, 2}  # «ну» — ярус-2, режется только по решению LLM


# ------------------------------------------------------------------ блоки


def test_blocks_respect_word_limit():
    ws = mk([(f"сл{i}", i * 0.3, i * 0.3 + 0.25) for i in range(12)])
    blocks = split_blocks(ws, [w.idx for w in ws], cfg=cfg(sub_max_words=3, sub_max_chars=99))
    assert all(len(b) <= 3 for b in blocks)
    assert [i for b in blocks for i in b] == list(range(12))


def test_blocks_break_on_char_limit():
    ws = mk([("длинноеслово", 0.0, 0.4), ("ещёодноогромное", 0.5, 0.9)])
    blocks = split_blocks(ws, [0, 1], cfg=cfg(sub_max_words=5, sub_max_chars=14))
    assert len(blocks) == 2


def test_blocks_break_on_gap():
    ws = mk([("раз", 0.0, 0.3), ("два", 0.4, 0.7), ("три", 2.0, 2.3)])
    blocks = split_blocks(ws, [0, 1, 2], cfg=cfg(sub_max_words=5, sub_max_chars=99, sub_gap_split=0.6))
    assert blocks == [[0, 1], [2]]


def test_blocks_break_on_sentence_end():
    ws = mk([("это", 0.0, 0.3), ("конец.", 0.35, 0.7), ("новое", 0.75, 1.1), ("начало", 1.15, 1.5)])
    blocks = split_blocks(ws, [0, 1, 2, 3], cfg=cfg(sub_max_words=5, sub_max_chars=99))
    assert blocks[0] == [0, 1]


def test_lonely_tail_word_is_glued():
    ws = mk([(f"сл{i}", i * 0.3, i * 0.3 + 0.25) for i in range(5)])
    blocks = split_blocks(ws, [w.idx for w in ws], cfg=cfg(sub_max_words=4, sub_max_chars=99))
    assert blocks[-1] != [4]
    assert [i for b in blocks for i in b] == [0, 1, 2, 3, 4]


# ------------------------------------------------------------------ сборка EDL


def test_deterministic_edl_is_always_valid():
    ws = mk(
        [("привет", 0.5, 1.0), ("это", 1.1, 1.3), ("ээ", 1.35, 1.55), ("тест", 4.0, 4.5),
         ("монтажа", 4.6, 5.2), ("бота.", 5.3, 5.9)]
    )
    edl = deterministic_edl(ws, 8.0, cfg=cfg())
    assert validate_edl(edl) == []
    captioned = [cw.word_idx for b in edl.captions for cw in b.words]
    assert 2 not in captioned  # «ээ» вырезано
    assert captioned == [0, 1, 3, 4, 5]


def test_edl_from_plan_is_valid():
    ws = mk([(f"сл{i}", i * 0.4, i * 0.4 + 0.3) for i in range(9)])
    plan = {
        "hook_end_word": 2,
        "drop_words": [4],
        "blocks": [
            {"words": [0, 1, 2], "accent_word": 1, "accent": "accent", "position": "auto"},
            {"words": [3, 5], "accent_word": None, "accent": None, "position": "auto"},
            {"words": [6, 7, 8], "accent_word": 8, "accent": "shout", "position": "auto"},
        ],
    }
    edl = edl_from_plan(ws, 5.0, plan, cfg=cfg())
    assert validate_edl(edl) == []
    assert [cw.word_idx for b in edl.captions for cw in b.words] == [0, 1, 2, 3, 5, 6, 7, 8]


def test_edl_from_plan_drops_unknown_indices():
    ws = mk([(f"сл{i}", i * 0.4, i * 0.4 + 0.3) for i in range(4)])
    plan = {
        "hook_end_word": 99,
        "drop_words": [77],
        "blocks": [{"words": [0, 1, 99], "accent_word": 99, "accent": "shout", "position": "auto"},
                   {"words": [2, 3], "accent_word": None, "accent": None, "position": "auto"}],
    }
    edl = edl_from_plan(ws, 3.0, plan, cfg=cfg())
    assert validate_edl(edl) == []


def test_edl_from_plan_ignores_accent_outside_block():
    ws = mk([(f"сл{i}", i * 0.4, i * 0.4 + 0.3) for i in range(4)])
    plan = {
        "hook_end_word": None,
        "drop_words": [],
        "blocks": [{"words": [0, 1], "accent_word": 3, "accent": "shout", "position": "auto"},
                   {"words": [2, 3], "accent_word": None, "accent": None, "position": "auto"}],
    }
    edl = edl_from_plan(ws, 3.0, plan, cfg=cfg())
    assert validate_edl(edl) == []
    assert all(cw.emphasis.value == "none" for b in edl.captions for cw in b.words)


def test_words_table_marks_pauses():
    ws = mk([("раз", 0.0, 0.4), ("два", 1.5, 1.9)])
    table = words_table(ws)
    assert "пауза 1.1" in table
    assert table.splitlines()[1].strip().startswith("1 два")


@pytest.mark.parametrize("n", [1, 2, 3, 7, 25, 60])
def test_deterministic_edl_valid_for_many_shapes(n):
    ws = mk([(f"w{i}", i * 0.45, i * 0.45 + 0.35) for i in range(n)])
    edl = deterministic_edl(ws, n * 0.45 + 1.0, cfg=cfg())
    assert validate_edl(edl) == []


# ------------------------------------------------------------------ крупность (Ф6)


def cuts_of(spec: list[tuple[float, float, str]]) -> list[Cut]:
    return [Cut(src_in=a, src_out=b, reason=r) for a, b, r in spec]


def test_hook_is_shot_tight():
    """Хук — единственный кусок, который обязан удержать зрителя."""
    cuts = cuts_of([(0.0, 0.2, "drop_silence"), (0.2, 5.0, "hook"), (5.0, 11.0, "keep")])
    assign_framing(cuts)
    assert cuts[1].framing == "tight"


def chain(durations: list[float], first: str = "hook") -> list[Cut]:
    """Череда встык идущих kept-кусков заданной длительности."""
    out, t = [], 0.0
    for i, d in enumerate(durations):
        out.append(Cut(src_in=t, src_out=t + d, reason=first if i == 0 else "keep"))
        t += d
    return out


def test_framing_alternates_on_long_cuts():
    """Куски длиннее выдержки чередуют крупность — иначе движения в ролике нет."""
    long = FRAMING_MIN_HOLD + 1.0
    cuts = chain([long, long, long])
    assign_framing(cuts)
    assert [c.framing for c in cuts] == ["tight", "wide", "tight"]


def test_framing_holds_through_short_cuts():
    """Череда коротких кусков не должна давать мигание крупности."""
    cuts = cuts_of([(i * 0.5, i * 0.5 + 0.5, "keep") for i in range(10)])
    assign_framing(cuts)
    assert len({c.framing for c in cuts}) == 1


def test_framing_waits_for_min_hold():
    """Смена не раньше FRAMING_MIN_HOLD секунд после предыдущей.

    Длительности выводятся из самой константы, а не вписаны числами: тест обязан
    проверять правило, а не то значение, с которым его когда-то написали.
    """
    half = FRAMING_MIN_HOLD * 0.6  # два таких куска перекрывают выдержку, один — нет
    cuts = chain([half, half, FRAMING_MIN_HOLD + 2.0], first="keep")
    assign_framing(cuts)
    changes = [i for i in range(1, len(cuts)) if cuts[i].framing != cuts[i - 1].framing]
    assert changes == [2]
    assert cuts[0].duration < FRAMING_MIN_HOLD <= sum(c.duration for c in cuts[:2])


def test_framing_ignores_dropped_cuts():
    """Вырезанный кусок не участвует ни в выдержке, ни в чередовании."""
    hold = FRAMING_MIN_HOLD + 1.0
    cuts = [
        Cut(src_in=0.0, src_out=hold, reason="keep"),
        # Длинная дырка: если бы выдержка считалась по исходному времени, её
        # хватило бы на смену крупности сама по себе.
        Cut(src_in=hold, src_out=hold + 20.0, reason="drop_silence"),
        Cut(src_in=hold + 20.0, src_out=hold + 20.0 + hold, reason="keep"),
    ]
    assign_framing(cuts)
    assert cuts[1].framing == "wide"  # значение по умолчанию, его никто не увидит
    assert cuts[0].framing != cuts[2].framing


def test_short_hook_is_not_shot_tight():
    """Хук короче порога — это вспышка, а не крупный план: крупность не трогаем."""
    cuts = chain([FRAMING_MIN_SEGMENT / 2, FRAMING_MIN_HOLD + 2.0])
    assign_framing(cuts)
    assert cuts[0].framing == "wide"


def test_framing_density_fits_the_crop_ceiling():
    """План режиссёра обязан влезать в потолок мельтешения, который держит кроп.

    Тест-предшественник проходил из-за выбора фикстуры: длины кусков брались из
    `uniform(1.5, 6.0)` при ОДНОМ зерне и только на 20/60/180 секундах, тогда как
    `cut_min_segment` в конфиге равен 0.3, а основной формат — ролик на 10-20
    секунд. Именно там инвариант и ломался: нарезка [4.8, 4.77, 2.11] даёт три
    крупности на 11.68 с, где старый потолок разрешал две, и Ф6 схлопывалась в одно
    окно. Поэтому здесь свип: сотни нарезок с длинами от `cut_min_segment` и
    длительности от 8 секунд, а хук стоит не только первым куском.
    """
    import random

    from src.pipeline.crop import MIN_FRAMING_SEGMENTS, MIN_SECONDS_PER_FRAMING

    settings = cfg()
    for seed in range(300):
        random.seed(seed)
        total = random.uniform(8.0, 120.0)
        durations, t = [], 0.0
        while t < total:
            d = random.uniform(settings.cut_min_segment, 6.0)
            durations.append(d)
            t += d
        cuts = chain(durations)
        hook_at = seed % 4  # хук не обязан быть первым куском: его задаёт LLM
        for i, cut in enumerate(cuts):
            cut.reason = "hook" if i == min(hook_at, len(cuts) - 1) else "keep"
        assign_framing(cuts)
        segments = 1 + sum(1 for a, b in zip(cuts, cuts[1:]) if a.framing != b.framing)
        # n крупностей занимают n−1 интервал — потолок кропа считает так же.
        allowed = max(MIN_FRAMING_SEGMENTS, int(t / MIN_SECONDS_PER_FRAMING) + 1)
        assert segments <= allowed, (seed, round(t, 2), segments, allowed, durations)


def test_framing_density_holds_on_the_known_counterexample():
    """Точный контрпример к прежнему согласованию — три крупности на 11.68 с."""
    from src.pipeline.crop import MIN_FRAMING_SEGMENTS, MIN_SECONDS_PER_FRAMING

    cuts = chain([4.8, 4.77, 2.11])
    assign_framing(cuts)
    segments = 1 + sum(1 for a, b in zip(cuts, cuts[1:]) if a.framing != b.framing)
    covered = sum(c.duration for c in cuts)
    assert segments == 3
    assert segments <= max(MIN_FRAMING_SEGMENTS, int(covered / MIN_SECONDS_PER_FRAMING) + 1)


def test_first_framing_holds_as_long_as_any_other():
    """Ролик не открывается вспышкой: короткий разгон перед хуком отдан хуку.

    Хук выбирает LLM (`hook_end_word`), и он не обязан попасть в первый kept-кусок.
    Тогда всё, что было до него, оставалось с крупностью по умолчанию: ролик
    открывался планом на 0.4 секунды и тут же менял крупность. Это и вспышка, и
    единственный случай, когда план режиссёра нарушал арифметику потолка кропа.
    """
    cuts = chain([0.43, 2.2, 2.31, 0.52, 0.5, 1.92], first="keep")
    cuts[1].reason = "hook"
    assign_framing(cuts)
    assert cuts[0].framing == cuts[1].framing == "tight"
    segments = 1 + sum(1 for a, b in zip(cuts, cuts[1:]) if a.framing != b.framing)
    assert segments == 2


def test_build_cuts_fills_framing():
    ws = mk([("раз", 0.5, 0.9), ("два", 1.1, 1.5), ("три", 5.0, 5.4), ("четыре", 5.6, 6.0)])
    cuts = build_cuts(ws, 7.0, hook_end_word=1, cfg=cfg())
    assert all(c.framing in ("wide", "tight") for c in cuts)
    assert next(c for c in cuts if c.reason == "hook").framing == "tight"


def test_edl_from_plan_fills_framing():
    ws = mk([(f"сл{i}", i * 0.4, i * 0.4 + 0.3) for i in range(9)])
    plan = {"hook_end_word": 2, "drop_words": [], "blocks": [{"words": list(range(9))}]}
    edl = edl_from_plan(ws, 5.0, plan, cfg=cfg())
    assert {c.framing for c in edl.kept_cuts} <= {"wide", "tight"}
    assert any(c.framing == "tight" for c in edl.kept_cuts)
