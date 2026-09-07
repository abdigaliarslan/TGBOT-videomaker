"""Тесты семантического валидатора EDL (§3) и пересчёта таймлайна."""

from __future__ import annotations

import pytest

from src.core.edl import (
    EDL,
    Broll,
    BrollAnim,
    CaptionBlock,
    CaptionWord,
    Cut,
    Emphasis,
    EDLValidationError,
    Timeline,
    Word,
    assert_valid,
    kept_word_indices,
    validate_edl,
)


def words(n: int, step: float = 0.5) -> list[Word]:
    return [Word(idx=i, text=f"w{i}", start=i * step, end=i * step + step * 0.8) for i in range(n)]


def blocks(groups: list[list[int]], accents: dict[int, Emphasis] | None = None) -> list[CaptionBlock]:
    accents = accents or {}
    return [
        CaptionBlock(
            words=[CaptionWord(word_idx=i, emphasis=accents.get(i, Emphasis.NONE)) for i in g]
        )
        for g in groups
    ]


def base_edl(**kw) -> EDL:
    ws = kw.pop("words", words(9))
    n = len(ws)
    groups = kw.pop("groups", [list(range(i, min(i + 3, n))) for i in range(0, n, 3)])
    data = dict(
        source_duration=kw.pop("source_duration", ws[-1].end + 0.5),
        words=ws,
        cuts=kw.pop("cuts", []),
        captions=kw.pop("captions", blocks(groups)),
    )
    data.update(kw)
    return EDL(**data)


# ------------------------------------------------------------------ счастливый путь


def test_valid_edl_passes():
    assert validate_edl(base_edl()) == []


def test_valid_with_cuts_and_dropped_words():
    ws = words(9)  # слова 0..8, слово i занимает [0.5i, 0.5i+0.4]
    # оставляем 0..1.2 (слова 0,1,2) и 2.0..4.5 (слова 4..8), слово 3 (1.5..1.9) вырезано
    cuts = [
        Cut(src_in=0.0, src_out=1.3, reason="hook"),
        Cut(src_in=1.3, src_out=2.0, reason="drop_silence"),
        Cut(src_in=2.0, src_out=4.5, reason="keep"),
    ]
    edl = base_edl(words=ws, cuts=cuts, captions=blocks([[0, 1, 2], [4, 5, 6], [7, 8]]))
    assert kept_word_indices(edl) == [0, 1, 2, 4, 5, 6, 7, 8]
    assert validate_edl(edl) == []


# ------------------------------------------------------------------ ссылки на слова


def test_unknown_word_idx_in_captions():
    edl = base_edl(captions=blocks([[0, 1, 2], [3, 4, 99]]))
    assert any("отсутствует в words" in e for e in validate_edl(edl))


def test_unknown_word_idx_in_broll():
    edl = base_edl(broll=[Broll(word_idx=99, duration=1.5, query="cat", anim=BrollAnim.POP_IN, anchor="center")])
    assert any("broll[0]" in e and "отсутствует" in e for e in validate_edl(edl))


def test_word_missing_from_all_blocks():
    edl = base_edl(captions=blocks([[0, 1, 2], [3, 4, 5], [6, 7]]))  # 8 потеряно
    assert any("не попали ни в один блок" in e for e in validate_edl(edl))


def test_word_in_two_blocks():
    edl = base_edl(captions=blocks([[0, 1, 2], [2, 3, 4], [5, 6, 7], [8]]))
    errs = validate_edl(edl)
    assert any("повторяются" in e for e in errs)


def test_order_must_be_increasing():
    edl = base_edl(captions=blocks([[0, 1, 2], [6, 7, 8], [3, 4, 5]]))
    assert any("порядок слов нарушен" in e for e in validate_edl(edl))


def test_dropped_word_must_not_be_captioned():
    ws = words(9)
    cuts = [
        Cut(src_in=0.0, src_out=1.3, reason="keep"),
        Cut(src_in=1.3, src_out=2.0, reason="drop_filler"),
        Cut(src_in=2.0, src_out=4.5, reason="keep"),
    ]
    edl = base_edl(words=ws, cuts=cuts, captions=blocks([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    assert any("вырезанных кусков" in e for e in validate_edl(edl))


# ------------------------------------------------------------------ cuts


def test_cuts_must_not_overlap():
    edl = base_edl(
        cuts=[Cut(src_in=0.0, src_out=2.0, reason="keep"), Cut(src_in=1.5, src_out=4.5, reason="keep")]
    )
    assert any("пересекается" in e for e in validate_edl(edl, strict_coverage=False))


def test_cuts_must_be_sorted():
    edl = base_edl(
        cuts=[Cut(src_in=2.5, src_out=4.5, reason="keep"), Cut(src_in=0.0, src_out=2.0, reason="keep")]
    )
    assert any("пересекается" in e for e in validate_edl(edl, strict_coverage=False))


def test_cut_outside_source():
    edl = base_edl(cuts=[Cut(src_in=0.0, src_out=99.0, reason="keep")])
    assert any("выход за пределы" in e for e in validate_edl(edl, strict_coverage=False))


def test_all_cuts_dropped_is_error():
    edl = base_edl(cuts=[Cut(src_in=0.0, src_out=4.5, reason="drop_silence")])
    assert any("kept" in e for e in validate_edl(edl, strict_coverage=False))


# ------------------------------------------------------------------ акценты


def test_at_most_one_accent_per_block():
    edl = base_edl(
        captions=blocks(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            accents={3: Emphasis.ACCENT, 4: Emphasis.SHOUT},
        )
    )
    assert any("акцентных слов" in e for e in validate_edl(edl))


def test_one_accent_per_block_is_fine():
    edl = base_edl(
        captions=blocks([[0, 1, 2], [3, 4, 5], [6, 7, 8]], accents={4: Emphasis.SHOUT})
    )
    assert validate_edl(edl) == []


def test_block_word_count_limit():
    edl = base_edl(words=words(6), captions=blocks([[0, 1, 2, 3, 4, 5]]))
    assert any("допустимо" in e for e in validate_edl(edl))


# ------------------------------------------------------------------ b-roll


def test_broll_count_cap():
    ws = words(40, step=0.5)  # ~20 c → потолок ceil(20/8) = 3
    edl = base_edl(
        words=ws,
        groups=[list(range(i, min(i + 4, 40))) for i in range(0, 40, 4)],
        broll=[
            Broll(word_idx=i, duration=1.0, query="q", anim=BrollAnim.POP_IN, anchor="center")
            for i in (0, 4, 8, 12, 16)
        ],
    )
    assert any("при потолке" in e for e in validate_edl(edl))


def test_broll_density():
    ws = words(40, step=0.5)
    edl = base_edl(
        words=ws,
        groups=[list(range(i, min(i + 4, 40))) for i in range(0, 40, 4)],
        broll=[
            Broll(word_idx=i, duration=1.0, query="q", anim=BrollAnim.SLIDE_UP, anchor="center")
            for i in (0, 2)  # 0.0 c и 1.0 c — слишком близко
        ],
    )
    assert any("ближе" in e for e in validate_edl(edl))


def test_broll_duration_bounds_are_pydantic():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        Broll(word_idx=0, duration=5.0, query="q", anim=BrollAnim.POP_IN, anchor="center")
    with pytest.raises(pydantic.ValidationError):
        Broll(word_idx=0, duration=0.1, query="q", anim=BrollAnim.POP_IN, anchor="center")


def test_closed_enums_reject_invented_effects():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        Broll(word_idx=0, duration=1.0, query="q", anim="explode", anchor="center")
    with pytest.raises(pydantic.ValidationError):
        CaptionWord(word_idx=0, emphasis="rainbow")


# ------------------------------------------------------------------ таймлайн


def test_timeline_maps_across_gaps():
    tl = Timeline(
        [Cut(src_in=0.0, src_out=2.0, reason="keep"), Cut(src_in=5.0, src_out=6.0, reason="keep")],
        source_duration=10.0,
    )
    assert tl.duration == pytest.approx(3.0)
    assert tl.map(0.0) == pytest.approx(0.0)
    assert tl.map(1.5) == pytest.approx(1.5)
    assert tl.map(5.0) == pytest.approx(2.0)
    assert tl.map(5.5) == pytest.approx(2.5)
    assert tl.keeps(3.0) is False
    assert tl.map(3.0, clamp=False) is None


def test_timeline_without_cuts_is_identity():
    tl = Timeline([], source_duration=7.0)
    assert tl.duration == pytest.approx(7.0)
    assert tl.map(3.3) == pytest.approx(3.3)


def test_word_window_none_for_dropped_word():
    tl = Timeline([Cut(src_in=0.0, src_out=2.0, reason="keep")], source_duration=10.0)
    assert tl.word_window(Word(idx=0, text="a", start=0.1, end=0.4)) is not None
    assert tl.word_window(Word(idx=1, text="b", start=5.0, end=5.4)) is None


def test_out_duration():
    edl = base_edl(
        cuts=[Cut(src_in=0.0, src_out=1.0, reason="keep"), Cut(src_in=2.0, src_out=4.0, reason="keep")],
        captions=blocks([[0, 1]]),
    )
    assert edl.out_duration == pytest.approx(3.0)


def test_assert_valid_raises_with_errors():
    edl = base_edl(captions=blocks([[0, 1, 2]]))
    with pytest.raises(EDLValidationError) as ei:
        assert_valid(edl)
    assert ei.value.errors
