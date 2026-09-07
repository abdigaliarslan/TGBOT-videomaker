"""Тесты плана рендера: пересчёт времени, перенос строк, разъезд с b-roll."""

from __future__ import annotations

import pytest

from src.core.config import Settings
from src.core.edl import (
    EDL,
    Broll,
    BrollAnim,
    CaptionBlock,
    CaptionWord,
    Cut,
    Emphasis,
    Word,
)
from src.core.preset import load_preset
from src.core.render_plan import RenderPlan, build_plan


def cfg() -> Settings:
    return Settings(sub_tail=0.15, render_fps=30, render_width=1080, render_height=1920)


def sample_edl(**kw) -> EDL:
    words = [Word(idx=i, text=f"сл{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(6)]
    data = dict(
        source_duration=10.0,
        words=words,
        cuts=[
            Cut(src_in=0.0, src_out=1.4, reason="hook"),
            Cut(src_in=1.4, src_out=2.0, reason="drop_silence"),
            Cut(src_in=2.0, src_out=3.2, reason="keep"),
            Cut(src_in=3.2, src_out=10.0, reason="drop_silence"),
        ],
        captions=[
            CaptionBlock(words=[CaptionWord(word_idx=0), CaptionWord(word_idx=1), CaptionWord(word_idx=2)]),
            CaptionBlock(words=[CaptionWord(word_idx=4, emphasis=Emphasis.ACCENT), CaptionWord(word_idx=5)]),
        ],
        preset="glass",
    )
    data.update(kw)
    return EDL(**data)


# ------------------------------------------------------------------ таймлайн


def test_times_are_remapped_to_output_timeline():
    edl = sample_edl()
    plan = build_plan(edl, "v.mp4", apply_cuts=True, cfg=cfg())
    first = plan.blocks[0].words[0]
    second = plan.blocks[1].words[0]
    # слово 4 в исходнике начинается на 2.0 с; после выреза 1.4–2.0 — на 1.4 с
    assert first.start == pytest.approx(0.0, abs=0.01)
    assert second.start == pytest.approx(1.4, abs=0.01)
    assert plan.duration == pytest.approx(2.6, abs=0.01)


def test_without_cuts_times_are_source_times():
    edl = sample_edl()
    plan = build_plan(edl, "v.mp4", apply_cuts=False, cfg=cfg())
    assert plan.blocks[1].words[0].start == pytest.approx(2.0, abs=0.01)
    assert plan.duration == pytest.approx(10.0)


def test_dropped_words_do_not_reach_the_plan():
    edl = sample_edl()
    plan = build_plan(edl, "v.mp4", apply_cuts=True, cfg=cfg())
    idxs = [w.word_idx for b in plan.blocks for w in b.words]
    assert 3 not in idxs  # слово 3 попало в вырез


def test_blocks_never_overlap():
    edl = sample_edl()
    plan = build_plan(edl, "v.mp4", apply_cuts=True, cfg=cfg())
    for a, b in zip(plan.blocks, plan.blocks[1:]):
        assert a.end <= b.start + 1e-6


def test_last_block_does_not_exceed_duration():
    edl = sample_edl()
    plan = build_plan(edl, "v.mp4", apply_cuts=True, cfg=cfg())
    assert plan.blocks[-1].end <= plan.duration + 1e-6


# ------------------------------------------------------------------ пресет


def test_uppercase_preset_uppercases_text():
    edl = sample_edl(preset="bold")
    plan = build_plan(edl, "v.mp4", preset=load_preset("bold"), cfg=cfg())
    assert all(w.text == w.text.upper() for b in plan.blocks for w in b.words)


def test_preset_is_embedded_whole():
    """Рендерер не читает файлы пресетов: всё приходит в плане (§0.3)."""
    plan = build_plan(sample_edl(), "v.mp4", cfg=cfg())
    assert plan.preset["name"] == "glass"
    assert "colors" in plan.preset and "animation" in plan.preset


def test_preset_scales_to_frame_height():
    tall = build_plan(sample_edl(), "v.mp4", height=1920, width=1080, cfg=cfg())
    short = build_plan(sample_edl(), "v.mp4", height=960, width=540, cfg=cfg())
    assert short.preset["font"]["size"] < tall.preset["font"]["size"]


def test_emphasis_survives_into_plan():
    plan = build_plan(sample_edl(), "v.mp4", cfg=cfg())
    accents = [w.emphasis for b in plan.blocks for w in b.words if w.emphasis != "none"]
    assert accents == ["accent"]


# ------------------------------------------------------------------ b-roll и камера


def broll_edl(anchor: str) -> EDL:
    edl = sample_edl(cuts=[Cut(src_in=0.0, src_out=10.0, reason="keep")])
    edl.broll = [
        Broll(word_idx=0, duration=2.0, query="q", anim=BrollAnim.POP_IN, anchor=anchor)
    ]
    return edl


def test_caption_moves_away_from_bottom_broll():
    """§7 DoD: картинка не должна перекрывать текст."""
    plan = build_plan(broll_edl("bottom_right"), "v.mp4", cfg=cfg())
    assert plan.blocks[0].position == "top"


def test_caption_stays_when_broll_is_on_top():
    plan = build_plan(broll_edl("top_right"), "v.mp4", cfg=cfg())
    assert plan.blocks[0].position == "bottom"


def test_explicit_position_is_respected_over_collision():
    edl = broll_edl("bottom_right")
    edl.captions[0].position = "bottom"
    plan = build_plan(edl, "v.mp4", cfg=cfg())
    assert plan.blocks[0].position == "bottom"


def test_broll_is_timed_and_clamped():
    plan = build_plan(broll_edl("center"), "v.mp4", cfg=cfg())
    item = plan.broll[0]
    assert item.start == pytest.approx(0.0, abs=0.05)
    assert item.end <= plan.duration + 1e-6
    assert item.end - item.start == pytest.approx(2.0, abs=0.05)


def test_plan_round_trips_through_json():
    plan = build_plan(sample_edl(), "v.mp4", cfg=cfg())
    again = RenderPlan.from_dict(plan.to_dict())
    assert again.to_dict() == plan.to_dict()


def test_plan_json_has_no_none_video():
    plan = build_plan(sample_edl(), "v.mp4", cfg=cfg())
    assert plan.to_dict()["video"] == "v.mp4"
    assert plan.to_dict()["fps"] == 30
