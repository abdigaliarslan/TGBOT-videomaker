"""Тесты плана рендера: пересчёт времени, перенос строк, разъезд с b-roll, геометрия."""

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
from src.core.geometry import Rect, broll_rect
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


# ------------------------------------------------------------------ геометрия (Ф4)


def test_every_block_carries_a_rect_inside_the_frame():
    """Рендерер получает готовый прямоугольник, а не половину кадра (AUDIT §8.4)."""
    plan = build_plan(sample_edl(), "v.mp4", cfg=cfg())
    assert plan.blocks
    for b in plan.blocks:
        assert b.rect is not None
        assert 0 <= b.rect.x and b.rect.right <= plan.width + 1e-6
        assert 0 <= b.rect.y and b.rect.bottom <= plan.height + 1e-6


def test_block_rect_follows_the_resolved_position():
    """Прямоугольник считается ПОСЛЕ разъезда с картинкой, а не до него."""
    top = build_plan(broll_edl("bottom_right"), "v.mp4", cfg=cfg()).blocks[0]
    bottom = build_plan(broll_edl("top_right"), "v.mp4", cfg=cfg()).blocks[0]
    assert top.position == "top" and bottom.position == "bottom"
    assert top.rect is not None and bottom.rect is not None
    assert top.rect.y < bottom.rect.y


def test_block_rect_scales_with_the_frame():
    tall = build_plan(sample_edl(), "v.mp4", height=1920, width=1080, cfg=cfg())
    short = build_plan(sample_edl(), "v.mp4", height=960, width=540, cfg=cfg())
    assert tall.blocks[0].rect.width > short.blocks[0].rect.width  # type: ignore[union-attr]
    assert short.blocks[0].rect.right <= 540 + 1e-6  # type: ignore[union-attr]


def test_broll_carries_the_rect_of_its_anchor():
    plan = build_plan(broll_edl("bottom_right"), "v.mp4", cfg=cfg())
    preset = load_preset("glass")
    assert plan.broll[0].rect == broll_rect("bottom_right", preset, (plan.width, plan.height))


def test_fullscreen_to_inset_rect_is_the_corner_it_ends_in():
    edl = broll_edl("fullscreen")
    edl.broll[0].anim = BrollAnim.FULLSCREEN_TO_INSET
    plan = build_plan(edl, "v.mp4", cfg=cfg())
    rect = plan.broll[0].rect
    assert rect is not None and rect.width < plan.width


def test_rects_survive_the_json_round_trip():
    plan = build_plan(broll_edl("bottom_right"), "v.mp4", cfg=cfg())
    again = RenderPlan.from_dict(plan.to_dict())
    assert again.blocks[0].rect == plan.blocks[0].rect.rounded()  # type: ignore[union-attr]
    assert again.broll[0].rect == plan.broll[0].rect.rounded()  # type: ignore[union-attr]


def test_plan_without_rects_still_loads():
    """Планы, собранные до Ф4, читаются: прямоугольников просто нет."""
    data = build_plan(broll_edl("bottom_right"), "v.mp4", cfg=cfg()).to_dict()
    for item in data["blocks"] + data["broll"]:
        item.pop("rect")
    plan = RenderPlan.from_dict(data)
    assert plan.blocks[0].rect is None
    assert plan.broll[0].rect is None


def test_missing_font_metrics_do_not_break_the_plan():
    """Нет метрик — нет прямоугольника, но ролик всё равно рендерится."""
    preset = load_preset("glass").model_copy(deep=True)
    preset.font.file = None
    plan = build_plan(sample_edl(), "v.mp4", preset=preset, cfg=cfg())
    assert plan.blocks and plan.blocks[0].rect is None


def test_caption_and_broll_rects_are_comparable():
    """Разводить субтитр с картинкой теперь можно пересечением, а не половинами."""
    plan = build_plan(broll_edl("bottom_right"), "v.mp4", cfg=cfg())
    caption = plan.blocks[0].rect
    picture = plan.broll[0].rect
    assert isinstance(caption, Rect) and isinstance(picture, Rect)
    assert not caption.overlaps(picture)
