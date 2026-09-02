"""Тесты автокропа (M6): чистая математика трека + реальный трекинг на фикстуре."""

from __future__ import annotations

import pytest

from src.pipeline.crop import (
    apply_deadzone,
    build_sendcmd,
    clamp_track,
    crop_size,
    fill_gaps,
    moving_average,
)


# ------------------------------------------------------------------ геометрия


@pytest.mark.parametrize(
    "w,h,expected",
    [
        (1920, 1080, (606, 1080)),  # горизонталь → узкое окно во всю высоту
        (1080, 1920, (1080, 1920)),  # уже 9:16 → окно во весь кадр
        (1080, 1080, (606, 1080)),  # квадрат → режем по горизонтали
        (1280, 720, (404, 720)),
    ],
)
def test_crop_size(w, h, expected):
    got = crop_size(w, h)
    assert got[0] % 2 == 0 and got[1] % 2 == 0
    assert got == expected


def test_crop_window_fits_frame_for_wide_source():
    w, h = crop_size(1920, 1080)
    assert w <= 1920 and h <= 1080
    assert abs(w / h - 9 / 16) < 0.01


def test_clamp_keeps_window_inside_frame():
    xs = clamp_track([-500.0, 0.0, 960.0, 5000.0], window=606, frame=1920)
    assert all(0 <= x <= 1920 - 606 for x in xs)
    assert xs[0] == 0
    assert xs[-1] == 1920 - 606


# ------------------------------------------------------------------ трек


def test_fill_gaps_interpolates():
    assert fill_gaps([0.0, None, 10.0], default=5.0) == [0.0, 5.0, 10.0]


def test_fill_gaps_holds_edges():
    assert fill_gaps([None, None, 7.0, None], default=0.0) == [7.0, 7.0, 7.0, 7.0]


def test_fill_gaps_all_none_uses_default():
    assert fill_gaps([None, None], default=3.5) == [3.5, 3.5]


def test_moving_average_smooths_spike():
    raw = [0.0, 0.0, 100.0, 0.0, 0.0]
    smoothed = moving_average(raw, 3)
    assert smoothed[2] < 100.0
    assert max(smoothed) < 40.0


def test_moving_average_window_one_is_identity():
    assert moving_average([1.0, 2.0, 3.0], 1) == [1.0, 2.0, 3.0]


def test_deadzone_suppresses_jitter():
    raw = [100.0, 101.0, 99.5, 100.5, 140.0]
    out = apply_deadzone(raw, threshold=5.0)
    assert out[:4] == [100.0, 100.0, 100.0, 100.0]
    assert out[4] == 140.0


def test_deadzone_does_not_drift_on_slow_ramp():
    """Медленный сдвиг внутри порога камеру двигать не должен."""
    raw = [100.0 + i * 0.5 for i in range(20)]
    out = apply_deadzone(raw, threshold=5.0)
    assert len(set(out)) < len(set(raw))


# ------------------------------------------------------------------ sendcmd


def test_sendcmd_skips_repeated_positions():
    script = build_sendcmd([0.0, 0.5, 1.0, 1.5], [10, 10, 10, 40], [0, 0, 0, 0], static_y=True)
    assert script.count("crop x") == 2  # начальная позиция и одно изменение


def test_sendcmd_intervals_are_ordered():
    script = build_sendcmd([0.0, 0.5, 1.0], [10, 20, 30], [0, 1, 2])
    starts = [float(line.split("-")[0]) for line in script.strip().splitlines()]
    assert starts == sorted(starts)


def test_sendcmd_omits_axis_that_does_not_move():
    script = build_sendcmd([0.0, 1.0], [10, 20], [5, 6], static_y=True)
    assert "crop y" not in script
    assert "crop x" in script


# ------------------------------------------------------------------ реальный трекинг


@pytest.mark.slow
def test_tracker_follows_moving_object():
    """На фикстуре лиц нет — проверяем запасной трек по движению.

    Оранжевый прямоугольник в фикстуре ходит по синусоиде с амплитудой 380 px
    и периодом 6 с. Трек обязан за ним следовать, а не стоять в центре.
    """
    from tests.fixtures.make_fixture import ensure
    from src.pipeline.crop import track_attention
    from src.pipeline.probe import probe

    video, _ = ensure()
    info = probe(video)
    times, xs, ys, _name = track_attention(video, info=info, sample_fps=6.0)

    filled = fill_gaps(xs, info.width / 2)
    assert len(filled) > 20
    spread = max(filled) - min(filled)
    assert spread > 300, f"трек почти не двигается (размах {spread:.0f} px)"
    assert all(0 <= v <= info.width for v in filled)


@pytest.mark.slow
def test_autocrop_produces_vertical_video(tmp_path):
    from tests.fixtures.make_fixture import ensure
    from src.pipeline.crop import autocrop
    from src.pipeline.probe import probe

    video, _ = ensure()
    out = autocrop(video, tmp_path / "vertical.mp4", work_dir=tmp_path)
    info = probe(out)
    assert (info.width, info.height) == (1080, 1920)
    assert info.has_audio
