"""Тесты автокропа (M6): математика окна и трека, цветокор, граф ffmpeg.

Главное, что здесь проверяется: крупность выводится из ДОЛИ ЛИЦА и никогда не
добирается апскейлом, макушка не срезается, а смена крупности происходит ровно
на резе. Всё это чистые функции — реальный трекинг вынесен в slow-тесты.
"""

from __future__ import annotations

import pytest

from src.core.edl import EDL, Cut, Word
from src.core.preset import Crop, Grade, load_preset
from src.pipeline.crop import (
    Detection,
    Segment,
    Track,
    apply_deadzone,
    build_crop_filter,
    build_sendcmd,
    clamp_track,
    clamp_window,
    crop_size,
    fill_gaps,
    framing_spans,
    grade_filter,
    moving_average,
    normalize_spans,
    place_window,
    plan_segments,
    window_bounds,
    window_height,
)
from src.pipeline.probe import MediaInfo

CROP = Crop()  # значения по умолчанию совпадают с пресетами


def info_of(width: int, height: int, duration: float = 20.0) -> MediaInfo:
    return MediaInfo(
        path="—", duration=duration, width=width, height=height, fps=30.0,
        has_audio=True, audio_channels=2, audio_rate=48000, rotation=0,
        video_codec="h264", bitrate=0,
    )


def face(y: float, h: float, x: float = 0.0) -> Detection:
    """Лицо с боксом высоты h и центром в (x, y). Линия глаз — по боксу."""
    return Detection(x=x, y=y, score=0.9, area=h * h * 0.75, w=h * 0.75, h=h)


def track_of(faces: list[Detection | None], step: float = 1 / 6) -> Track:
    return Track(
        times=[i * step for i in range(len(faces))],
        faces=faces,
        motion_x=[None] * len(faces),
        detector="test",
    )


# ------------------------------------------------------------------ геометрия окна


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


def test_clamp_window_takes_edges_not_centers():
    assert clamp_window([-10.0, 100.0, 9999.0], window=600, frame=1000) == [0, 100, 400]


def test_window_bounds_forbid_upscaling_beyond_max_scale():
    """Исходник крупнее выхода — есть куда зумиться, но только до max_scale."""
    lo, hi = window_bounds(2160, 3840, 1080, 1920, max_scale=1.0)
    assert (lo, hi) == (1920.0, 3840.0)


def test_window_bounds_collapse_when_source_smaller_than_output():
    """720p селфи: любое окно — уже апскейл, выбирать не из чего."""
    lo, hi = window_bounds(720, 1280, 1080, 1920, max_scale=1.0)
    assert lo == hi == 1280.0


def test_window_bounds_never_exceed_frame():
    lo, hi = window_bounds(1920, 1080, 1080, 1920, max_scale=1.0)
    assert hi == 1080.0 and lo == 1080.0  # горизонталь: окно во всю высоту, и всё


def test_window_height_follows_face_ratio():
    """Высота окна выводится из доли лица, а не из вписывания бокса."""
    h = window_height(400.0, 0.0, ratio=0.22, crop=CROP, lo=0.0, hi=1e9)
    assert h == pytest.approx(400 / 0.22, rel=1e-6)


def test_tight_window_is_smaller_than_wide():
    wide = window_height(400.0, 0.0, ratio=CROP.face_height_ratio, crop=CROP, lo=0.0, hi=1e9)
    tight = window_height(
        400.0, 0.0, ratio=CROP.tight_face_height_ratio, crop=CROP, lo=0.0, hi=1e9
    )
    assert tight < wide


def test_window_height_respects_headroom():
    """Голова с запасом обязана влезть — иначе окно увеличивается."""
    # eye_to_head = 700 при eye_line 0.36 и headroom 0.05 требует окна ≥ 2258 px,
    # хотя по доле лица хватило бы 1000/0.22 = 4545... берём лицо помельче.
    h = window_height(200.0, 700.0, ratio=0.9, crop=CROP, lo=0.0, hi=1e9)
    assert h >= 700.0 / (CROP.eye_line_ratio - CROP.headroom_ratio) - 1e-6


def test_window_height_clamped_into_bounds():
    assert window_height(900.0, 0.0, ratio=0.22, crop=CROP, lo=1280.0, hi=1280.0) == 1280.0


def test_window_height_without_face_is_widest():
    assert window_height(0.0, 0.0, ratio=0.22, crop=CROP, lo=100.0, hi=1280.0) == 1280.0


# ------------------------------------------------------------------ позиция окна


def test_eye_line_sits_at_preset_ratio():
    f = face(y=900.0, h=400.0)
    _, y = place_window(
        540.0, f.eyes, f.head_top, win_w=1080, win_h=1920,
        frame_w=2160, frame_h=3840, crop=CROP,
    )
    assert (f.eyes - y) / 1920 == pytest.approx(CROP.eye_line_ratio, abs=1e-3)


def test_head_top_stays_inside_window():
    """Макушка с запасом внутри кадра при любом разумном окне."""
    f = face(y=900.0, h=400.0)
    for win_h in (1920, 2400, 3000, 3840):
        win_w = int(win_h * 9 / 16)
        _, y = place_window(
            540.0, f.eyes, f.head_top, win_w=win_w, win_h=win_h,
            frame_w=2160, frame_h=3840, crop=CROP,
        )
        assert y <= f.head_top - CROP.headroom_ratio * win_h + 1e-6, win_h


def test_window_never_leaves_frame():
    f = face(y=100.0, h=300.0, x=0.0)
    x, y = place_window(
        f.x, f.eyes, f.head_top, win_w=1080, win_h=1920,
        frame_w=1080, frame_h=1920, crop=CROP,
    )
    assert (x, y) == (0.0, 0.0)


def test_horizontal_placement_centers_on_attention():
    x, _ = place_window(
        1500.0, 400.0, 200.0, win_w=600, win_h=1067,
        frame_w=1920, frame_h=1080, crop=CROP,
    )
    assert x == pytest.approx(1500 - 300)


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


def test_detection_uses_landmarks_for_eye_line():
    with_marks = Detection(x=0, y=100, score=1.0, area=1.0, w=50, h=100, eye_y=70.0)
    without = Detection(x=0, y=100, score=1.0, area=1.0, w=50, h=100)
    assert with_marks.eyes == 70.0
    assert without.eyes == pytest.approx(50 + 0.3 * 100)  # верх бокса + доля высоты


def test_detection_rescale_keeps_ratios():
    d = Detection(x=10, y=20, score=1.0, area=100.0, w=10, h=10, eye_y=18.0).rescaled(2.0)
    assert (d.x, d.y, d.w, d.h, d.eye_y) == (20, 40, 20, 20, 36.0)
    assert d.area == 400.0


def test_track_medians_ignore_missing_faces():
    t = track_of([face(500, 300), None, face(500, 500), face(500, 400)])
    assert t.face_height() == 400.0


# ------------------------------------------------------------------ крупность по резам


def edl_of(cuts: list[tuple[float, float, str, str]], duration: float) -> EDL:
    return EDL(
        source_duration=duration,
        words=[Word(idx=0, text="раз", start=0.1, end=0.4)],
        cuts=[Cut(src_in=a, src_out=b, reason=r, framing=f) for a, b, r, f in cuts],
        captions=[],
    )


def test_framing_spans_use_output_time():
    """После нарезки куски склеены встык — крупность живёт в выходном времени."""
    edl = edl_of(
        [
            (0.0, 0.2, "drop_silence", "wide"),
            (0.2, 5.0, "hook", "tight"),
            (5.0, 6.0, "drop_filler", "wide"),
            (6.0, 12.0, "keep", "wide"),
        ],
        12.0,
    )
    assert framing_spans(edl) == [(0.0, 4.8, "tight"), (4.8, 10.8, "wide")]


def test_framing_does_not_change_without_real_cuts():
    """Паузы не режем — резов нет, и менять крупность посреди плана не на чем."""
    edl = edl_of(
        [(0.0, 5.0, "hook", "tight"), (5.0, 6.0, "drop_silence", "wide"),
         (6.0, 12.0, "keep", "wide")],
        12.0,
    )
    assert framing_spans(edl, applied_cuts=False) == [(0.0, 12.0, "wide")]


def test_framing_spans_empty_without_kept_cuts():
    assert framing_spans(edl_of([(0.0, 3.0, "drop_silence", "wide")], 3.0)) == []


def test_normalize_spans_covers_whole_clip():
    spans = normalize_spans([(2.0, 5.0, "tight")], 10.0)
    assert spans[0][0] == 0.0 and spans[-1][1] == 10.0
    for a, b in zip(spans, spans[1:]):
        assert a[1] == pytest.approx(b[0])


def test_normalize_spans_without_input_is_all_wide():
    assert normalize_spans(None, 8.0) == [(0.0, 8.0, "wide")]


def test_normalize_spans_rejects_unknown_framing():
    assert normalize_spans([(0.0, 4.0, "dolly_zoom")], 4.0) == [(0.0, 4.0, "wide")]


# ------------------------------------------------------------------ план кропа


def test_plan_segments_hits_target_face_share():
    """На исходнике с запасом доля лица попадает ровно в заказанную."""
    preset = load_preset("glass")
    info = info_of(5120, 2880, duration=12.0)
    track = track_of([face(y=1400.0, h=580.0, x=2560.0) for _ in range(72)], step=1 / 6)
    segments = plan_segments(
        track, [(0.0, 6.0, "wide"), (6.0, 12.0, "tight")],
        info=info, out_w=1080, out_h=1920, preset=preset,
    )
    assert len(segments) == 2
    shares = [s.face_share(580.0) for s in segments]
    assert shares[0] == pytest.approx(preset.crop.face_height_ratio, abs=0.01)
    assert shares[1] == pytest.approx(preset.crop.tight_face_height_ratio, abs=0.01)
    assert all(0.20 <= s <= 0.30 for s in shares)


def test_plan_segments_never_upscale_beyond_max_scale():
    """720p селфи: окно = кадр, крупность не трогаем ни на одном куске."""
    preset = load_preset("glass")
    info = info_of(720, 1280, duration=12.0)
    track = track_of([face(y=600.0, h=600.0, x=360.0) for _ in range(72)])
    segments = plan_segments(
        track, [(0.0, 6.0, "tight"), (6.0, 12.0, "wide")],
        info=info, out_w=1080, out_h=1920, preset=preset,
    )
    assert {(s.win_w, s.win_h) for s in segments} == {(720, 1280)}


def test_plan_segments_merge_identical_windows():
    """Одинаковые окна на соседних кусках — одна ветка графа, а не две."""
    preset = load_preset("glass")
    info = info_of(720, 1280, duration=12.0)
    track = track_of([face(y=600.0, h=600.0, x=360.0) for _ in range(72)])
    segments = plan_segments(
        track, [(0.0, 6.0, "tight"), (6.0, 12.0, "wide")],
        info=info, out_w=1080, out_h=1920, preset=preset,
    )
    assert len(segments) == 1 and segments[0].end == pytest.approx(12.0)


def test_plan_segments_switch_only_on_cut():
    """Внутри куска размер окна постоянен — зум происходит на резе, и только там."""
    preset = load_preset("glass")
    info = info_of(5120, 2880, duration=12.0)
    faces = [face(y=1400.0, h=580.0 + (i % 7) * 12.0, x=2560.0) for i in range(72)]
    segments = plan_segments(
        track_of(faces), [(0.0, 6.0, "wide"), (6.0, 12.0, "tight")],
        info=info, out_w=1080, out_h=1920, preset=preset,
    )
    assert len(segments) == 2
    assert segments[0].win_h != segments[1].win_h
    assert segments[0].end == pytest.approx(segments[1].start)


def test_plan_segments_never_cut_the_top_of_the_head():
    preset = load_preset("glass")
    info = info_of(5120, 2880, duration=12.0)
    head_top = face(y=1400.0, h=580.0).head_top
    track = track_of([face(y=1400.0, h=580.0, x=2560.0) for _ in range(72)])
    segments = plan_segments(
        track, [(0.0, 6.0, "wide"), (6.0, 12.0, "tight")],
        info=info, out_w=1080, out_h=1920, preset=preset,
    )
    for seg in segments:
        assert max(seg.ys) <= head_top, seg


def test_plan_segments_rects_stay_inside_frame():
    preset = load_preset("glass")
    info = info_of(1920, 1080, duration=8.0)
    faces = [face(y=500.0, h=200.0, x=200.0 + i * 20.0) for i in range(48)]
    segments = plan_segments(
        track_of(faces), [(0.0, 8.0, "wide")],
        info=info, out_w=1080, out_h=1920, preset=preset,
    )
    for seg in segments:
        assert all(0 <= x <= info.width - seg.win_w for x in seg.xs)
        assert all(0 <= y <= info.height - seg.win_h for y in seg.ys)


def test_plan_segments_collapse_when_too_many_changes():
    preset = load_preset("glass")
    info = info_of(5120, 2880, duration=30.0)
    faces = [face(y=1400.0, h=580.0, x=2560.0) for _ in range(180)]
    spans = [
        (i * 1.0, i * 1.0 + 1.0, "tight" if i % 2 else "wide") for i in range(30)
    ]
    segments = plan_segments(
        track_of(faces), spans, info=info, out_w=1080, out_h=1920, preset=preset
    )
    assert len(segments) == 1


# ------------------------------------------------------------------ цветокор (Ф3)


def test_grade_filter_from_preset():
    got = grade_filter(load_preset("glass").grade)
    assert got.startswith("eq=contrast=1.08:saturation=1.12:gamma=0.98")
    assert "vignette=angle=0.63" in got


def test_grade_filter_disabled_is_empty():
    assert grade_filter(Grade(enabled=False, contrast=1.5)) == ""


def test_grade_filter_without_vignette():
    got = grade_filter(Grade(vignette=0.0))
    assert "vignette" not in got and got.startswith("eq=")


def test_grade_filter_neutral_values_produce_nothing():
    assert grade_filter(Grade(contrast=1.0, saturation=1.0, gamma=1.0, vignette=0.0)) == ""


def test_grade_saturation_stays_under_red_skin_threshold():
    """Выше 1.2 на лицах даёт красноту — это ограничение пресета, не кода."""
    for name in ("glass", "bold"):
        assert load_preset(name).grade.saturation <= 1.2


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


def test_sendcmd_addresses_named_filter():
    script = build_sendcmd([0.0, 1.0], [10, 20], [5, 6], target="crop@c2")
    assert "crop@c2 x" in script and "crop@c2 y" in script


def test_sendcmd_static_both_axes_is_empty():
    assert build_sendcmd([0.0, 1.0], [10, 10], [5, 5], static_x=True, static_y=True) == ""


# ------------------------------------------------------------------ граф ffmpeg


def seg(start, end, framing, w, h, x=0, y=0) -> Segment:
    return Segment(start, end, framing, w, h, [start], [x], [y])


def test_filter_single_segment_has_no_split():
    vf = build_crop_filter(
        [seg(0.0, 10.0, "wide", 600, 1066, 100, 5)],
        frame_w=1920, frame_h=1080, out_w=1080, out_h=1920, grade="eq=contrast=1.08",
    )
    assert "split=" not in vf and "concat" not in vf
    assert "crop@c0=w=600:h=1066:x=100:y=5" in vf
    assert vf.endswith("eq=contrast=1.08,setsar=1[vout]")


def test_filter_skips_crop_when_window_is_whole_frame():
    vf = build_crop_filter(
        [seg(0.0, 10.0, "wide", 720, 1280)],
        frame_w=720, frame_h=1280, out_w=1080, out_h=1920,
    )
    assert "crop" not in vf
    assert "scale=1080:1920:flags=lanczos" in vf


def test_filter_branches_per_framing_segment():
    vf = build_crop_filter(
        [seg(0.0, 5.0, "tight", 600, 1066), seg(5.0, 12.0, "wide", 800, 1422)],
        frame_w=1920, frame_h=1080, out_w=1080, out_h=1920,
    )
    assert "split=2[b0][b1]" in vf
    assert "trim=start=0.0000:end=5.0000" in vf
    assert "trim=start=5.0000," in vf  # у последнего конец не ставим
    assert vf.count("setpts=PTS-STARTPTS") == 2
    assert "concat=n=2:v=1:a=0[vcat]" in vf
    # Размер окна на ходу не меняется: каждая ветка — свой crop постоянного размера.
    assert "crop@c0=w=600:h=1066" in vf and "crop@c1=w=800:h=1422" in vf


def test_filter_grades_once_after_concat():
    vf = build_crop_filter(
        [seg(0.0, 5.0, "tight", 600, 1066), seg(5.0, 12.0, "wide", 800, 1422)],
        frame_w=1920, frame_h=1080, out_w=1080, out_h=1920, grade="eq=contrast=1.1",
    )
    assert vf.count("eq=contrast=1.1") == 1
    assert vf.index("concat") < vf.index("eq=contrast=1.1")


def test_filter_without_grade_is_plain_setsar():
    vf = build_crop_filter(
        [seg(0.0, 5.0, "wide", 600, 1066)],
        frame_w=1920, frame_h=1080, out_w=1080, out_h=1920,
    )
    assert vf.endswith("[vcat]setsar=1[vout]")


def test_filter_inserts_sendcmd_before_split(tmp_path):
    script = tmp_path / "a.sendcmd"
    script.write_text("0.000-1.000 crop@c0 x 10;\n", encoding="utf-8")
    vf = build_crop_filter(
        [seg(0.0, 5.0, "tight", 600, 1066), seg(5.0, 12.0, "wide", 800, 1422)],
        frame_w=1920, frame_h=1080, out_w=1080, out_h=1920, cmd_file=script,
    )
    assert vf.index("sendcmd=") < vf.index("split=")


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
    track = track_attention(video, info=info, sample_fps=6.0)

    filled = fill_gaps(track.motion_x, info.width / 2)
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


@pytest.mark.slow
def test_autocrop_keeps_duration_when_framing_changes(tmp_path):
    """Смена крупности режет граф на ветки — длительность и звук обязаны выжить."""
    from tests.fixtures.make_fixture import ensure
    from src.pipeline.crop import autocrop
    from src.pipeline.probe import probe

    video, _ = ensure()
    before = probe(video)
    mid = before.duration / 2
    out = autocrop(
        video, tmp_path / "framed.mp4", work_dir=tmp_path,
        framing=[(0.0, mid, "wide"), (mid, before.duration, "tight")],
    )
    after = probe(out)
    assert (after.width, after.height) == (1080, 1920)
    assert after.has_audio
    assert after.duration == pytest.approx(before.duration, abs=0.2)
