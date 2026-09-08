"""М1: трек лица как данные — пересчёт, пропуски, сглаживание.

Проверяется не «детектор нашёл лицо» (это работа OpenCV и замеряется
инструментом `tests/face_track_check.py` на настоящем ролике), а наша часть:
что координаты пересчитаны тем же окном, каким ffmpeg реально режет кадр, что
дырка детекции видна потребителю, и что сглаживание давит дрожь, не запаздывая
на движении.
"""

from __future__ import annotations

import math

import pytest

from src.core.track import (
    REF_HEIGHT,
    REF_WIDTH,
    FaceTrack,
    OneEuro,
    TrackPoint,
    fill_gaps,
    resample,
    smooth,
)
from src.pipeline.crop import Detection, Segment, Track, project_track, window_at
from src.pipeline.probe import MediaInfo


def info(width: int = 1080, height: int = 1920, duration: float = 4.0) -> MediaInfo:
    return MediaInfo(
        path="v.mp4", duration=duration, width=width, height=height, fps=30.0,
        has_audio=True, audio_channels=2, audio_rate=48000, rotation=0,
        video_codec="h264", bitrate=0,
    )


def face(x: float, y: float, h: float = 300.0, score: float = 0.9) -> Detection:
    return Detection(x=x, y=y, score=score, area=h * h * 0.7, w=h * 0.8, h=h)


# ------------------------------------------------------------------ окно кропа


def test_window_is_the_whole_frame_when_there_is_no_crop_filter():
    """Окно во весь кадр — это ОТСУТСТВИЕ фильтра, а не `crop` с позицией из xs.

    `build_crop_filter` в этом случае не ставит `crop` вовсе, поэтому взять
    `seg.xs[0]` значило бы пересчитать трек по окну, которого в графе нет:
    там лежит зажатая клампом позиция, а кадр приходит целиком.
    """
    seg = Segment(0.0, 4.0, "wide", 1080, 1920, [0.0], [37], [12])
    assert window_at(seg, 1.0, frame_w=1080, frame_h=1920) == (0, 0, 1080, 1920)


def test_window_holds_its_value_until_the_next_command():
    """`sendcmd` — ступенька: значение держится от своей команды до следующей."""
    seg = Segment(0.0, 3.0, "wide", 600, 1066, [0.0, 1.0, 2.0], [10, 50, 90], [0, 4, 8])
    frame = {"frame_w": 1080, "frame_h": 1920}
    assert window_at(seg, 0.0, **frame)[:2] == (10, 0)
    assert window_at(seg, 0.99, **frame)[:2] == (10, 0)  # ещё не доехали до команды
    assert window_at(seg, 1.0, **frame)[:2] == (50, 4)
    assert window_at(seg, 1.5, **frame)[:2] == (50, 4)  # между командами не едем
    assert window_at(seg, 9.0, **frame)[:2] == (90, 8)  # после последней держим


# ------------------------------------------------------------------ пересчёт


def track_of(times: list[float], faces: list[Detection | None]) -> Track:
    return Track(
        times=times, faces=faces, motion_x=[None] * len(times),
        detector="yunet", analyzed=(640, 1138),
    )


def test_face_in_the_middle_of_the_window_lands_in_the_middle_of_the_frame():
    """Координаты нормализованные и от размера окна не зависят — только от места."""
    seg = Segment(0.0, 2.0, "wide", 600, 1066, [0.0], [200], [400])
    src = track_of([0.0, 0.5, 1.0], [face(200 + 300, 400 + 533)] * 3)
    out = project_track(src, [seg], info=info(duration=2.0), fps=30)
    assert out.points
    for p in out.points:
        assert p.cx == pytest.approx(0.5, abs=0.01)
        assert p.cy == pytest.approx(0.5, abs=0.01)


def test_the_same_face_reads_differently_in_a_tighter_window():
    """Крупность — свойство окна: в узком окне то же лицо занимает большую долю."""
    wide = Segment(0.0, 2.0, "wide", 600, 1066, [0.0], [200], [400])
    tight = Segment(0.0, 2.0, "tight", 400, 711, [0.0], [300], [600])
    src = track_of([0.0, 1.0], [face(500, 933, h=300)] * 2)
    a = project_track(src, [wide], info=info(duration=2.0), fps=30)
    b = project_track(src, [tight], info=info(duration=2.0), fps=30)
    assert b.points[0].h > a.points[0].h


def test_each_segment_uses_its_own_window():
    """Кроп меняется по кускам, значит и преобразование посегментное."""
    left = Segment(0.0, 1.0, "wide", 600, 1066, [0.0], [0], [0])
    right = Segment(1.0, 2.0, "tight", 600, 1066, [1.0], [480], [0])
    src = track_of([0.0, 0.5, 1.0, 1.5], [face(600, 500)] * 4)
    out = project_track(src, [left, right], info=info(duration=2.0), fps=30)
    first = out.at(0)
    last = out.at(out.points[-1].f)
    assert first is not None and last is not None
    # Лицо стоит на месте, а окно уехало вправо на 480 px — значит в кадре оно
    # должно уехать ВЛЕВО ровно на 480/600 ширины окна.
    assert first.cx - last.cx == pytest.approx(480 / 600, abs=0.05)


def test_track_is_empty_when_no_face_was_ever_found():
    """Пустой трек честнее выдуманного центра кадра: потребитель должен знать."""
    src = track_of([0.0, 0.5, 1.0], [None, None, None])
    out = project_track(src, [Segment(0.0, 1.0, "wide", 600, 1066, [0.0], [0], [0])],
                        info=info(duration=1.0), fps=30)
    assert out.points == []
    assert out.fps == 30


def test_a_gap_in_detection_drags_confidence_down():
    """Предохранители М2 обязаны узнать про потерю лица, а не про её длительность."""
    seg = Segment(0.0, 3.0, "wide", 600, 1066, [0.0], [0], [0])
    src = track_of(
        [0.0, 0.5, 1.0, 1.5, 2.0],
        [face(300, 500), face(300, 500), None, face(300, 500), face(300, 500)],
    )
    out = project_track(src, [seg], info=info(duration=2.5), fps=30)
    dip = min(p.conf for p in out.points)
    assert dip < 0.5, "провал уверенности в дырке не виден"
    assert max(p.conf for p in out.points) > 0.8, "уверенность просела там, где лицо было"
    # Геометрия при этом не рвётся: пропуск заполнен, а не обнулён.
    assert all(0.0 <= p.cx <= 1.0 for p in out.points)


def test_track_survives_json():
    src = track_of([0.0, 0.5], [face(300, 500)] * 2)
    out = project_track(
        src, [Segment(0.0, 1.0, "wide", 600, 1066, [0.0], [0], [0])],
        info=info(duration=1.0), fps=30,
    )
    again = FaceTrack.model_validate_json(out.model_dump_json())
    assert again.points == out.points
    assert again.analyzed == (640, 1138)


# ------------------------------------------------------------------ пропуски и пересчёт


def test_gaps_are_interpolated_inside_and_held_at_the_edges():
    assert fill_gaps([None, None, 2.0, None, 4.0, None]) == [2.0, 2.0, 2.0, 3.0, 4.0, 4.0]


def test_a_row_without_a_single_value_is_an_error_not_a_zero():
    with pytest.raises(ValueError):
        fill_gaps([None, None])


def test_resample_lands_between_the_samples():
    got = resample([0.0, 1.0], [0.0, 10.0], [0.0, 0.25, 0.5, 1.0, 2.0])
    assert got == pytest.approx([0.0, 2.5, 5.0, 10.0, 10.0])


# ------------------------------------------------------------------ сглаживание


def test_one_euro_kills_jitter_on_a_static_shot():
    """DoD М1: на статичном плане дрожание центра меньше 2 px.

    Шум ±3 px на кадре 1080 — это ±0.0028 доли; фильтр обязан увести соседнюю
    разницу ниже 2 px, то есть ниже 0.00185.
    """
    noise = [0.5 + (0.003 if i % 2 else -0.003) for i in range(90)]
    got = smooth(noise, fps=30, min_cutoff=1.0, beta=0.007, scale=REF_WIDTH)
    jitter = max(abs(b - a) for a, b in zip(got[10:], got[11:]))
    assert jitter * REF_WIDTH < 2.0, f"дрожание {jitter * REF_WIDTH:.2f} px"


def test_one_euro_does_not_lag_behind_real_movement():
    """На быстром движении фильтр обязан почти не вмешиваться.

    Иначе он ничем не лучше скользящего среднего, ради отказа от которого
    и выбран: голова поехала — срез поднимается — задержка падает.
    """
    ramp = [i / 30 for i in range(60)]  # проезд через кадр за две секунды
    got = smooth(ramp, fps=30, min_cutoff=1.0, beta=0.007, scale=REF_HEIGHT)
    lag = ramp[-1] - got[-1]
    assert lag * REF_HEIGHT < 40, f"отставание {lag * REF_HEIGHT:.0f} px"


def test_beta_is_calibrated_in_pixels_not_in_fractions():
    """Без нормировки `beta` из MOTION_PLAN не работает вовсе.

    `beta` умножается на измеренную СКОРОСТЬ, то есть привязан к единицам
    сигнала. В долях кадра скорость выходит порядка 1 в секунду вместо ~2000
    px/с, срез поднимается на 0.007 Гц вместо 14, и фильтр перестаёт уступать
    движению — то есть становится обычным ФНЧ, ради отказа от которого он и
    выбран. Замер и есть тест: та же дорожка, разница только в масштабе.
    """
    ramp = [i / 30 for i in range(60)]
    raw = ramp[-1] - smooth(ramp, fps=30, min_cutoff=1.0, beta=0.007)[-1]
    scaled = ramp[-1] - smooth(ramp, fps=30, min_cutoff=1.0, beta=0.007, scale=REF_HEIGHT)[-1]
    assert raw * REF_HEIGHT > 200, "без нормировки отставание обязано быть большим"
    assert scaled < raw / 5


def test_one_euro_raises_its_cutoff_with_speed():
    """Сердце фильтра: срез ведёт скорость, а не время. Это и отличает его от среднего."""
    slow = OneEuro(30, min_cutoff=1.0, beta=0.007)
    fast = OneEuro(30, min_cutoff=1.0, beta=0.007)
    slow(0.0)
    fast(0.0)
    slow_out = slow(0.001)
    fast_out = fast(0.5)
    # Медленный шаг фильтр почти съедает, быстрый — почти пропускает.
    assert slow_out / 0.001 < fast_out / 0.5


def test_one_euro_needs_a_positive_frequency():
    with pytest.raises(ValueError):
        OneEuro(0)


def test_smoothing_is_a_pure_function_of_the_input():
    """Тот же ряд — тот же результат: трек обязан быть воспроизводимым."""
    row = [math.sin(i / 5) * 0.1 + 0.5 for i in range(50)]
    assert smooth(row, fps=30) == smooth(row, fps=30)


def test_track_lookup_is_by_frame_number():
    t = FaceTrack(fps=30, points=[TrackPoint(f=i, cx=0.5, cy=0.5, h=0.2, conf=1.0) for i in range(5)])
    assert t.at(3) is not None and t.at(3).f == 3
    assert t.at(99) is None
