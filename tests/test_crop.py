"""Тесты автокропа (M6): математика окна и трека, цветокор, граф ffmpeg.

Главное, что здесь проверяется: крупность выводится из ДОЛИ ЛИЦА и никогда не
добирается апскейлом, макушка не срезается, а смена крупности происходит ровно
на резе. Всё это чистые функции — реальный трекинг вынесен в slow-тесты.

Отдельный слой — тесты, привязанные к ЗАМЕРУ, а не к реализации: растр анализа
закреплён числами по пропорциям и его расхождение с каноном `tests/face_share.py`
измерено на эталоне (slow), а оценка макушки обязана лежать выше макушек, снятых
с реальных кадров. Такой тест ломается, когда правят константу «на глаз», и это
его работа. Обратное тоже верно: тест, который повторяет последнюю строку
проверяемой функции, не проверяет ничего — см. историю
`test_analyze_raster_is_pinned_per_aspect`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.core.edl import EDL, Cut, Word
from src.core.preset import Crop, Grade, load_preset
from src.pipeline.crop import (
    ANALYZE_HEIGHT,
    _log_framing,
    ANALYZE_MIN_HEIGHT,
    ANALYZE_PIXELS,
    HEAD_ABOVE_BOX,
    VIGNETTE_VISIBLE,
    Detection,
    Segment,
    Track,
    analyze_scale,
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
    pick_face,
    place_window,
    plan_segments,
    warn_missing_inputs,
    window_bounds,
    window_height,
)
from src.pipeline.probe import MediaInfo

CROP = Crop()  # значения по умолчанию совпадают с пресетами

# Замер по samples/phase-0-source.mp4 на каноничном растре 640x1138: бокс YuNet,
# его ландмарки и макушка, снятая с кадра по пиксельной линейке (пять кадров:
# 1, 5, 10, 15, 19 с). Числа — вход для проверки правила «макушка не обрезана»;
# на них же калибровались константы в crop.py.
#
# Границы этой выборки: один человек, одна причёска, пять кадров, макушка снята
# рукой с округлением до 5 px. HEAD_ABOVE_EYES = 1.6 взята как «максимум по этим
# пяти кадрам плюс чуть-чуть», и на худшем кадре (t=15) запас составляет 6.7 px —
# меньше, чем точность самого замера. Поэтому тесты ниже проверяют не «правило
# откалибровано», а «правило не срезает НИ ОДНУ из ЭТИХ пяти макушек», и запас
# записан числом, чтобы его тонкость была видна. Расширять выборку — вторым и
# третьим человеком, а не подкруткой константы.
HEAD_SAMPLES: tuple[tuple[float, float, float, float, float], ...] = (
    # (верх бокса, высота бокса, линия глаз, линия рта, макушка)
    (537.1, 496.1, 600.5, 853.1, 300.0),
    (423.0, 616.8, 578.9, 853.4, 300.0),
    (544.1, 544.4, 603.3, 865.1, 325.0),
    (582.4, 463.5, 664.9, 884.7, 320.0),
    (420.9, 579.5, 612.4, 865.0, 330.0),
)


def info_of(width: int, height: int, duration: float = 20.0) -> MediaInfo:
    return MediaInfo(
        path="—", duration=duration, width=width, height=height, fps=30.0,
        has_audio=True, audio_channels=2, audio_rate=48000, rotation=0,
        video_codec="h264", bitrate=0,
    )


def face(y: float, h: float, x: float = 0.0) -> Detection:
    """Лицо с боксом высоты h и центром в (x, y). Линия глаз — по боксу."""
    return Detection(x=x, y=y, score=0.9, area=h * h * 0.75, w=h * 0.75, h=h)


def measured_face(top: float, h: float, eye: float, mouth: float) -> Detection:
    """Лицо из замера: бокс плюс настоящие ландмарки."""
    return Detection(
        x=0.0, y=top + h / 2, score=0.9, area=h * h * 0.75, w=h * 0.75, h=h,
        eye_y=eye, mouth_y=mouth,
    )


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
    assert out[4] == 135.0  # люфт: цель минус порог, а не полная новая позиция


def test_deadzone_never_moves_faster_than_the_target():
    """Мёртвая зона гасит дрожь, но не создаёт скачков.

    Прежняя версия при срабатывании переставляла окно сразу на полную новую
    позицию, то есть добавляла к движению цели прыжок на целый порог: в реальном
    прогоне из sendcmd это 14 px за один кадр на окне 1214 px. Это тот же артефакт
    «ступеньки», ради запрета которого в Ф6 запрещён зум внутри куска, поэтому
    проверяем не «камера стоит», а «камера не обгоняет цель».
    """
    raw = [100.0] * 5 + [100.0 + i for i in range(1, 30)]
    out = apply_deadzone(raw, threshold=5.0)
    steps_out = [abs(b - a) for a, b in zip(out, out[1:])]
    steps_raw = [abs(b - a) for a, b in zip(raw, raw[1:])]
    assert max(steps_out) <= max(steps_raw) + 1e-9
    assert max(abs(v - r) for v, r in zip(out, raw)) <= 5.0 + 1e-9  # лаг не больше порога


def test_deadzone_does_not_drift_on_slow_ramp():
    """Медленный сдвиг внутри порога камеру двигать не должен."""
    raw = [100.0 + i * 0.5 for i in range(20)]
    out = apply_deadzone(raw, threshold=5.0)
    assert len(set(out)) < len(set(raw))
    assert out[:11] == [100.0] * 11  # пока цель не ушла за порог, окно стоит


def test_detection_uses_landmarks_for_eye_line():
    with_marks = Detection(x=0, y=100, score=1.0, area=1.0, w=50, h=100, eye_y=70.0)
    without = Detection(x=0, y=100, score=1.0, area=1.0, w=50, h=100)
    assert with_marks.eyes == 70.0
    assert without.eyes == pytest.approx(50 + 0.2 * 100)  # верх бокса + доля высоты


def test_detection_rescale_keeps_ratios():
    d = Detection(
        x=10, y=20, score=1.0, area=100.0, w=10, h=10, eye_y=18.0, mouth_y=25.0
    ).rescaled(2.0)
    assert (d.x, d.y, d.w, d.h, d.eye_y, d.mouth_y) == (20, 40, 20, 20, 36.0, 50.0)
    assert d.area == 400.0


def test_track_medians_ignore_missing_faces():
    t = track_of([face(500, 300), None, face(500, 500), face(500, 400)])
    assert t.face_height() == 400.0


# ------------------------------------------------------------------ макушка (замер)


def test_head_top_from_landmarks_is_above_every_measured_crown():
    """Оценка макушки не срезает ни одну из пяти макушек, снятых с эталона."""
    for top, h, eye, mouth, crown in HEAD_SAMPLES:
        d = measured_face(top, h, eye, mouth)
        assert d.head_top <= crown, (top, h, d.head_top, crown)


def test_head_margin_on_the_worst_measured_frame_is_thin():
    """Запас правила над снятой макушкой — числом, а не «с запасом».

    На худшем кадре выборки он меньше 10 px на растре высотой 1138, то есть в
    пределах округления, с которым макушка снималась (5 px). Это не повод крутить
    константу — это повод не называть её калибровкой: на другой причёске или в
    шапке правило поедет, и увидят это глазами, а не тестом.
    """
    margins = [
        crown - measured_face(top, h, eye, mouth).head_top
        for top, h, eye, mouth, crown in HEAD_SAMPLES
    ]
    assert min(margins) > 0  # ни одна макушка не срезана
    assert min(margins) < 10  # но запас тонкий — выборка n=1 человек, 5 кадров


def test_head_top_from_box_is_above_every_measured_crown():
    """Запасная оценка (без ландмарок) тоже обязана быть с запасом, а не примерно."""
    for top, h, eye, mouth, crown in HEAD_SAMPLES:
        d = face(y=top + h / 2, h=h)
        assert d.head_top <= crown, (top, h, d.head_top, crown)


def test_old_uncalibrated_head_constant_would_cut_the_crown():
    """Почему правило переписано: 0.4 высоты бокса — это не запас, а лотерея.

    Константа 0.4 стояла со ссылкой на множитель 1.4 из tests/face_share.py,
    который там сам помечен как некалиброванный. На замеренных кадрах требуемая
    доля гуляет от 0.157 до 0.566, то есть 0.4 то срезает макушку, то оставляет
    над ней лишнюю десятую кадра.
    """
    cut = [
        crown
        for top, h, _eye, _mouth, crown in HEAD_SAMPLES
        if top - 0.4 * h > crown
    ]
    assert cut, "калибровка сдвинулась — перепроверьте замер, а не константу"


def test_landmark_head_estimate_is_steadier_than_the_box_one():
    """Ландмарки выбраны не по вкусу: их разброс на тех же кадрах меньше.

    Бокс YuNet на одном и том же лице гуляет по высоте на ±15 %, пара
    «глаза-рот» — плоскостной размер лица, он к дрожи бокса не привязан.
    """
    def spread(errors: list[float]) -> float:
        return max(errors) - min(errors)

    by_marks = [
        measured_face(top, h, eye, mouth).head_top - crown
        for top, h, eye, mouth, crown in HEAD_SAMPLES
    ]
    by_box = [
        top - HEAD_ABOVE_BOX * h - crown for top, h, _e, _m, crown in HEAD_SAMPLES
    ]
    assert spread(by_marks) < spread(by_box)


def test_head_top_falls_back_to_box_when_mouth_is_missing():
    """Половина ландмарок — не ландмарки: правило переключается целиком."""
    only_eyes = Detection(x=0, y=100, score=1.0, area=1.0, w=50, h=100, eye_y=70.0)
    assert only_eyes.head_top == pytest.approx(50 - HEAD_ABOVE_BOX * 100)


# ------------------------------------------------------------------ растр анализа


REFERENCE_CLIP = Path(__file__).resolve().parents[1] / "samples" / "phase-0-source.mp4"


def _measured_shares(canvas: tuple[int, int], face_h: int, probes: int = 8):
    """Медианная доля лица на растре кропа и на каноне face_share.py (h=1138).

    Кадры эталона масштабируются до заданной высоты и добиваются полями до нужной
    пропорции — поля не меняют долю по высоте, зато меняют растр анализа, а это и
    есть предмет замера. Никакого ffmpeg: только те же cv2-операции, что делает сам
    трекер.
    """
    import statistics

    import cv2
    import numpy as np

    from src.pipeline.crop import _resize, make_detector

    if not REFERENCE_CLIP.exists():
        pytest.skip(f"нет эталона {REFERENCE_CLIP}")

    cap = cv2.VideoCapture(str(REFERENCE_CLIP))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or probes
    frames = []
    for k in range(probes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 0.5) / probes))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    assert frames, "эталон не читается"

    detector = make_detector()
    cw, ch = canvas
    on_crop: list[float] = []
    on_canon: list[float] = []
    for frame in frames:
        k = face_h / frame.shape[0]
        img = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        board = np.zeros((ch, cw, 3), np.uint8)
        top, left = max(0, (ch - h) // 2), max(0, (cw - w) // 2)
        board[top:top + min(h, ch), left:left + min(w, cw)] = img[:min(h, ch), :min(w, cw)]

        for bucket, scale in (
            (on_crop, analyze_scale(cw, ch)),
            (on_canon, ANALYZE_HEIGHT / ch),
        ):
            small = _resize(board, scale)
            found = [d for d in detector.detect(small) if d.h > 0]
            if found:
                bucket.append(max(found, key=lambda d: d.area).h / small.shape[0])

    assert on_crop and on_canon, "детектор не нашёл лица на эталоне"
    return statistics.median(on_crop), statistics.median(on_canon)


def test_analyze_raster_matches_face_share_on_vertical_input():
    """9:16 — основной вход, и на нём растр обязан совпасть с каноном замера.

    tests/face_share.py нормирует кадр по ВЫСОТЕ 1138. Если крупность целится по
    одному растру, а DoD меряется по другому, доля лица в логе — про другое число.
    """
    scale = analyze_scale(720, 1280)
    assert round(1280 * scale) == ANALYZE_HEIGHT
    assert round(720 * scale) == 640


@pytest.mark.parametrize(
    "w,h", [(1920, 1080), (3840, 2160), (1080, 1350), (720, 1280), (480, 854)]
)
def test_analyze_raster_stays_within_pixel_budget(w, h):
    """Потолок площади — цена детекции: YuNet линеен по пикселям."""
    scale = analyze_scale(w, h)
    assert scale * scale * w * h <= ANALYZE_PIXELS * 1.001


def test_analyze_raster_pays_extra_for_ultrawide_input():
    """21:9 — единственный случай, где площадь уступает: иначе лицо вдвое мельче."""
    scale = analyze_scale(2560, 1080)
    assert round(1080 * scale) == ANALYZE_MIN_HEIGHT
    assert scale * scale * 2560 * 1080 <= ANALYZE_PIXELS * 1.6


@pytest.mark.parametrize(
    "w,h,raster",
    [
        (1920, 1080, (1138, 640)),
        (3840, 2160, (1138, 640)),
        (1280, 720, (1138, 640)),
        (1080, 1350, (763, 954)),
        (1080, 1080, (853, 853)),
        (2560, 1080, (1349, 569)),
        (720, 1280, (640, 1138)),
        (1080, 1920, (640, 1138)),
    ],
)
def test_analyze_raster_is_pinned_per_aspect(w, h, raster):
    """Растр закреплён числом, а не свойством «не хуже половины канона».

    Прежний тест носил имя про нормировку по высоте, но проверял
    `round(h * analyze_scale(w, h)) >= ANALYZE_HEIGHT / 2` — то есть буквально
    последнюю строку самой функции (`max(scale, ANALYZE_MIN_HEIGHT / height)`) и не
    мог упасть ни на каком входе: перебор 200 000 случайных пар (w, h) даёт ноль
    контрпримеров. При этом настоящее свойство — высота 1138 — держится ТОЛЬКО на
    9:16: на всех остальных пропорциях выигрывает потолок площади. Здесь это
    записано числами, поэтому любое изменение правила растра теперь видно.
    """
    scale = analyze_scale(w, h)
    assert (round(w * scale), round(h * scale)) == raster


def test_analyze_raster_matches_canon_only_on_vertical_input():
    """Канон h=1138 достаётся вертикали и никому больше — это цена потолка площади."""
    canon = [(w, h) for w, h in ((720, 1280), (1080, 1920), (1920, 1080), (1080, 1080))
             if round(h * analyze_scale(w, h)) == ANALYZE_HEIGHT]
    assert canon == [(720, 1280), (1080, 1920)]


def test_analyze_raster_upscales_small_sources():
    """Мелкий исходник тоже приводится к растру, а не анализируется нативно.

    Ранний выход «меньше запрошенного — оставим как есть» означал бы, что растр
    зависит от разрешения входа, а вместе с ним и доля лица.
    """
    assert analyze_scale(480, 854) > 1.0


def test_analyze_raster_survives_degenerate_size():
    assert analyze_scale(0, 0) == 1.0


@pytest.mark.slow
@pytest.mark.parametrize(
    "canvas,face_h,bound",
    [
        ((1920, 1080), 518, 0.10),  # 16:9, доля лица ≈ 0.29 — рабочая точка DoD
        ((1080, 1080), 520, 0.10),  # 1:1, та же доля
        ((1080, 1920), 1920, 0.02),  # 9:16 — растр тот же самый, расхождения нет
    ],
)
def test_analyze_raster_divergence_from_canon(canvas, face_h, bound):
    """Насколько «доля лица» на растре кропа расходится с каноном face_share.py.

    Утверждение в докстринге модуля («совпадает с каноном, расхождение ~2 %») было
    неверным: потолок площади уводит растр с 1138 на всём, кроме 9:16. Здесь
    расхождение МЕРЯЕТСЯ на реальных кадрах эталона, добитых полями до нужной
    пропорции, а не выводится из формулы. В коридоре DoD (доля лица около 0.29,
    именно там живёт цель 0.22-0.28) оно обязано оставаться в пределах 10 %.

    Крупный план — отдельный случай, см. `test_analyze_raster_diverges_on_big_face`.
    """
    crop_share, canon_share = _measured_shares(canvas, face_h)
    assert abs(crop_share / canon_share - 1) <= bound, (crop_share, canon_share)


@pytest.mark.slow
def test_analyze_raster_diverges_on_big_face():
    """А вот на лице во весь кадр расхождение большое — и это надо знать.

    Тест закрепляет не желаемое, а измеренное: числа из докстринга модуля. Если
    потолок площади когда-нибудь снимут, тест упадёт — и вместе с ним придётся
    поправить обещание в докстринге, а не наоборот.
    """
    crop_share, canon_share = _measured_shares((1920, 1080), 1080)
    assert crop_share / canon_share - 1 > 0.10, (crop_share, canon_share)


# ------------------------------------------------------------------ выбор лица


def test_pick_face_takes_the_largest_without_history():
    small, big = face(y=100, h=100, x=100), face(y=100, h=300, x=900)
    assert pick_face([small, big], None) is big


def test_pick_face_keeps_tracking_the_same_person():
    """Двое в кадре: окно не должно скакать между ними от пробы к пробе."""
    tracked = face(y=500, h=300, x=300)
    rival = face(y=500, h=330, x=1400)  # чуть крупнее, но это другой человек
    assert pick_face([rival, tracked], tracked) is tracked


def test_pick_face_yields_to_a_much_bigger_face():
    """Если в кадр реально вошёл человек вдвое крупнее — переключаемся."""
    tracked = face(y=500, h=200, x=300)
    newcomer = face(y=500, h=500, x=1400)
    assert pick_face([tracked, newcomer], tracked) is newcomer


def test_pick_face_on_empty_frame_is_none():
    assert pick_face([], None) is None


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


def test_normalize_spans_stretches_onto_the_frame_grid():
    """Границы крупности обязаны совпасть с резами в cut.mp4, а не с суммой EDL.

    `framing_spans` складывает точные длительности EDL, а `cut.py` режет каждый
    кусок фильтром `fps=` — то есть округляет его до целого числа кадров, и ошибка
    копится по числу резов. На эталонном EDL три куска (4.565 + 6.225 + 7.68 =
    18.47 с) дают в cut.mp4 445 кадров при 24 fps, то есть 18.542 с. Раньше всю
    разницу съедал последний span, а внутренние границы оставались ранними, и
    смена крупности приходилась на несколько кадров ВНУТРЬ предыдущего плана —
    ровно тот самопроизвольный зум, ради запрета которого крупность и меняется
    только на резах.
    """
    spans = [(0.0, 4.565, "tight"), (4.565, 10.79, "wide"), (10.79, 18.47, "tight")]
    fitted = normalize_spans(spans, 18.542)
    assert fitted[-1][1] == pytest.approx(18.542)
    k = 18.542 / 18.47
    assert fitted[0][1] == pytest.approx(4.565 * k, abs=1e-6)
    assert fitted[1][1] == pytest.approx(10.79 * k, abs=1e-6)
    # Ошибка каждой внутренней границы — доли кадра, а не полтора кадра, как было.
    assert abs(fitted[0][1] - 4.565) * 24 < 1.0


def test_normalize_spans_does_not_stretch_a_foreign_timeline():
    """Растягиваем только то, что похоже на округление кадров, а не любой разрыв."""
    spans = [(0.0, 2.0, "tight"), (2.0, 4.0, "wide")]
    got = normalize_spans(spans, 20.0)
    assert got[0] == (0.0, 2.0, "tight")
    assert got[-1] == (2.0, 20.0, "wide")  # хвост отдан последней крупности, не растянут


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


def test_plan_segments_merge_windows_that_differ_below_perception():
    """Окна, различающиеся на проценты, — одна ветка графа, а не две.

    На 4K-входе кламп `max_scale` сводит wide и tight почти в одно окно: замер
    реального прогона дал 1214x2158 против 1204x2140, то есть 0.8 % по масштабу и
    0.002 по доле лица. Раньше `_merge_segments` требовал побитового равенства, и
    ради невидимой разницы граф уходил в split/trim/concat — лишний стык и лишний
    риск (именно этот путь до правки падал на разном SAR).
    """
    preset = load_preset("glass")
    info = info_of(3840, 2160, duration=20.0)
    track = track_of([face(y=1000.0, h=0.28 * 2160, x=1920.0) for _ in range(120)])
    segments = plan_segments(
        track, [(0.0, 10.0, "wide"), (10.0, 20.0, "tight")],
        info=info, out_w=1080, out_h=1920, preset=preset,
    )
    assert len(segments) == 1, [(s.framing, s.win_w, s.win_h) for s in segments]
    assert segments[0].end == pytest.approx(20.0)


def test_plan_segments_keep_windows_that_differ_enough_to_see():
    """Порог склейки не должен съедать настоящую смену крупности."""
    preset = load_preset("glass")
    info = info_of(5120, 2880, duration=12.0)
    track = track_of([face(y=1400.0, h=580.0, x=2560.0) for _ in range(72)])
    segments = plan_segments(
        track, [(0.0, 6.0, "wide"), (6.0, 12.0, "tight")],
        info=info, out_w=1080, out_h=1920, preset=preset,
    )
    assert len(segments) == 2
    ratio = segments[0].win_h / segments[1].win_h
    assert ratio > 1.04  # различимая разница, ради неё ветка и заводится


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


def test_plan_segments_keep_the_directors_own_plan_on_a_short_clip(caplog):
    """Потолок мельтешения не имеет права бить по плану самого режиссёра.

    Он считает КОЛИЧЕСТВО кусков, а `director.FRAMING_MIN_HOLD` гарантирует
    ИНТЕРВАЛ между сменами: n кусков занимают n−1 интервал. Без этой поправки
    ролик на 11.7 с с тремя крупностями (реальный выход `assign_framing` на
    нарезке [4.8, 4.77, 2.11]) схлопывался в одно окно — то есть Ф6 выключалась
    именно на 10-20-секундных роликах, на основном формате.
    """
    preset = load_preset("glass")
    info = info_of(5120, 2880, duration=11.68)
    track = track_of([face(y=1400.0, h=580.0, x=2560.0) for _ in range(71)])
    spans = [(0.0, 4.8, "tight"), (4.8, 9.57, "wide"), (9.57, 11.68, "tight")]
    with caplog.at_level(logging.WARNING, logger="src.pipeline.crop"):
        segments = plan_segments(
            track, spans, info=info, out_w=1080, out_h=1920, preset=preset
        )
    assert len(segments) == 3
    assert not [r for r in caplog.records if "свожу к одному окну" in r.getMessage()]


# --------------------------------------------------- план кропа: реальные размеры


def test_face_ratio_is_inert_below_output_height():
    """Честная граница Ф6: на входе ниже 1920 px доля лица не регулируется вовсе.

    `max_scale=1.0` требует окно не ниже высоты выхода, а сузить кадр — значит
    сделать окно МЕНЬШЕ. На 1920x1080 (и на любом 720p-селфи) оба предела сходятся
    в высоту кадра, и `face_height_ratio` не может изменить ничего: крупность
    остаётся ровно такой, какой её снял пользователь. Тест закрепляет ограничение,
    чтобы «доля лица 0.45 при цели 0.22» в логе не выглядела ошибкой арифметики.
    """
    preset = load_preset("glass")
    info = info_of(1920, 1080, duration=12.0)
    track = track_of([face(y=500.0, h=430.0, x=960.0) for _ in range(72)])
    segments = plan_segments(
        track, [(0.0, 6.0, "wide"), (6.0, 12.0, "tight")],
        info=info, out_w=1080, out_h=1920, preset=preset,
    )
    assert {s.win_h for s in segments} == {1080}
    assert len(segments) == 1  # крупность одна на весь ролик — менять нечего


def test_four_k_source_moves_face_share_toward_the_target():
    """4K 16:9: запас на зум есть, но всего 2160/1920 = 1.125 — цель не достаётся.

    Числа с реального прогона (фикстура из эталона, лицо ≈ 0.20 кадра): окно
    ужимается с 2160 до 1948-2090 px, доля растёт примерно на 10 %. Проверяем
    именно направление и предел, а не выдуманную «попали в 0.22».
    """
    preset = load_preset("glass")
    info = info_of(3840, 2160, duration=12.0)
    face_h = 0.20 * 2160
    track = track_of([face(y=1000.0, h=face_h, x=1920.0) for _ in range(72)])
    segments = plan_segments(
        track, [(0.0, 12.0, "tight")], info=info, out_w=1080, out_h=1920, preset=preset,
    )
    seg = segments[0]
    assert seg.win_h < 2160  # зум произошёл
    assert seg.win_h >= 1920  # но не глубже клампа апскейла
    assert seg.face_share(face_h) > 0.20
    assert seg.face_share(face_h) < preset.crop.tight_face_height_ratio  # цель не достигнута


def test_segment_carries_its_own_face_height():
    """Доля лица в логе считается по лицу ЭТОГО куска, а не по медиане ролика.

    Иначе на ролике, где человек подошёл к камере, лог показывает одну и ту же
    долю на всех кусках, и «крупность 0.22» перестаёт означать что-либо.
    """
    preset = load_preset("glass")
    info = info_of(5120, 2880, duration=12.0)
    near = [face(y=1400.0, h=620.0, x=2560.0) for _ in range(36)]
    far = [face(y=1400.0, h=500.0, x=2560.0) for _ in range(36)]
    segments = plan_segments(
        track_of(near + far), [(0.0, 6.0, "wide"), (6.0, 12.0, "wide")],
        info=info, out_w=1080, out_h=1920, preset=preset,
    )
    assert len(segments) == 2
    assert segments[0].face_h == 620.0 and segments[1].face_h == 500.0
    for s in segments:
        assert s.face_share(s.face_h) == pytest.approx(preset.crop.face_height_ratio, abs=0.02)


# ------------------------------------------------------------------ граничные случаи


def test_plan_segments_without_any_face():
    """Лиц нет ни на одном кадре: окно максимально широкое и стоит внутри кадра."""
    preset = load_preset("glass")
    info = info_of(1920, 1080, duration=6.0)
    track = track_of([None] * 36)
    segments = plan_segments(
        track, [(0.0, 6.0, "wide")], info=info, out_w=1080, out_h=1920, preset=preset,
    )
    assert len(segments) == 1
    seg = segments[0]
    assert seg.win_h == 1080 and seg.face_h == 0.0
    assert all(0 <= x <= info.width - seg.win_w for x in seg.xs)
    assert all(0 <= y <= info.height - seg.win_h for y in seg.ys)


def test_plan_segments_without_face_centers_the_window_vertically():
    """Без лиц окно обязано встать ПО ЦЕНТРУ — как обещают комментарий и лог.

    Прежние запасные ряды (eyes = 0.36·H, head_top = 0.05·H) описывали лицо во
    весь кадр, поэтому правило макушки в `place_window` выигрывало и тащило окно
    вверх: на 1080x2340 оно вставало на y=21 при центре 210, то есть у любого
    современного телефона (19.5:9, 20:9) терялся низ кадра. Старый тест поймать
    это не мог: он гонялся на 1920x1080, где окно равно высоте кадра и y может
    быть только нулём.
    """
    preset = load_preset("glass")
    info = info_of(1080, 2340, duration=6.0)
    track = track_of([None] * 36)
    seg = plan_segments(
        track, [(0.0, 6.0, "wide")], info=info, out_w=1080, out_h=1920, preset=preset,
    )[0]
    assert seg.win_h < info.height  # ход по вертикали есть, иначе тест бессмыслен
    assert set(seg.ys) == {(info.height - seg.win_h) // 2}


def test_plan_segments_without_face_still_follows_motion_horizontally():
    """По горизонтали центр внимания есть и без лиц — центр масс движения."""
    preset = load_preset("glass")
    info = info_of(1920, 1080, duration=6.0)
    track = Track(
        times=[i / 6 for i in range(36)],
        faces=[None] * 36,
        motion_x=[500.0 + i * 8 for i in range(36)],
        detector="test",
    )
    seg = plan_segments(
        track, [(0.0, 6.0, "wide")], info=info, out_w=1080, out_h=1920, preset=preset,
    )[0]
    assert max(seg.xs) - min(seg.xs) > 100  # окно едет за движением, а не стоит


def test_log_tells_motion_from_a_static_center(caplog):
    """Лог не имеет права называть «центром кадра» ход за движением.

    В единственной ветке, которую описывала эта строка, она говорила неправду:
    лиц нет, окно едет за центром масс движения (замер на клипе без лиц: xs от
    507 до 655 при центре кадра 656), а в логе стояло «окно стоит по центру
    кадра». Теперь ветки две и обе называются своими именами.
    """
    preset = load_preset("glass")
    info = info_of(1920, 1080, duration=6.0)
    moving = Track(
        times=[i / 6 for i in range(36)], faces=[None] * 36,
        motion_x=[500.0 + i * 8 for i in range(36)], detector="test",
    )
    still = track_of([None] * 36)

    for track, expected in ((moving, "центр масс движения"), (still, "движения не найдено")):
        segments = plan_segments(
            track, [(0.0, 6.0, "wide")], info=info, out_w=1080, out_h=1920, preset=preset,
        )
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="src.pipeline.crop"):
            _log_framing(track, segments, preset.crop, info=info)
        assert any(expected in r.getMessage() for r in caplog.records), caplog.text


def test_plan_segments_on_an_empty_track_centers_the_window():
    """Проб нет вовсе: окно по центру кадра, без зашитых в код долей."""
    preset = load_preset("glass")
    info = info_of(1080, 2340, duration=6.0)
    seg = plan_segments(
        Track(detector="test"), [(0.0, 6.0, "wide")],
        info=info, out_w=1080, out_h=1920, preset=preset,
    )[0]
    assert seg.xs == [(info.width - seg.win_w) // 2]
    assert seg.ys == [(info.height - seg.win_h) // 2]


def test_plan_segments_with_a_single_sample():
    """Кусок короче шага семплирования — берём ближайшую пробу, а не падаем."""
    preset = load_preset("glass")
    info = info_of(3840, 2160, duration=0.2)
    segments = plan_segments(
        track_of([face(y=1000.0, h=500.0, x=1920.0)]), [(0.0, 0.2, "wide")],
        info=info, out_w=1080, out_h=1920, preset=preset,
    )
    assert len(segments) == 1 and segments[0].xs and segments[0].ys


def test_plan_segments_with_face_at_the_frame_edge():
    """Лицо у самого края: окно прижимается к границе, но не выходит за неё."""
    preset = load_preset("glass")
    info = info_of(3840, 2160, duration=6.0)
    track = track_of([face(y=180.0, h=300.0, x=20.0) for _ in range(36)])
    segments = plan_segments(
        track, [(0.0, 6.0, "wide")], info=info, out_w=1080, out_h=1920, preset=preset,
    )
    seg = segments[0]
    assert min(seg.xs) == 0 and min(seg.ys) == 0
    assert all(0 <= x <= info.width - seg.win_w for x in seg.xs)
    assert all(0 <= y <= info.height - seg.win_h for y in seg.ys)


def test_plan_segments_on_vertical_input_never_zooms():
    """Вертикальный вход зумиться не должен — окно равно кадру (QUALITY_PLAN Ф6)."""
    preset = load_preset("glass")
    info = info_of(1080, 1920, duration=6.0)
    track = track_of([face(y=800.0, h=700.0, x=540.0) for _ in range(36)])
    segments = plan_segments(
        track, [(0.0, 6.0, "tight")], info=info, out_w=1080, out_h=1920, preset=preset,
    )
    assert (segments[0].win_w, segments[0].win_h) == (1080, 1920)


def test_plan_segments_on_ultrawide_input():
    """21:9: окно узкое, по высоте — во весь кадр, и оно внутри кадра."""
    preset = load_preset("glass")
    info = info_of(2560, 1080, duration=6.0)
    track = track_of([face(y=500.0, h=300.0, x=2000.0) for _ in range(36)])
    seg = plan_segments(
        track, [(0.0, 6.0, "wide")], info=info, out_w=1080, out_h=1920, preset=preset,
    )[0]
    assert seg.win_h == 1080
    assert seg.win_w == pytest.approx(1080 * 9 / 16, abs=2)
    assert all(0 <= x <= info.width - seg.win_w for x in seg.xs)


def test_track_switching_faces_does_not_move_the_window():
    """Двое в кадре: пока трек держится за одного, окно не мечется между ними."""
    preset = load_preset("glass")
    info = info_of(3840, 2160, duration=6.0)
    left, right = face(y=1000.0, h=400.0, x=800.0), face(y=1000.0, h=420.0, x=3000.0)
    kept = [pick_face([left, right], None)]
    for _ in range(35):
        kept.append(pick_face([left, right], kept[-1]))
    seg = plan_segments(
        track_of(kept), [(0.0, 6.0, "wide")],
        info=info, out_w=1080, out_h=1920, preset=preset,
    )[0]
    assert len(set(seg.xs)) == 1


# ------------------------------------------------------------------ цветокор (Ф3)


def test_grade_filter_from_preset():
    """Фильтр собирается из данных пресета, а не из чисел, зашитых в тест.

    Числа в JSON — предмет настройки: виньетку уже опускали с 0.63 до 0.22,
    когда замер показал, что на 9:16 она гасит макушку и полосу субтитров.
    Тест, повторяющий значение, ловит правку данных, а не поломку кода.
    """
    grade = load_preset("glass").grade
    got = grade_filter(grade)
    assert got.startswith(
        f"eq=contrast={grade.contrast:g}:saturation={grade.saturation:g}:gamma={grade.gamma:g}"
    )
    assert f"vignette=angle={grade.vignette:g}" in got


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


def test_grade_warns_when_saturation_reddens_skin(caplog):
    """Пресет с перебором по насыщенности собирается, но молча — не должен.

    Краснота на коже с одного кадра не читается, а на готовом ролике уже поздно;
    единственное дешёвое лекарство — строчка в логе на этапе, где фильтр собран.
    """
    with caplog.at_level(logging.WARNING, logger="src.pipeline.crop"):
        got = grade_filter(Grade(saturation=1.35))
    assert "saturation=1.35" in got  # поведение не меняем, только предупреждаем
    assert any("красноту" in r.getMessage() for r in caplog.records)


def test_grade_stays_quiet_at_the_threshold(caplog):
    with caplog.at_level(logging.WARNING, logger="src.pipeline.crop"):
        grade_filter(Grade(saturation=1.2))
    assert not caplog.records


def test_grade_warns_when_vignette_is_a_visible_frame(caplog):
    """Ф3 обязана мерить не только насыщенность.

    `vignette=angle` при значении из пресета glass (0.63 — дефолт ffmpeg PI/5)
    гасит углы до трети яркости центра, а верх и низ 9:16-кадра до 45 %, то есть
    затемняет и макушку, и полосу субтитров. Это художественное решение владельца
    пресета, поэтому код его не меняет, но и молчать про него не должен — ровно
    как с насыщенностью.
    """
    with caplog.at_level(logging.WARNING, logger="src.pipeline.crop"):
        got = grade_filter(Grade(vignette=0.63))
    assert "vignette=angle=0.63" in got  # поведение не меняем, только предупреждаем
    assert any("рамка" in r.getMessage() for r in caplog.records)


def test_grade_stays_quiet_at_the_vignette_threshold(caplog):
    with caplog.at_level(logging.WARNING, logger="src.pipeline.crop"):
        grade_filter(Grade(vignette=VIGNETTE_VISIBLE))
    assert not caplog.records


@pytest.mark.slow
def test_vignette_threshold_is_calibrated_by_luma(tmp_path):
    """Порог виньетки — замер по картинке, а не вкус.

    Строим ту же цепочку, что собирает `grade_filter`, на плоской серой карте
    1080x1920 и смотрим яркость края относительно центра. На пороге падение
    обязано быть незаметным (не глубже 15 %), а на значении из пресета glass —
    заметным (глубже 40 %), иначе предупреждение врёт в одну из сторон.
    """
    import cv2

    from src.core.proc import ffmpeg

    def edge_ratio(angle: float) -> tuple[float, float]:
        png = tmp_path / f"vig-{angle}.png"
        ffmpeg([
            "-f", "lavfi", "-i", "color=c=gray:s=1080x1920:d=0.1",
            "-vf", grade_filter(Grade(contrast=1.0, saturation=1.0, gamma=1.0, vignette=angle)),
            "-frames:v", "1", str(png),
        ])
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        center = float(img[h // 2, w // 2])
        return float(img[0, w // 2]) / center, float(img[0, 0]) / center

    top_ok, corner_ok = edge_ratio(VIGNETTE_VISIBLE)
    assert top_ok > 0.85 and corner_ok > 0.85, (top_ok, corner_ok)

    top_bad, corner_bad = edge_ratio(0.63)
    assert top_bad < 0.6 and corner_bad < 0.5, (top_bad, corner_bad)


def test_autocrop_warns_when_preset_and_framing_are_missing(caplog):
    """Тихая деградация Ф3 и Ф6 обязана быть слышна.

    Оба входа `autocrop` необязательные, и подстановка по умолчанию отменяет по
    целой фазе: без пресета цветокор берётся из конфига (а пресеты различаются
    именно им — glass с виньеткой, bold без), без крупности весь ролик снимается
    одним окном. Ровно так сегодня зовут `autocrop` в `pipeline/job.py` и
    `worker/tasks.py`, поэтому дефект обязан быть виден в логе.
    """
    from src.core.config import settings

    with caplog.at_level(logging.WARNING, logger="src.pipeline.crop"):
        warn_missing_inputs(None, None, settings)
    messages = [r.getMessage() for r in caplog.records]
    assert any("пресет не передан" in m for m in messages)
    assert any("крупность не передана" in m for m in messages)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.pipeline.crop"):
        warn_missing_inputs(load_preset("bold"), [(0.0, 5.0, "wide")], settings)
    assert not caplog.records


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


def test_filter_sets_sar_in_every_branch():
    """Ветки разной крупности обязаны прийти в concat с одинаковым SAR.

    Окна округляются до чётных пикселей и потому не идеально 9:16; `scale`
    доводит пропорцию через SAR, и без `setsar` в ветке concat падает с «Input
    link parameters do not match» — то есть смена крупности роняла весь этап.
    Один `setsar` в хвосте после concat от этого не спасает: он позже отказа.
    """
    vf = build_crop_filter(
        [seg(0.0, 4.0, "wide", 1096, 1948), seg(4.0, 8.0, "tight", 1080, 1920)],
        frame_w=3840, frame_h=2160, out_w=1080, out_h=1920,
    )
    branches = [p for p in vf.split(";") if p.startswith("[b")]
    assert len(branches) == 2
    assert all("setsar=1" in b for b in branches)
    assert vf.index("setsar=1") < vf.index("concat")


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
def test_ffmpeg_survives_branches_of_different_window_sizes(tmp_path):
    """Регрессия на падение concat: ветки разной крупности собираются в один поток.

    Прежний граф ставил `setsar` один раз после `concat`, и ffmpeg отказывался
    склеивать ветки: округление окон до чётных пикселей даёт им разный SAR
    (1:1 против 4384:4383). Проверять это в юнит-тесте по строке недостаточно —
    отказ приходит от самого ffmpeg, поэтому здесь настоящий прогон.

    Существующий slow-тест на смену крупности этот путь не трогал: на фикстуре
    без лиц оба окна выходят одинаковыми и `_merge_segments` схлопывает их в
    одну ветку, то есть `concat` в графе вообще не появляется.
    """
    from src.core.proc import ffmpeg
    from src.pipeline.probe import probe

    src = tmp_path / "src.mp4"
    ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=1200x2000:rate=24:duration=2",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(src),
    ])
    vf = build_crop_filter(
        [seg(0.0, 1.0, "wide", 1096, 1948), seg(1.0, 2.0, "tight", 1080, 1920)],
        frame_w=1200, frame_h=2000, out_w=1080, out_h=1920,
    )
    out = tmp_path / "out.mp4"
    ffmpeg([
        "-i", str(src), "-filter_complex", vf, "-map", "[vout]",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out),
    ])
    info = probe(out)
    assert (info.width, info.height) == (1080, 1920)
    assert info.duration == pytest.approx(2.0, abs=0.2)


@pytest.mark.slow
def test_autocrop_keeps_duration_on_a_faceless_clip(tmp_path):
    """Длительность и звук выживают, когда крупность передана, но не действует.

    Раньше тест назывался «смена крупности режет граф на ветки», и это было
    неправдой: на фикстуре лиц нет, окна обоих кусков выходят одинаковыми,
    `_merge_segments` схлопывает их в один — `concat` в графе не появляется вовсе.
    Ветки и `sendcmd` проверяет `test_autocrop_branches_survive_with_sendcmd`, а
    здесь остаётся то, что тест действительно меряет: вырожденная крупность не
    роняет ни звук, ни длительность.
    """
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


@pytest.mark.slow
def test_autocrop_branches_survive_with_sendcmd(tmp_path):
    """Самый рискованный путь Ф6 целиком: две ветки + `sendcmd` с адресацией.

    Ни один прежний тест его не проходил насквозь. Slow-тест на разный SAR гоняет
    ветки БЕЗ командного файла, а сквозной прогон `autocrop` шёл по фикстуре без
    лиц, где ветка ровно одна. Здесь настоящее лицо, которое ещё и ездит по кадру:
    получаются два окна разного размера (то есть `split`/`trim`/`concat`) и
    `sendcmd` с адресами `crop@c0`/`crop@c1` и перекрывающимися интервалами.
    Проверяем то, что ломается первым: число кадров на входе и на выходе.
    """
    from src.core.proc import ffmpeg, run
    from src.pipeline.crop import autocrop
    from src.pipeline.probe import probe

    if not REFERENCE_CLIP.exists():
        pytest.skip(f"нет эталона {REFERENCE_CLIP}")

    def frames(path) -> int:
        out = run([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
        ])
        return int(out.strip().splitlines()[0])

    # Лицо среднего размера (доля около 0.26) на широком поле и с проездом по
    # горизонтали: первое даёт разные окна у wide и tight, второе — движение окна,
    # ради которого и существует sendcmd.
    src = tmp_path / "pan.mp4"
    ffmpeg([
        "-f", "lavfi", "-i", "color=c=black:s=1620x2160:r=30",
        "-t", "4", "-i", str(REFERENCE_CLIP),
        "-filter_complex",
        "[1:v]scale=-2:900[f];[0:v][f]overlay=x='(W-w)/2+300*sin(2*PI*t/4)':y=(H-h)/2",
        "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", "4", str(src),
    ])

    out = autocrop(
        src, tmp_path / "framed.mp4", work_dir=tmp_path, preset=load_preset("glass"),
        framing=[(0.0, 2.0, "wide"), (2.0, 4.0, "tight")],
    )

    script = (tmp_path / "framed.crop.sendcmd").read_text(encoding="utf-8")
    assert "crop@c0" in script and "crop@c1" in script, script[:200]
    assert probe(out).width == 1080 and probe(out).height == 1920
    assert frames(out) == frames(src)
