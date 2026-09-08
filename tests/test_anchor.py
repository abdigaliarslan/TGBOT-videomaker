"""М2: субтитр, привязанный к треку лица.

Проверяются три предохранителя и, главное, то, КАК они срабатывают: мгновенный
откат телепортирует блок, то есть лечение оказывается хуже болезни. Поэтому
почти каждый тест здесь смотрит не только «отпустил ли», но и «за сколько
кадров».
"""

from __future__ import annotations

import pytest

from src.core.anchor import Safeguards, WeightSpring, fallbacks, follow_offsets
from src.core.edl import ANCHOR_TRACK_MAX, AnchorTrack, CaptionBlock, CaptionWord, validate_edl
from src.core.track import FaceTrack, TrackPoint

FRAME = (1080, 1920)
REST = (540.0, 1500.0)  # центр блока в покое: низ кадра


def track_of(
    points: list[tuple[float, float, float]], fps: int = 30, start: int = 0
) -> FaceTrack:
    """Трек из троек (cx, cy, conf). Высота лица одна и та же — она здесь не при чём."""
    return FaceTrack(
        fps=fps,
        points=[
            TrackPoint(f=start + i, cx=cx, cy=cy, h=0.25, conf=conf)
            for i, (cx, cy, conf) in enumerate(points)
        ],
    )


def offsets_for(track: FaceTrack, frames: int, **kw):
    return follow_offsets(
        track, start_frame=0, frames=frames, rest_center=REST, frame_size=FRAME, **kw
    )


# ------------------------------------------------------------------ привязка


def test_block_arrives_above_the_head_and_stays():
    """Смещение по умолчанию тянет блок ВВЕРХ от лица: текст висит над головой."""
    steady = track_of([(0.5, 0.5, 0.9)] * 60)
    out = offsets_for(steady, 60)
    dx, dy = out[-1]
    assert dy < -600  # блок в покое внизу кадра, приехал заметно выше
    assert abs(dx) < 1.0  # по горизонтали лицо ровно по центру, ехать некуда


def test_the_block_never_lands_on_the_face():
    """Одного `offset_y` мало, и это видно на кадре.

    Смещение задано долей ВЫСОТЫ КАДРА, а нужная величина зависит от размера
    ГОЛОВЫ. На эталонном клипе (селфи в упор, лицо 0.49 высоты) плановые −0.14
    поставили блок ровно на глаза — проверено рендером. Поэтому цель
    дополнительно поднимается над верхом бокса лица.
    """
    for face_h in (0.15, 0.25, 0.49):
        track = FaceTrack(
            fps=30,
            points=[TrackPoint(f=i, cx=0.5, cy=0.5, h=face_h, conf=0.9) for i in range(60)],
        )
        block_h = 139.0
        out = follow_offsets(
            track, start_frame=0, frames=60, rest_center=REST, frame_size=FRAME,
            block_height=block_h,
        )
        bottom = REST[1] + out[-1][1] + block_h / 2
        face_top = (0.5 - face_h / 2) * FRAME[1]
        assert bottom <= face_top, f"лицо {face_h}: низ блока {bottom:.0f} ниже лба {face_top:.0f}"


def test_the_block_stays_inside_the_frame():
    """На сверхкрупном плане «над головой» уводит за верхний край — зажимаем."""
    huge = FaceTrack(
        fps=30, points=[TrackPoint(f=i, cx=0.5, cy=0.5, h=0.9, conf=0.9) for i in range(60)]
    )
    top = FRAME[1] * 0.07
    out = follow_offsets(
        huge, start_frame=0, frames=60, rest_center=REST, frame_size=FRAME,
        block_height=139.0, bounds=(top, FRAME[1] - top),
    )
    assert REST[1] + out[-1][1] - 139.0 / 2 >= top - 1e-6


def test_the_block_does_not_teleport_on_arrival():
    """Привязка включается пружиной: первый кадр не должен брать всё смещение."""
    steady = track_of([(0.5, 0.5, 0.9)] * 60)
    out = offsets_for(steady, 60)
    full = abs(out[-1][1])
    assert abs(out[0][1]) < full * 0.15
    assert abs(out[9][1]) > full * 0.6  # за ~9 кадров основная часть пути пройдена


def test_follow_axes_can_be_switched_off():
    moving = track_of([(0.2, 0.3, 0.9)] * 40)
    only_y = offsets_for(moving, 40, follow_x=False)
    only_x = offsets_for(moving, 40, follow_y=False)
    assert all(dx == 0.0 for dx, _ in only_y)
    assert all(dy == 0.0 for _, dy in only_x)


# ------------------------------------------------------------------ предохранители


def test_low_confidence_releases_the_head():
    """Первый предохранитель: детекции не верим — блок возвращается в покой."""
    points = [(0.5, 0.5, 0.9)] * 40 + [(0.5, 0.5, 0.2)] * 40
    out = offsets_for(track_of(points), 80)
    assert abs(out[39][1]) > 100  # держался
    assert abs(out[-1][1]) < 5  # отпустил


def test_a_fast_turn_releases_the_head():
    """Второй: за резким поворотом текст ехать не должен — это читается как рывок.

    Порог задан в MOTION_PLAN как «порядка 25 px/кадр при 1080x1920», поэтому
    проверяется именно он: медленный проезд привязку не роняет, быстрый роняет.
    """
    slow = track_of([(0.5 + i * 0.005, 0.5, 0.9) for i in range(60)])  # 5.4 px/кадр
    fast = track_of([(0.5 + i * 0.04, 0.5, 0.9) for i in range(30)])  # 43.2 px/кадр
    assert abs(offsets_for(slow, 60)[-1][1]) > 100
    assert abs(offsets_for(fast, 30)[-1][1]) < 20


def test_a_long_loss_of_the_face_releases_the_head():
    """Третий: лицо потеряно дольше нескольких кадров подряд."""
    points = [(0.5, 0.5, 0.9)] * 40 + [(0.5, 0.5, 0.0)] * 40
    out = offsets_for(track_of(points), 80)
    assert abs(out[-1][1]) < 5


def test_a_short_loss_does_not_release_the_head():
    """Моргнувшая детекция — не повод возвращать блок вниз.

    Это и есть смысл `miss_limit_frames`. Если бы провал `conf` ронял привязку
    сам по себе, один кадр моргания стоил бы заметной части смещения — замерено
    37 % на двух кадрах, — то есть блок покачивался бы на ровном месте, ради
    чего вся эта пружина и написана.
    """
    points = [(0.5, 0.5, 0.9)] * 40 + [(0.5, 0.5, 0.0)] * 2 + [(0.5, 0.5, 0.9)] * 40
    out = offsets_for(track_of(points), 82)
    settled = abs(out[39][1])
    assert abs(out[42][1]) > settled * 0.98


def test_a_loss_just_past_the_limit_does_release_the_head():
    """Граница предохранителя настоящая: на кадр длиннее — уже откат."""
    guards = Safeguards()
    blink = [(0.5, 0.5, 0.0)] * (guards.miss_limit_frames + 1)
    points = [(0.5, 0.5, 0.9)] * 40 + blink + [(0.5, 0.5, 0.9)] * 40
    out = offsets_for(track_of(points), 40 + len(blink) + 40)
    settled = abs(out[39][1])
    assert abs(out[40 + len(blink) - 1][1]) < settled * 0.8


def test_the_release_is_not_a_teleport():
    """DoD М2: при потере лица откат незаметен — за 8-10 кадров, а не за один."""
    points = [(0.5, 0.5, 0.9)] * 40 + [(0.5, 0.5, 0.0)] * 40
    out = offsets_for(track_of(points), 80)
    steps = [abs(b[1] - a[1]) for a, b in zip(out[38:], out[39:])]
    settled = abs(out[39][1])
    # Ни один кадр отката не берёт больше четверти пути: 8-10 кадров перехода
    # означают именно это.
    assert max(steps) < settled * 0.25


def test_the_arrival_is_not_a_teleport_either():
    """Симметрично: возвращение привязки тоже идёт пружиной."""
    points = [(0.5, 0.5, 0.1)] * 30 + [(0.5, 0.5, 0.9)] * 50
    out = offsets_for(track_of(points), 80)
    steps = [abs(b[1] - a[1]) for a, b in zip(out[29:], out[30:])]
    assert max(steps) < abs(out[-1][1]) * 0.25


# ------------------------------------------------------------------ мёртвая зона


def test_micro_movement_does_not_move_the_block():
    """Блок не ездит за каждым микродвижением головы: внутри порога он стоит."""
    jitter = [(0.5 + (0.002 if i % 2 else -0.002), 0.5, 0.9) for i in range(60)]
    out = offsets_for(track_of(jitter), 60)
    tail = [dx for dx, _ in out[30:]]
    assert max(tail) - min(tail) < 1.0


def test_a_real_move_gets_through_the_deadzone():
    points = [(0.3, 0.5, 0.9)] * 30 + [(0.7, 0.5, 0.9)] * 30
    out = offsets_for(track_of(points), 60)
    assert out[-1][0] - out[29][0] > 200


def test_the_deadzone_is_a_backlash_not_a_latch():
    """Люфт, а не защёлка: срабатывание не должно давать скачок на весь порог.

    Защёлка переставляет блок сразу на новую позицию — то есть ровно на порог
    (3 % ширины, 32 px). Тот же артефакт уже был замерен в автокропе.
    """
    points = [(0.5 + i * 0.004, 0.5, 0.9) for i in range(60)]
    out = offsets_for(track_of(points), 60)
    steps = [abs(b[0] - a[0]) for a, b in zip(out[20:], out[21:])]
    assert max(steps) < 0.03 * FRAME[0] / 2


# ------------------------------------------------------------------ края и учёт


def test_no_track_means_no_path():
    assert offsets_for(FaceTrack(fps=30), 30) == []


def test_a_block_outliving_the_track_falls_back_instead_of_freezing():
    """Хвост за концом трека — это потеря лица, а не «привязки не было».

    Иначе блок замирал бы на последней известной позиции без всякого отката, и
    зритель видел бы текст, висящий над пустым местом.
    """
    out = offsets_for(track_of([(0.5, 0.5, 0.9)] * 40), 90)
    assert abs(out[39][1]) > 100
    assert abs(out[-1][1]) < 5


def test_fallbacks_are_counted_for_the_report():
    """Метрика из «что измерять»: больше двух откатов за ролик — сигнал."""
    points = [(0.5, 0.5, 0.9)] * 40 + [(0.5, 0.5, 0.0)] * 40
    assert fallbacks(offsets_for(track_of(points), 80)) == 1
    assert fallbacks(offsets_for(track_of([(0.5, 0.5, 0.9)] * 80), 80)) == 0


def test_the_weight_spring_never_overshoots():
    """Критическое демпфирование: перелёт означал бы рывок самого предохранителя."""
    spring = WeightSpring(9)
    values = [spring.step(1.0) for _ in range(120)]
    assert max(values) <= 1.0 + 1e-9
    assert values[-1] == pytest.approx(1.0, abs=1e-3)
    back = [spring.step(0.0) for _ in range(120)]
    assert min(back) >= -1e-9


def test_guards_are_data_not_constants():
    """Пороги настраиваются: клип с плохой детекцией лечится ими, а не правкой кода."""
    points = [(0.5, 0.5, 0.4)] * 60
    strict = offsets_for(track_of(points), 60)
    loose = offsets_for(track_of(points), 60, guards=Safeguards(conf_floor=0.3))
    assert abs(strict[-1][1]) < 5
    assert abs(loose[-1][1]) > 100


# ------------------------------------------------------------------ потолок в контракте


def word_blocks(n: int, anchored: int) -> list[CaptionBlock]:
    return [
        CaptionBlock(
            words=[CaptionWord(word_idx=i)],
            anchor_track=AnchorTrack() if i < anchored else None,
        )
        for i in range(n)
    ]


def test_too_many_anchored_blocks_is_a_validation_error():
    """Когда за головой ездят все субтитры, смотреть невозможно."""
    from src.core.edl import EDL, Cut, Word

    words = [Word(idx=i, text=f"с{i}", start=i * 1.0, end=i * 1.0 + 0.5) for i in range(4)]
    edl = EDL(
        source_duration=4.0,
        words=words,
        cuts=[Cut(src_in=0.0, src_out=4.0, reason="keep")],
        captions=word_blocks(4, anchored=ANCHOR_TRACK_MAX + 1),
    )
    errors = validate_edl(edl)
    assert any("привязано к треку лица" in e for e in errors)

    edl.captions = word_blocks(4, anchored=ANCHOR_TRACK_MAX)
    assert not [e for e in validate_edl(edl) if "привязано к треку" in e]
