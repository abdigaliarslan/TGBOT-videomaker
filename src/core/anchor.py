"""М2: путь блока субтитров, привязанного к треку лица.

Почему это Python, а не TSX. Инвариант §0 — рендерер ничего не решает и не
считает: он получает готовые секунды, готовые строки и готовые прямоугольники.
Привязка к голове выглядит исключением («движение обязано считаться на каждом
кадре»), но им не является: кадров конечное число, трек известен заранее,
значит и путь известен заранее. Поэтому здесь считается ГОТОВЫЙ путь — список
смещений в пикселях по кадрам, — а рендерер его просто применяет. Побочная
выгода: предохранители и откат проверяются обычным тестом на Python, а не
глазами по видео.

Три предохранителя обязательны все, иначе приём превращается в дёрганье:

1. `conf` ниже порога — детекция ненадёжна;
2. скорость бокса выше порога — резкий поворот головы, за которым текст ехать
   не должен;
3. лицо не найдено дольше нескольких кадров подряд.

Первый и третий в реализации сведены в один — почему, написано у самой проверки
в `follow_offsets`: порознь третий не срабатывает никогда, а первый начинает
дёргать блок на однокадровом моргании детекции.

И главное: **откат не мгновенный**. Переход между привязанной и статичной
позицией ведётся отдельным параметром `follow_weight`, а блок ставится в
`lerp(static, tracked, follow_weight)`. Без этого блок телепортируется в момент
срабатывания предохранителя — то есть лечение оказывается хуже болезни.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from src.core.track import REF_HEIGHT, REF_WIDTH, FaceTrack

# Уверенность, ниже которой трек не ведёт блок.
CONF_FLOOR = 0.5
# Скорость бокса, выше которой блок отпускает голову, px/кадр в опорном кадре
# 1080x1920. За быстрым поворотом текст ехать не должен: это читается как рывок
# самого текста, а не как движение головы.
SPEED_LIMIT_PX = 25.0
# Сколько кадров подряд лицо может отсутствовать, прежде чем блок откатится.
MISS_LIMIT_FRAMES = 5
# Мёртвая зона: блок стоит, пока лицо не ушло дальше этой доли ШИРИНЫ кадра.
# Тот же приём, что в автокропе, и по той же причине: без него блок ездит за
# каждым микродвижением головы.
DEADZONE_RATIO = 0.03
# За сколько кадров происходит переход между привязанной и статичной позицией.
FOLLOW_RAMP_FRAMES = 9


@dataclass(frozen=True, slots=True)
class Safeguards:
    conf_floor: float = CONF_FLOOR
    speed_limit_px: float = SPEED_LIMIT_PX
    miss_limit_frames: int = MISS_LIMIT_FRAMES
    deadzone_ratio: float = DEADZONE_RATIO
    ramp_frames: int = FOLLOW_RAMP_FRAMES


class WeightSpring:
    """Критически задемпфированная пружина для `follow_weight` от 0 до 1.

    Критически задемпфированная, а не обычная: перелёт здесь означал бы, что при
    откате блок сначала уезжает ДАЛЬШЕ статичной позиции, а потом возвращается —
    то есть предохранитель сам делает рывок, который должен был предотвратить.

    Шаг аналитический, а не численный. Явная схема при `omega`, дающей посадку
    за 9 кадров, стоит на границе устойчивости (omega·dt ≈ 0.53) и на коротком
    блоке начинает звенеть; аналитическая устойчива при любом шаге.
    """

    def __init__(self, ramp_frames: int) -> None:
        # 4.74 — момент, когда критически задемпфированный отклик берёт 95 %:
        # решение (1 + ωt)·e^(−ωt) = 0.05.
        self.omega = 4.74 / max(1, ramp_frames)
        self.value = 0.0
        self.velocity = 0.0

    def step(self, target: float, dt: float = 1.0) -> float:
        w, v, k = self.omega, self.velocity, self.value - target
        decay = math.exp(-w * dt)
        temp = (v + w * k) * dt
        self.value = target + (k + temp) * decay
        self.velocity = (v - temp * w) * decay
        return self.value


def _deadzone(values: Sequence[float], threshold: float) -> list[float]:
    """Люфт, а не защёлка: выход отстаёт от цели ровно на порог.

    Защёлка («ушли дальше порога — переставили на новую позицию целиком») даёт
    скачок ровно на величину порога в момент срабатывания. В автокропе это уже
    было замерено как видимый артефакт (см. `crop.apply_deadzone`), и здесь
    ошибка была бы та же, только заметнее: текст крупнее лица.
    """
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        last = out[-1]
        if v - last > threshold:
            out.append(v - threshold)
        elif last - v > threshold:
            out.append(v + threshold)
        else:
            out.append(last)
    return out


# Зазор между низом блока и верхом бокса лица, доля высоты кадра. Не ноль:
# бокс YuNet начинается от бровей, волосы и лоб остаются выше него.
HEAD_CLEARANCE = 0.06


def _above_the_head(
    point,
    offset_y: float,
    height: int,
    block_height: float,
    bounds: tuple[float, float] | None,
) -> float:
    """Куда по вертикали целится центр блока на этом кадре.

    Плановое `offset_y` — доля ВЫСОТЫ КАДРА, а нужная величина зависит от
    размера ГОЛОВЫ, и на крупном плане одно с другим расходится: при лице в
    0.49 высоты кадра (эталонный клип) −0.14 ставит блок на глаза. Поэтому цель
    не опускается ниже, чем «низ блока над верхом бокса лица», а потом зажимается
    в кадр — иначе на совсем крупном плане текст уехал бы за верхний край.

    Смещение вниз (`offset_y > 0`, блок под подбородком) правилу не подчиняется:
    оно задумано как «на голову не наезжаем снизу», а верх головы к нему
    отношения не имеет.
    """
    target = (point.cy + offset_y) * height
    if offset_y < 0:
        face_top = (point.cy - point.h / 2 - HEAD_CLEARANCE) * height
        target = min(target, face_top - block_height / 2)
    if bounds is not None:
        lo, hi = bounds
        target = min(max(target, lo + block_height / 2), hi - block_height / 2)
    return target


def follow_offsets(
    track: FaceTrack,
    *,
    start_frame: int,
    frames: int,
    rest_center: tuple[float, float],
    frame_size: tuple[int, int],
    offset_x: float = 0.0,
    offset_y: float = -0.14,
    follow_x: bool = True,
    follow_y: bool = True,
    block_height: float = 0.0,
    bounds: tuple[float, float] | None = None,
    guards: Safeguards | None = None,
) -> list[tuple[float, float]]:
    """Смещения блока по кадрам, в пикселях кадра. Пустой список — привязки нет.

    `rest_center` — центр блока в покое, тот же, что задаёт `PlanBlock.rect`.
    Возвращается именно СМЕЩЕНИЕ от него, а не абсолютная позиция: рендерер уже
    ставит блок раскладкой, и подменять её значило бы считать вёрстку дважды.

    `block_height` и `bounds` — высота блока и вертикальные границы, за которые
    его нельзя выпускать. Оба нужны затем, что одного `offset_y` мало, и это
    видно на кадре: смещение задано долей ВЫСОТЫ КАДРА, а нужная величина
    зависит от размера ГОЛОВЫ. На эталонном клипе (селфи в упор, лицо занимает
    0.49 высоты) блок с плановыми −0.14 встал ровно на глаза. Поэтому цель
    дополнительно поднимается над верхом бокса лица, а потом зажимается в кадр.
    """
    guards = guards or Safeguards()
    width, height = frame_size
    if frames <= 0 or not track.points or width <= 0 or height <= 0:
        return []

    # --- цель по кадрам: где должен стоять центр блока, если верить треку
    #
    # Кадры, на которых трека нет вовсе (блок вылез за его конец), считаются
    # потерей лица — не «привязки нет», а именно потерей: иначе блок в хвосте
    # ролика замер бы на последней известной позиции без всякого отката.
    target_x: list[float] = []
    target_y: list[float] = []
    low: list[bool] = []
    for i in range(frames):
        point = track.at(start_frame + i)
        if point is None:
            # Кадр за концом трека — тоже потеря лица, а не «привязки нет»: иначе
            # блок в хвосте ролика замер бы на последней известной позиции без
            # всякого отката, то есть висел бы над пустым местом.
            target_x.append(target_x[-1] if target_x else rest_center[0])
            target_y.append(target_y[-1] if target_y else rest_center[1])
            low.append(True)
            continue
        target_x.append((point.cx + offset_x) * width)
        target_y.append(_above_the_head(point, offset_y, height, block_height, bounds))
        low.append(point.conf < guards.conf_floor)

    # Первый и третий предохранители — ОДИН, и это отклонение от плана.
    #
    # По плану «conf ниже 0.5» и «лицо не найдено больше 5 кадров подряд» —
    # разные триггеры. Но в треке потеря лица И ВЫРАЖАЕТСЯ нулевой уверенностью
    # (`track.MISSING_CONF`), поэтому порознь третий не срабатывает никогда:
    # первый уже отработал. А по отдельности первый даёт другую беду — моргнувшая
    # на один-два кадра детекция стоит 37 % смещения (замерено), то есть видимое
    # покачивание блока на ровном месте. Ради невидимости отката и написан весь
    # этот код, поэтому короткие провалы терпим, а отпускаем на длинных.
    valid = [not v for v in low]
    run_start = None
    for i in range(frames + 1):
        if i < frames and low[i]:
            run_start = i if run_start is None else run_start
            continue
        if run_start is not None:
            if i - run_start <= guards.miss_limit_frames:
                for j in range(run_start, i):  # моргание — держим голову
                    valid[j] = True
            run_start = None

    # --- мёртвая зона: блок стоит, пока лицо не ушло дальше порога
    zone = guards.deadzone_ratio * width
    held_x = _deadzone(target_x, zone)
    held_y = _deadzone(target_y, zone)

    # --- скорость: считается по УДЕРЖАННОЙ цели, то есть по тому, куда блок
    # действительно собирается ехать. По сырому треку она включала бы дрожь,
    # которую мёртвая зона уже погасила, и предохранитель срабатывал бы на
    # неподвижном блоке.
    kx, ky = REF_WIDTH / width, REF_HEIGHT / height
    for i in range(1, frames):
        dx = (held_x[i] - held_x[i - 1]) * kx
        dy = (held_y[i] - held_y[i - 1]) * ky
        if math.hypot(dx, dy) > guards.speed_limit_px:
            valid[i] = False

    # --- вес привязки: пружиной, а не скачком
    spring = WeightSpring(guards.ramp_frames)
    out: list[tuple[float, float]] = []
    for i in range(frames):
        w = spring.step(1.0 if valid[i] else 0.0)
        w = min(1.0, max(0.0, w))
        dx = (held_x[i] - rest_center[0]) * w if follow_x else 0.0
        dy = (held_y[i] - rest_center[1]) * w if follow_y else 0.0
        out.append((round(dx, 2), round(dy, 2)))
    return out


def fallbacks(offsets: Sequence[tuple[float, float]], *, epsilon: float = 0.5) -> int:
    """Сколько раз за блок привязка отпускала голову.

    Метрика из «что измерять» MOTION_PLAN: больше двух откатов за ролик значит,
    что предохранители слишком чувствительные или исходник неподходящий.
    Считается по самому пути, а не по флагам: важно не «сработал предохранитель»,
    а «блок это заметил» — короткий провал веса пружина съедает, и на кадре его
    нет.
    """
    if not offsets:
        return 0
    near_rest = [abs(dx) < epsilon and abs(dy) < epsilon for dx, dy in offsets]
    return sum(1 for a, b in zip(near_rest, near_rest[1:]) if b and not a)
