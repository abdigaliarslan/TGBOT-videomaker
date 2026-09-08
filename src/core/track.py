"""Трек лица как данные (М1).

Трек уже считался в `pipeline/crop.py`: YuNet проходит по кадрам, из проб
получается геометрия окна кропа — и на этом трек выбрасывался. Здесь он
становится артефактом, которым можно пользоваться дальше: к нему привязываются
субтитры (М2), по нему видно качество детекции на реальных исходниках.

Три решения, которые стоит объяснить.

**Координаты нормализованные, а не пиксельные.** Трек переживает смену
разрешения, пресета и перерендер — то есть согласуется с инвариантом 4. Пиксели
1080x1920 в артефакте означали бы, что превью 720p и финал 1080p требуют разных
файлов, хотя лицо в кадре у них в одном и том же месте.

**Координаты ВЫХОДНОГО кадра, а не исходника.** Потребителю трека нужно знать,
где лицо на той картинке, поверх которой он рисует. Пересчёт делает
`pipeline/crop.project_track`: окно кропа меняется по кускам, поэтому
преобразование посегментное.

**Трек лежит отдельным файлом, а не внутри EDL.** Минута при 30 fps — это 1800
точек; EDL перестал бы быть читаемым документом, а он контракт, который правят
руками.

Про пересборку через резы. MOTION_PLAN требует «резы → тайминги слов → трек»,
и это требование выполняется само: `autocrop` работает уже над НАРЕЗАННЫМ
видео, поэтому пробы трекера с самого начала лежат в выходном времени. Ловушка
осталась бы, только если бы трекинг переехал на исходник.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)


class TrackPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    f: int  # номер кадра ВЫХОДНОГО ролика
    cx: float = Field(ge=-1.0, le=2.0)  # центр лица по X, доля ширины
    cy: float = Field(ge=-1.0, le=2.0)  # по Y, доля высоты
    h: float = Field(ge=0.0, le=4.0)  # высота лица, доля высоты кадра
    conf: float = Field(ge=0.0, le=1.0)


class FaceTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    fps: int
    points: list[TrackPoint] = []
    # Чем и на каком растре получен трек. То же соображение, что в
    # `samples/README.md`: доля лица — свойство пары «кадр + растр анализа», и
    # без этих полей числа из разных прогонов несравнимы.
    detector: str = "—"
    analyzed: tuple[int, int] = (0, 0)

    def at(self, frame: int) -> TrackPoint | None:
        """Точка на кадре. Трек плотный, поэтому это индекс, а не поиск."""
        if not self.points:
            return None
        i = frame - self.points[0].f
        if 0 <= i < len(self.points) and self.points[i].f == frame:
            return self.points[i]
        for p in self.points:  # разреженный трек из чужого инструмента
            if p.f == frame:
                return p
        return None


def load_track(path: str | Path) -> FaceTrack | None:
    """Трек с диска или `None`, если его нет или он не читается.

    Отсутствие трека — законный путь, а не ошибка: кроп мог не найти ни одного
    лица, а старая задача снималась до М1 вовсе. Разбор битого файла гасится по
    той же причине: без трека выйдет ролик со статичными субтитрами, а с
    исключением не выйдет ничего.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        return FaceTrack.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — приём важен, но ролик важнее
        log.warning("трек лица не прочитан (%s) — субтитры останутся статичными", e)
        return None


# --------------------------------------------------------------------------- пропуски


# Насколько занижается уверенность в точке, которой не было. Ноль, а не «чуть
# меньше»: предохранители М2 обязаны узнать про потерю лица, а не про то, что
# она была недолгой. Промежуточные значения появляются сами при пересчёте на
# кадровую сетку — линейная интерполяция между 0 и score соседей даёт мягкий
# спад к дырке и мягкий подъём из неё.
MISSING_CONF = 0.0


def fill_gaps(values: Sequence[float | None]) -> list[float]:
    """Пропуски — линейной интерполяцией, края — удержанием крайнего известного.

    Тот же алгоритм, что у `pipeline/crop.fill_gaps`, но без значения по
    умолчанию: ряд без единого известного значения здесь не бывает — такой трек
    не пишется вовсе.
    """
    n = len(values)
    if not any(v is not None for v in values):
        raise ValueError("ряд без единого известного значения")

    # Два линейных прохода, а не поиск соседей на каждом шаге: минута при 30 fps
    # это 1800 точек, и квадрат тут не нужен.
    prev_known: list[int | None] = [None] * n
    next_known: list[int | None] = [None] * n
    last: int | None = None
    for i in range(n):
        if values[i] is not None:
            last = i
        prev_known[i] = last
    nxt: int | None = None
    for i in range(n - 1, -1, -1):
        if values[i] is not None:
            nxt = i
        next_known[i] = nxt

    out: list[float] = [0.0] * n
    for i in range(n):
        v = values[i]
        if v is not None:
            out[i] = float(v)
            continue
        a, b = prev_known[i], next_known[i]
        if a is not None and b is not None:
            va, vb = float(values[a]), float(values[b])  # type: ignore[arg-type]
            out[i] = va + (vb - va) * (i - a) / (b - a)
        else:
            k = a if a is not None else b
            out[i] = float(values[k])  # type: ignore[arg-type,index]
    return out


def resample(times: Sequence[float], values: Sequence[float], at: Iterable[float]) -> list[float]:
    """Ряд, снятый пробами, на произвольные моменты времени — линейно.

    Пробы идут 6 раз в секунду, а кадров 24-60: без пересчёта трек описывал бы
    каждый пятый кадр, и потребителю пришлось бы интерполировать самому. Раз
    интерполяция всё равно нужна, она живёт здесь, в одном месте.
    """
    if not times:
        return []
    out: list[float] = []
    j = 0
    for t in at:
        while j + 1 < len(times) and times[j + 1] < t:
            j += 1
        if t <= times[0]:
            out.append(values[0])
        elif t >= times[-1]:
            out.append(values[-1])
        else:
            t0, t1 = times[j], times[min(j + 1, len(times) - 1)]
            v0, v1 = values[j], values[min(j + 1, len(values) - 1)]
            k = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            out.append(v0 + (v1 - v0) * k)
    return out


# --------------------------------------------------------------------------- сглаживание


class OneEuro:
    """One euro filter: на малых скоростях режет шум, на больших не мешает.

    Скользящее среднее здесь не годится, и это не вкус, а свойство фильтра:
    окно приходится выбирать между дрожанием на статике (узкое окно) и
    запаздыванием на быстром движении (широкое), потому что оно одно на оба
    режима. One euro ведёт частоту среза от измеренной скорости — стоит голове
    поехать, срез поднимается и фильтр почти перестаёт вмешиваться.

    Параметры: `min_cutoff` задаёт, насколько сильно давится дрожь в покое,
    `beta` — насколько быстро фильтр уступает движению. Стартовые 1.0 и 0.007 —
    из статьи Casiez et al. и из MOTION_PLAN; живут в пресете, а не здесь.
    """

    def __init__(
        self, freq: float, *, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0
    ) -> None:
        if freq <= 0:
            raise ValueError("частота кадров должна быть положительной")
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: float | None = None
        self._dx = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * max(1e-6, cutoff))
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self._x is None:
            self._x = x
            return x
        dx = (x - self._x) * self.freq
        a_d = self._alpha(self.d_cutoff)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff)
        self._x = a * x + (1 - a) * self._x
        return self._x


# Кадр, в ПИКСЕЛЯХ которого калиброваны `min_cutoff` и `beta`.
#
# Это не мелочь реализации, а условие, без которого стартовые значения из
# MOTION_PLAN (1.0 и 0.007) не работают вовсе. `beta` умножается на измеренную
# СКОРОСТЬ, то есть его величина привязана к единицам сигнала. Трек хранится в
# долях кадра, и скорость в них выходит порядка 1 в секунду вместо ~2000 px/с —
# срез поднимается на 0.007 Гц вместо 14, фильтр перестаёт уступать движению и
# превращается в обычный ФНЧ. Замерено: на равномерном проезде через весь кадр
# за две секунды отставание было 294 px против 21 px после нормировки.
#
# Поэтому сглаживание идёт в пикселях опорного кадра 1080x1920 — того самого, в
# котором MOTION_PLAN задаёт и порог скорости для предохранителей М2
# («порядка 25 px/кадр при 1080×1920»). Числа в пресете и числа в плане после
# этого означают одно и то же.
REF_WIDTH = 1080
REF_HEIGHT = 1920


def smooth(
    values: Sequence[float],
    *,
    fps: float,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
    scale: float = 1.0,
) -> list[float]:
    """Один проход one euro по ряду.

    `scale` переводит значения в единицы, в которых калиброваны `min_cutoff` и
    `beta` (см. `REF_WIDTH`/`REF_HEIGHT`), и результат возвращается обратно в
    исходных. Единица означает «ряд уже в нужных единицах».
    """
    if scale <= 0:
        raise ValueError("масштаб должен быть положительным")
    filt = OneEuro(fps, min_cutoff=min_cutoff, beta=beta)
    return [filt(v * scale) / scale for v in values]
