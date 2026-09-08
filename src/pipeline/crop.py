"""M6: автокроп в 9:16 с трекингом центра внимания и лёгким цветокором.

Схема: детекция по кадрам → заполнение пропусков → прямоугольник кропа по
правилам пресета → сглаживание прямоугольника → динамический `crop` через
`sendcmd` → `scale` → `grade`.

Про крупность (Ф6). Масштаб окна выводится ИЗ ДОЛИ ЛИЦА по высоте кадра
(`preset.crop.face_height_ratio`), а не из вписывания bbox. Зритель видит именно
долю, а абсолютный размер бокса детектора сам по себе не значит ничего: одно и
то же лицо на 720p и на 4K даёт разное число пикселей и одну и ту же долю.

Про движение. Крупность меняется ТОЛЬКО на резах — по полю `Cut.framing`,
которое расставляет Python в `director.assign_framing`. Зум внутри куска спорит
с субтитрами, требует физики по номеру кадра и на коротких окнах вырождается в
ступеньку (AUDIT §8.2); смена на резе не требует ничего, потому что в этот
момент картинка и так меняется целиком.

Про кламп. `max_scale` запрещает добирать крупность апскейлом: окно кропа
никогда не ниже `out_h / max_scale`. На вертикальном селфи, снятом в упор, это
значит, что кроп не делает ничего — и это честный ответ, а не отказ. Кроп умеет
только сужать кадр, то есть делать лицо ЕЩЁ крупнее; уменьшить долю лица можно
лишь полями или апскейлом того, чего не снято. Такой случай пишется в лог, чтобы
«доля 47 %» не выглядела ошибкой кадрирования.

Про детектор. В §1 указан mediapipe, но здесь стоит YuNet из OpenCV, который
уже есть в зависимостях:

- opencv-python-headless уже нужен пайплайну, mediapipe тянет отдельное дерево
  зависимостей и жёстко пинит protobuf<5, ломая torch/whisperx в этом окружении;
- YuNet (232 КБ) на дальних и повёрнутых лицах точнее короткофокусного
  BlazeFace, который mediapipe использует по умолчанию.

Интерфейс детектора абстрактный: добавить mediapipe = добавить класс, не трогая
остальную логику.

Про растр анализа. Доля лица — свойство пары «кадр + растр, на котором работал
детектор»: бокс YuNet зависит от пиксельного размера лица, и на одном и том же
кадре число гуляет вдвое (см. samples/README.md). Значит цель `face_height_ratio`
и замер `tests/face_share.py` обязаны говорить об одном растре, иначе кроп целится
не туда, куда меряет DoD. Канон замера — фиксированная ВЫСОТА 1138; здесь она же,
но с потолком по числу пикселей, потому что детекция линейна по площади и на
16:9-входе канон стоил бы 280 мс на кадр вместо 80 (замер на этой машине).

С каноном растр совпадает ТОЛЬКО на 9:16 (и на любом входе не шире 640x1138 по
площади). На всех остальных пропорциях выигрывает потолок площади, и высота
растра уходит от 1138: 16:9 → 640, 1:1 → 853, 4:5 → 954. Насколько при этом
расходится само число, зависит от того, насколько крупно снято лицо. Замер по
samples/phase-0-source.mp4, добитому полями до нужной пропорции (растр crop против
канона h=1138, тест `test_analyze_raster_divergence_from_canon`):

    доля лица ≈ 0.29 (рабочая точка DoD)   16:9  0.281 / 0.294  (−4 %)
                                           1:1   0.295 / 0.298  (−1 %)
    лицо во весь кадр                      16:9  0.627 / 0.472  (+33 %)
                                           1:1   0.613 / 0.459  (+34 %)
                                           4:5   0.554 / 0.489  (+13 %)

То есть в коридоре DoD (0.20-0.30) числа сходятся в пределах нескольких
процентов, а на крупном плане расходятся в разы больше — и там «доля лица» в
логе уже нельзя сравнивать с выводом tests/face_share.py напрямую. Само по себе
расхождение безопасно: растр переоценивает долю, а значит кроп берёт окно шире
заказанного, то есть ошибается в сторону несрезанного лица.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Protocol, Sequence

import cv2
import numpy as np

from src.core.config import Settings, settings as default_settings
from src.core.edl import EDL
from src.core.preset import Crop, Grade, Preset, load_preset
from src.core.proc import ffmpeg
from src.core.track import (
    MISSING_CONF,
    REF_HEIGHT,
    REF_WIDTH,
    FaceTrack,
    TrackPoint,
    fill_gaps as fill_track_gaps,
    resample,
    smooth,
)
from src.pipeline.probe import MediaInfo, probe

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "assets" / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

TARGET_AR = 9 / 16

# Растр анализа: нормировка по высоте, как в tests/face_share.py, но не дороже
# вертикального канона 640x1138 по площади. На 9:16-входе оба правила совпадают до
# пикселя; на остальных пропорциях побеждает потолок площади, и высота уходит ниже
# 1138 (16:9 → 640, 1:1 → 853). Что это стоит по числу — в докстринге модуля:
# в коридоре DoD расхождение с каноном единицы процентов, на крупном плане — до
# 34 %. Прежняя нормировка по ширине 640 давала на 16:9 растр 640x360, то есть
# высоту лица вдвое меньше канона.
ANALYZE_HEIGHT = 1138
ANALYZE_PIXELS = 640 * ANALYZE_HEIGHT
# Пол по высоте растра: на сверхшироком входе (21:9 и шире) потолок площади увёл бы
# анализ на 554 px, то есть лицо пришло бы к детектору вдвое мельче канона и бокс
# уехал бы вместе с ним. Такие пропорции редки, поэтому здесь площадь уступает —
# перерасход по цене доходит до полутора раз, но число остаётся сравнимым.
ANALYZE_MIN_HEIGHT = ANALYZE_HEIGHT // 2

# Где макушка относительно того, что видит детектор. Правило существует ровно
# затем, чтобы не срезать лоб, поэтому обязано ошибаться в сторону запаса.
#
# Замер по samples/phase-0-source.mp4 на растре 640x1138, макушка снята по
# пиксельной линейке на пяти кадрах (1, 5, 10, 15, 19 с). Требуемый множитель:
#
#   (верх бокса − макушка) / высота бокса   0.478 0.199 0.403 0.566 0.157  → 3.6×
#   (глаза − макушка) / (рот − глаза)       1.190 1.016 1.063 1.569 1.118  → 1.5×
#
# Отсюда правило по ландмаркам, а не по боксу. Бокс YuNet на одном и том же лице
# гуляет по высоте на ±15 % от кадра к кадру, и его верх ходит от линии бровей до
# середины лба; пара «глаза-рот» — плоскостной размер лица, он стабильнее вдвое.
# Прежняя константа HEAD_ABOVE_BOX = 0.4 ссылалась на множитель 1.4 из
# tests/face_share.py, который там сам помечен как некалиброванный: на этих кадрах
# она то завышает запас на 12 % высоты кадра, то срезает макушку.
# Граница по выборке, а не калибровка правила: пять кадров ОДНОГО лица с одной
# причёской, макушка снята рукой с округлением до 5 px на растре высотой 1138.
# 1.6 — это «максимум по выборке (1.569) плюс чуть-чуть»: на худшем кадре (t=15)
# запас оценки над снятой макушкой 6.7 px, то есть в пределах точности самого
# замера. На длинных волосах, шапке или высокой причёске правило поедет, и это
# видно будет только глазами. Расширять — вторым и третьим человеком в HEAD_SAMPLES,
# а не подкруткой константы.
HEAD_ABOVE_EYES = 1.6
# Запасные оценки для детектора без ландмарок (Haar). Проверить их на том же
# клипе нечем: Haar нашёл лицо на 1 кадре из 5, — поэтому они калиброваны по
# геометрии YuNet и взяты по верхней границе разброса, то есть заведомо свободнее.
HEAD_ABOVE_BOX = 0.6
EYES_BELOW_BOX_TOP = 0.2  # медиана замеренного (глаза − верх бокса)/h = 0.178

# Насколько крупнее должно быть чужое лицо, чтобы отобрать окно у того, за которым
# трек уже идёт. Без гистерезиса на кадре с двумя людьми окно скачет между ними,
# потому что «самое крупное» меняется от пробы к пробе.
FACE_SWITCH_RATIO = 1.5

# Потолок частоты смен крупности: чаще одной на столько секунд — это уже не
# монтаж, а мельтешение, и такой EDL сводится к одному окну на весь ролик.
#
# Раньше здесь стоял АБСОЛЮТНЫЙ потолок в 12 кусков, и он тихо выключал всю Ф6 на
# длинных роликах: при выдержке режиссёра 2.2 с смоделированная нарезка даёт 14
# кусков на минуте и 42 на трёх, то есть любой ролик длиннее полуминуты терял
# крупность целиком — и без единой строчки в логе на уровне WARNING.
#
# Пересчёт кусков в секунды. n кусков занимают не n интервалов, а n−1: между
# первой и последней сменой их ровно n−1. `director.FRAMING_MIN_HOLD` = 4.5 с
# гарантирует именно ИНТЕРВАЛ, поэтому честный потолок — `covered / 4.0 + 1`.
# Без «+1» согласование было мнимым: на 10-20-секундных роликах (то есть на
# основном формате) собственный план режиссёра пробивал потолок и Ф6 схлопывалась
# в одно окно. Свип 20 000 случайных нарезок с длинами от cut_min_segment: 20
# нарушений старой формулой (все на 10-20 с), 0 — новой. Пример: куски
# [4.8, 4.77, 2.11] дают три крупности на 11.68 с, где старый потолок разрешал две.
MIN_SECONDS_PER_FRAMING = 4.0
# Меньше двух кусков смысла не имеет: это и есть «одно окно».
MIN_FRAMING_SEGMENTS = 2

# Насколько окна соседних кусков должны различаться, чтобы держать ради этого
# отдельную ветку графа. На 4K-входе кламп `max_scale` сводит wide и tight почти
# в одно окно (замер: 1214x2158 против 1204x2140 — 0.8 % по масштабу и 0.002 по
# доле лица), и ради невидимой разницы граф уходил в split/trim/concat — лишний
# стык и лишний риск. Порог 4 % — примерно там, где смена крупности вообще
# начинает читаться на глаз.
FRAMING_MERGE_TOLERANCE = 0.04

# Порог насыщенности, выше которого кожа уходит в красноту (QUALITY_PLAN, Ф3).
SATURATION_RED_SKIN = 1.2

# Порог виньетки, выше которого рамка перестаёт быть «лёгким цветокором».
# Замер по плоской серой карте 1080x1920 (ffmpeg -f lavfi color=gray, тот же
# фильтр, что собирает grade_filter), яркость относительно центра:
#
#   angle   верх-центр   угол
#   0.15       0.96      0.95
#   0.20       0.92      0.91
#   0.25       0.89      0.86
#   0.40       0.74      0.66
#   0.63       0.45      0.33   ← дефолт ffmpeg PI/5 и значение пресета glass
#
# 0.63 гасит углы до трети центра, а верх и низ 9:16-кадра — до 45 %, то есть
# затемняет и макушку, и полосу субтитров. Это уже не «настолько, чтобы не было
# видно», а художественное решение, и принимать его должен владелец пресета, а не
# код. Порог 0.25 — верх диапазона, на котором падение к краю ещё в пределах
# ~10 % и на глаз не читается.
VIGNETTE_VISIBLE = 0.25


@dataclass(slots=True)
class Detection:
    x: float  # центр по горизонтали, пиксели исходника
    y: float
    score: float
    area: float
    # Размеры бокса нужны отдельно от площади: доля лица меряется по ВЫСОТЕ кадра,
    # а из area её не вытащить. Обязательные, без значений по умолчанию: детектор,
    # который их не заполнит, обязан упасть на конструкторе, а не молча выдать
    # «лиц не найдено» в замере доли лица.
    w: float
    h: float
    # Ландмарки детектора: линия глаз и линия рта. None — считаем оценкой по боксу.
    eye_y: float | None = None
    mouth_y: float | None = None

    @property
    def top(self) -> float:
        """Верхняя граница бокса детектора (от бровей до середины лба, см. замер)."""
        return self.y - self.h / 2

    @property
    def head_top(self) -> float:
        """Оценка макушки — волосы и лоб выше бокса.

        По ландмаркам, когда они есть: «глаза-рот» устойчив к дрожи бокса, а на
        нём и стоит правило «макушка не обрезана». По высоте бокса — только для
        детектора без ландмарок.
        """
        if self.eye_y is not None and self.mouth_y is not None and self.mouth_y > self.eye_y:
            return self.eye_y - HEAD_ABOVE_EYES * (self.mouth_y - self.eye_y)
        return self.top - HEAD_ABOVE_BOX * self.h

    @property
    def eyes(self) -> float:
        """Линия глаз: из ландмарок, иначе оценка по высоте бокса."""
        if self.eye_y is not None:
            return self.eye_y
        return self.top + EYES_BELOW_BOX_TOP * self.h

    def rescaled(self, k: float) -> "Detection":
        """Тот же бокс в другом масштабе кадра: детекция идёт на ужатой копии."""
        return Detection(
            x=self.x * k,
            y=self.y * k,
            score=self.score,
            area=self.area * k * k,
            w=self.w * k,
            h=self.h * k,
            eye_y=None if self.eye_y is None else self.eye_y * k,
            mouth_y=None if self.mouth_y is None else self.mouth_y * k,
        )


class Detector(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class YuNetDetector:
    """DNN-детектор лиц из OpenCV Zoo. Основной путь."""

    name = "yunet"

    def __init__(self, model_path: Path = YUNET_MODEL, score: float = 0.6) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"нет модели детектора: {model_path}")
        self._net = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score, 0.3, 5000)
        self._size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._net.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._net.detect(frame)
        if faces is None:
            return []
        out: list[Detection] = []
        for f in faces:
            fx, fy, fw, fh, *_rest = f
            # f[4:6] и f[6:8] — правый и левый глаз, f[10:12] и f[12:14] — углы рта.
            # Линии глаз и рта из ландмарок честнее любой доли от высоты бокса: на
            # наклоне головы бокс едет, ландмарки нет.
            has_marks = len(f) >= 15
            eye_y = (float(f[5]) + float(f[7])) / 2 if has_marks else None
            mouth_y = (float(f[11]) + float(f[13])) / 2 if has_marks else None
            out.append(
                Detection(
                    x=float(fx) + float(fw) / 2,
                    y=float(fy) + float(fh) / 2,
                    score=float(f[-1]),
                    area=float(fw) * float(fh),
                    w=float(fw),
                    h=float(fh),
                    eye_y=eye_y,
                    mouth_y=mouth_y,
                )
            )
        return out


class HaarDetector:
    """Фолбэк без внешних файлов: каскад идёт в комплекте с OpenCV."""

    name = "haar"

    def __init__(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"не загрузился каскад {path}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48)
        )
        return [
            Detection(
                x=float(x) + w / 2,
                y=float(y) + h / 2,
                score=0.5,
                area=float(w * h),
                w=float(w),
                h=float(h),
            )
            for (x, y, w, h) in faces
        ]


def make_detector() -> Detector:
    try:
        return YuNetDetector()
    except Exception as e:  # noqa: BLE001 — детектор не должен ронять пайплайн
        log.warning("YuNet недоступен (%s), беру Haar", e)
        return HaarDetector()


# --------------------------------------------------------------------------- трек


@dataclass(slots=True)
class Track:
    """Проход по кадрам: в каждой пробе либо лицо, либо запасной центр внимания."""

    times: list[float] = field(default_factory=list)
    faces: list[Detection | None] = field(default_factory=list)
    # Центр масс движения — только по горизонтали: для кадрирования головы у
    # вертикали центра масс нет никакого смысла.
    motion_x: list[float | None] = field(default_factory=list)
    detector: str = "—"
    # Растр, на котором работал детектор. Не деталь реализации, а часть числа:
    # доля лица без него несравнима между прогонами (samples/README.md).
    analyzed: tuple[int, int] = (0, 0)

    def __len__(self) -> int:
        return len(self.times)

    def face_height(self, lo: int = 0, hi: int | None = None) -> float:
        """Медианная высота лица на срезе проб. 0.0 — лиц не было.

        Медиана, а не среднее: один сорвавшийся бокс на пол-кадра иначе утащит
        крупность всего куска.
        """
        hs = [f.h for f in self.faces[lo:hi] if f is not None and f.h > 0]
        return float(median(hs)) if hs else 0.0

    def eye_to_head(self, lo: int = 0, hi: int | None = None) -> float:
        """Медианное расстояние от макушки до линии глаз. 0.0 — лиц не было."""
        ds = [f.eyes - f.head_top for f in self.faces[lo:hi] if f is not None and f.h > 0]
        return float(median(ds)) if ds else 0.0


def _motion_center(prev: np.ndarray, cur: np.ndarray) -> float | None:
    """Центр масс движения — запасной «центр внимания», когда лиц нет."""
    diff = cv2.absdiff(prev, cur)
    diff = cv2.GaussianBlur(diff, (21, 21), 0)
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    column = mask.sum(axis=0).astype(np.float64)
    total = column.sum()
    if total < 255 * 40:  # почти статичный кадр
        return None
    xs = np.arange(column.size, dtype=np.float64)
    return float((column * xs).sum() / total)


def analyze_scale(width: int, height: int) -> float:
    """Во сколько раз ужать (или растянуть) кадр перед детекцией.

    Правило: нормировка по ВЫСОТЕ до `ANALYZE_HEIGHT`, пока хватает бюджета
    площади; не хватило — растр фиксированной площади с той же пропорцией кадра;
    и пол по высоте на сверхшироком входе.

    Нормировка по высоте нужна затем, что доля лица меряется по высоте: при
    фиксированной высоте растра лицо одной и той же доли всегда приходит к
    детектору одного пиксельного размера. Нормировка по ширине (как было) этого не
    даёт вовсе: 16:9-вход уезжал на 640x360.

    Но бюджет площади выигрывает у высоты на ВСЁМ, кроме 9:16 (16:9 → 1138x640,
    1:1 → 853x853, 4:5 → 763x954), поэтому «растр совпадает с tests/face_share.py»
    верно только для вертикального входа. Насколько расходится само число — в
    докстринге модуля: в коридоре DoD единицы процентов, на крупном плане до 34 %.
    Цена канона — 281 мс на пробу против 80 мс на 16:9 (замер YuNet на этой
    машине), поэтому потолок оставлен, а расхождение задокументировано и
    закреплено тестом, а не выдано за отсутствующее.
    """
    if width <= 0 or height <= 0:
        return 1.0
    scale = ANALYZE_HEIGHT / height
    if scale * scale * width * height > ANALYZE_PIXELS:
        scale = (ANALYZE_PIXELS / (width * height)) ** 0.5
    return max(scale, ANALYZE_MIN_HEIGHT / height)


def _resize(frame: np.ndarray, scale: float) -> np.ndarray:
    """Кадр в растр анализа. Интерполяция та же, что в tests/face_share.py."""
    if abs(scale - 1.0) < 1e-9:
        return frame
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=interp)


def pick_face(dets: Sequence[Detection], prev: Detection | None) -> Detection | None:
    """Из всех лиц кадра — то, за которым идёт трек.

    Без памяти о предыдущей пробе выбор «самое крупное» на кадре с двумя людьми
    переключается туда-сюда от пробы к пробе, и окно кропа скачет между лицами
    (сглаживание такой скачок не лечит: он больше мёртвой зоны). Поэтому лицо,
    перекрывающееся с прошлым, уступает соседу только когда тот крупнее в
    FACE_SWITCH_RATIO раз — то есть когда в кадр реально вошёл другой человек.
    """
    if not dets:
        return None
    best = max(dets, key=lambda d: d.area * (0.5 + d.score))
    if prev is None or prev.w <= 0 or prev.h <= 0:
        return best
    near = [
        d
        for d in dets
        if abs(d.x - prev.x) < 0.6 * prev.w and abs(d.y - prev.y) < 0.6 * prev.h
    ]
    if not near:
        return best
    keep = max(near, key=lambda d: d.area * (0.5 + d.score))
    return keep if keep.area * FACE_SWITCH_RATIO >= best.area else best


def track_attention(
    path: str | Path,
    *,
    info: MediaInfo | None = None,
    sample_fps: float = 6.0,
    detector: Detector | None = None,
) -> Track:
    """Кадры → трек лица в пикселях ИСХОДНИКА.

    Детекция идёт на кадре, приведённом к растру анализа (`analyze_scale`): у
    YuNet ограниченный диапазон размеров лиц, и на крупном селфи в нативном
    разрешении он цепляется за часть лица вместо целого.
    """
    path = Path(path)
    info = info or probe(path)
    detector = detector or make_detector()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV не открыл файл: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or info.fps or 30.0
    step = max(1, int(round(src_fps / max(0.5, sample_fps))))
    scale = analyze_scale(info.width, info.height)

    track = Track(detector=detector.name)
    prev_gray: np.ndarray | None = None
    prev_face: Detection | None = None
    idx = 0
    faces_found = 0

    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        idx += 1
        if not ok or frame is None:
            continue

        small = _resize(frame, scale)
        track.analyzed = (small.shape[1], small.shape[0])
        t = (idx - 1) / src_fps
        best = pick_face(detector.detect(small), prev_face)
        if best is not None:
            track.faces.append(best.rescaled(1.0 / scale))
            track.motion_x.append(None)
            faces_found += 1
            prev_face = best
            prev_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            mx = _motion_center(prev_gray, gray) if prev_gray is not None else None
            track.faces.append(None)
            track.motion_x.append(mx / scale if mx is not None else None)
            prev_gray = gray
        track.times.append(t)

    cap.release()
    log.info(
        "трекинг: %d проб, лицо найдено в %d (%.0f%%), детектор=%s на %dx%d",
        len(track), faces_found, 100 * faces_found / max(1, len(track)), detector.name,
        track.analyzed[0], track.analyzed[1],
    )
    return track


def fill_gaps(values: Sequence[float | None], default: float) -> list[float]:
    """Пропуски — линейной интерполяцией между известными, края — удержанием."""
    n = len(values)
    out: list[float] = [default] * n
    if not any(v is not None for v in values):
        return out

    # Два линейных прохода вместо поиска соседей на каждом шаге: трёхминутный
    # ролик даёт больше тысячи проб, а рядов три — квадрат тут не нужен.
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

    for i in range(n):
        v = values[i]
        if v is not None:
            out[i] = float(v)
            continue
        a, b = prev_known[i], next_known[i]
        if a is not None and b is not None:
            va, vb = float(values[a]), float(values[b])  # type: ignore[arg-type]
            out[i] = va + (vb - va) * (i - a) / (b - a)
        elif a is not None:
            out[i] = float(values[a])  # type: ignore[arg-type]
        else:
            out[i] = float(values[b])  # type: ignore[arg-type]
    return out


def moving_average(values: Sequence[float], window: int) -> list[float]:
    """Скользящее среднее с симметричным окном и удержанием краёв."""
    if window <= 1 or not values:
        return list(values)
    half = window // 2
    n = len(values)
    out: list[float] = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def apply_deadzone(values: Sequence[float], threshold: float) -> list[float]:
    """Камера стоит, пока объект не ушёл дальше порога, дальше идёт за ним с лагом.

    Без мёртвой зоны центр дрожит на пиксель-другой каждый кадр и картинка
    «плывёт». Но гасить дрожь скачком нельзя: прежняя версия при срабатывании
    переставляла окно сразу на полную новую позицию, то есть ровно на порог
    (1.2 % ширины окна — 14 px на окне 1214), и `sendcmd` делал это мгновенно,
    без интерполяции. Получался тот же артефакт, что «ступенька масштаба» из
    AUDIT §8.2, только по горизонтали: из реального прогона
    `1.500-1.667 crop@c0 x 1307` после `x 1321` — 14 px за один кадр.

    Поэтому здесь люфт, а не защёлка: выход отстаёт от цели ровно на порог и
    дальше движется с её собственным шагом. Дрожь внутри порога по-прежнему не
    проходит, а начавшееся движение не даёт скачка больше шага самой цели.
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


# --------------------------------------------------------------------------- геометрия


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def _even(value: float, *, minimum: int = 2) -> int:
    """Чётное число пикселей: yuv420p не умеет нечётные размеры."""
    n = int(round(value))
    return max(minimum, n - n % 2)


def crop_size(width: int, height: int, target_ar: float = TARGET_AR) -> tuple[int, int]:
    """Наибольшее окно с нужным соотношением, влезающее в кадр. Чётные размеры."""
    src_ar = width / height
    if src_ar > target_ar:
        w = int(height * target_ar)
        h = height
    else:
        w = width
        h = int(width / target_ar)
    return (w - w % 2, h - h % 2)


def window_bounds(
    frame_w: int, frame_h: int, out_w: int, out_h: int, max_scale: float
) -> tuple[float, float]:
    """Допустимая высота окна кропа: (минимум, максимум) в пикселях исходника.

    Максимум — наибольшее окно нужного соотношения, влезающее в кадр: шире не
    бывает, кроп не добавляет того, чего не снято. Минимум — кламп апскейла: окно
    ниже `out_h / max_scale` пришлось бы растягивать сильнее, чем разрешено, и
    крупность добиралась бы мылом. Если кламп строже кадра (исходник мельче
    выхода, как любое 720p селфи), минимум опускается до максимума: выбора всё
    равно нет, а падать здесь не за что.
    """
    _, base_h = crop_size(frame_w, frame_h, out_w / out_h)
    hi = float(base_h)
    lo = min(hi, out_h / max(0.01, max_scale))
    return lo, hi


def window_height(
    face_h: float, eye_to_head: float, *, ratio: float, crop: Crop, lo: float, hi: float
) -> float:
    """Высота окна кропа из доли лица, с поправкой на макушку.

    `face_h / ratio` даёт ровно заказанную долю лица. Второе слагаемое — чистая
    геометрия: линия глаз стоит на `eye_line_ratio` высоты кадра, макушка обязана
    оказаться выше неё ещё на `headroom_ratio`, а значит окно не может быть ниже
    `(глаза − макушка) / (eye_line_ratio − headroom_ratio)`. Не влезает —
    уменьшаем крупность, то есть увеличиваем окно.
    """
    if face_h <= 0:
        return hi
    want = face_h / max(0.01, ratio)
    room = crop.eye_line_ratio - crop.headroom_ratio
    if room > 0.01 and eye_to_head > 0:
        want = max(want, eye_to_head / room)
    return _clamp(want, lo, hi)


def place_window(
    center_x: float,
    eyes: float,
    head_top: float,
    *,
    win_w: float,
    win_h: float,
    frame_w: int,
    frame_h: int,
    crop: Crop,
) -> tuple[float, float]:
    """Левый верхний угол окна кропа для одной пробы.

    По горизонтали — центр внимания. По вертикали — линия глаз на
    `eye_line_ratio` высоты кадра, а не центр: голова по центру читается как
    кадр из видеозвонка. Если окно оказалось слишком тесным (кламп по апскейлу не
    дал его увеличить), макушка важнее линии глаз — срезанный лоб не лечится
    ничем, а съехавшая на пару процентов линия глаз незаметна.
    """
    x = center_x - win_w / 2
    y = eyes - crop.eye_line_ratio * win_h
    y = min(y, head_top - crop.headroom_ratio * win_h)
    return _clamp(x, 0, frame_w - win_w), _clamp(y, 0, frame_h - win_h)


def clamp_window(edges: Sequence[float], window: int, frame: int) -> list[int]:
    """Левая (верхняя) граница окна, прижатая внутрь кадра."""
    limit = max(0, frame - window)
    return [int(round(_clamp(e, 0, limit))) for e in edges]


def clamp_track(centers: Sequence[float], window: int, frame: int) -> list[int]:
    """Центр → левая граница окна, прижатая внутрь кадра."""
    return clamp_window([c - window / 2 for c in centers], window, frame)


# --------------------------------------------------------------------------- крупность


@dataclass(slots=True)
class Segment:
    """Кусок выходного таймлайна с постоянным размером окна кропа."""

    start: float
    end: float
    framing: str
    win_w: int
    win_h: int
    times: list[float] = field(default_factory=list)
    xs: list[int] = field(default_factory=list)
    ys: list[int] = field(default_factory=list)
    # Медианная высота лица НА ЭТОМ куске, пиксели исходника. Ради неё и выбран
    # размер окна, поэтому доля лица в логе обязана считаться от неё, а не от
    # медианы по всему ролику: куски и различаются тем, как далеко сидит человек.
    face_h: float = 0.0

    @property
    def static_x(self) -> bool:
        return len(set(self.xs)) <= 1

    @property
    def static_y(self) -> bool:
        return len(set(self.ys)) <= 1

    def face_share(self, face_h: float) -> float:
        """Доля лица по высоте кадра при этом окне — то самое число из DoD."""
        return face_h / self.win_h if self.win_h else 0.0


Span = tuple[float, float, str]


def _merge_spans(spans: Sequence[Span]) -> list[Span]:
    out: list[Span] = []
    for start, end, framing in spans:
        if end - start <= 1e-6:
            continue
        if out and out[-1][2] == framing and abs(out[-1][1] - start) < 1e-6:
            out[-1] = (out[-1][0], end, framing)
        else:
            out.append((start, end, framing))
    return out


def framing_spans(edl: EDL, *, applied_cuts: bool = True) -> list[Span]:
    """EDL → интервалы крупности в том времени, которое увидит `autocrop`.

    Кроп работает уже после нарезки, поэтому времена нужны выходные: kept-куски
    склеены встык, дырок между ними нет.

    `applied_cuts=False` значит, что паузы не вырезаны и настоящих резов в
    картинке нет вообще. Тогда любая смена крупности пришлась бы на середину
    непрерывного плана и читалась бы как самопроизвольный зум — весь ролик
    снимаем одной крупностью.
    """
    kept = sorted(edl.kept_cuts, key=lambda c: c.src_in)
    if not kept:
        return []
    if not applied_cuts:
        return [(0.0, edl.source_duration, "wide")]
    spans: list[Span] = []
    acc = 0.0
    for c in kept:
        spans.append((acc, acc + c.duration, c.framing))
        acc += c.duration
    return _merge_spans(spans)


# Насколько покрытие может разойтись с фактической длительностью, чтобы разницу
# можно было считать накопленным округлением кадров, а не другим таймлайном.
SPAN_DRIFT_TOLERANCE = 0.05


def _fit_spans(spans: Sequence[Span], duration: float) -> list[Span]:
    """Границы крупности — на кадровую сетку файла, который увидит кроп.

    `framing_spans` складывает ТОЧНЫЕ длительности EDL, а `cut.py` режет каждый
    кусок фильтром `fps=` и склеивает, то есть каждый кусок округляется до целого
    числа кадров. Ошибка копится по числу резов: на эталонном EDL три kept-куска
    (4.565 + 6.225 + 7.68 = 18.47 с) дают в cut.mp4 445 кадров при 24 fps, то есть
    18.54 с — плюс 1.7 кадра. Раньше эту разницу целиком съедал последний span, а
    все внутренние границы оставались ранними, и смена крупности приходилась не на
    рез, а на несколько кадров ВНУТРЬ предыдущего плана — то есть выглядела
    самопроизвольным зумом, ради запрета которого вся схема и построена.

    Растягиваем покрытие пропорционально: округление набегает примерно поровну на
    кусок, поэтому пропорция раскладывает его обратно почти без остатка. Правка
    применяется только когда расхождение похоже на округление кадров
    (SPAN_DRIFT_TOLERANCE); всё, что больше, — это другой таймлайн, и его честнее
    оставить клипу.
    """
    covered = spans[-1][1] - spans[0][0]
    if covered <= 0 or spans[0][0] > 1e-6:
        return list(spans)
    k = duration / covered
    if abs(k - 1.0) > SPAN_DRIFT_TOLERANCE:
        return list(spans)
    return [(s * k, e * k, f) for s, e, f in spans]


def normalize_spans(spans: Sequence[Span] | None, duration: float) -> list[Span]:
    """Интервалы крупности → непрерывное покрытие [0, duration] без дырок."""
    if duration <= 0:
        return [(0.0, 0.0, "wide")]
    if spans:
        spans = _fit_spans(sorted(spans, key=lambda sp: sp[0]), duration)
    clipped = [
        (max(0.0, s), min(duration, e), f if f in ("wide", "tight") else "wide")
        for s, e, f in sorted(spans or [], key=lambda sp: sp[0])
        if min(duration, e) - max(0.0, s) > 1e-6
    ]
    if not clipped:
        return [(0.0, duration, "wide")]

    out: list[Span] = []
    cursor = 0.0
    for start, end, framing in clipped:
        if start > cursor + 1e-6:
            # Дырку отдаём предыдущей крупности: смена обязана происходить на резе,
            # а в середине куска для неё нет повода.
            out.append((cursor, start, out[-1][2] if out else framing))
        out.append((max(cursor, start), end, framing))
        cursor = end
    if duration - cursor > 1e-6:
        out.append((cursor, duration, out[-1][2]))
    return _merge_spans(out)


def _close_windows(a: Segment, b: Segment, tolerance: float) -> bool:
    """Различима ли смена окна на глаз. Сравнение относительное, а не побитовое."""
    dh = abs(a.win_h - b.win_h) / max(1, a.win_h, b.win_h)
    dw = abs(a.win_w - b.win_w) / max(1, a.win_w, b.win_w)
    return max(dh, dw) <= tolerance


def _merge_segments(
    segments: Sequence[Segment], tolerance: float = FRAMING_MERGE_TOLERANCE
) -> list[Segment]:
    """Соседние куски с практически одинаковым окном — один кусок.

    Побитового равенства мало: на 4K-входе кламп `max_scale` сводит wide и tight
    почти в одно окно (1214x2158 против 1204x2140), и граф ради 0.8 % разницы
    уходил в split/trim/concat — тот самый путь, который до правки падал на SAR.
    Из двух окон оставляем БОЛЬШЕЕ: ошибиться в сторону более широкого кадра
    безопаснее, срезанное лицо не лечится ничем.
    """
    out: list[Segment] = []
    for seg in segments:
        if out and _close_windows(out[-1], seg, tolerance):
            out[-1].end = seg.end
            if seg.win_h > out[-1].win_h:
                out[-1].win_w, out[-1].win_h = seg.win_w, seg.win_h
            # Из двух лиц берём крупное: доля лица в логе должна показывать худший
            # случай на куске, а не тот, что оказался первым.
            out[-1].face_h = max(out[-1].face_h, seg.face_h)
        else:
            out.append(seg)
    return out


def plan_segments(
    track: Track,
    spans: Sequence[Span],
    *,
    info: MediaInfo,
    out_w: int,
    out_h: int,
    preset: Preset,
    smooth_seconds: float = 1.2,
    sample_fps: float = 6.0,
    deadzone_ratio: float = 0.012,
) -> list[Segment]:
    """Трек + крупность по кускам → готовые прямоугольники кропа.

    Сглаживается ПРЯМОУГОЛЬНИК, а не позиция лица, и сглаживается внутри куска:
    на резе крупность обязана меняться мгновенно, размазанная по секунде смена
    читается как самопроизвольный зум, то есть ровно как дефект из AUDIT §8.2.
    """
    crop = preset.crop
    lo, hi = window_bounds(info.width, info.height, out_w, out_h, crop.max_scale)
    ratios = {"wide": crop.face_height_ratio, "tight": crop.tight_face_height_ratio}

    # По горизонтали запасной ряд осмысленный — центр масс движения, а при полной
    # статике центр кадра. По вертикали запасного ряда нет: у движения нет линии
    # глаз, поэтому при полном отсутствии лиц окно центрируется явно (`place_at`),
    # а дефолты `eyes`/`heads` в геометрии не участвуют вовсе.
    no_faces = not any(f is not None for f in track.faces)
    cx = fill_gaps(
        [f.x if f is not None else m for f, m in zip(track.faces, track.motion_x)],
        info.width / 2,
    )
    eyes = fill_gaps([f.eyes if f is not None else None for f in track.faces], 0.0)
    heads = fill_gaps([f.head_top if f is not None else None for f in track.faces], 0.0)

    all_face_h = track.face_height()
    all_eye_to_head = track.eye_to_head()
    window = max(3, int(round(smooth_seconds * sample_fps)) | 1)

    def place_at(i: int, seg: Segment) -> tuple[float, float]:
        """Окно для одной пробы: по лицу, а когда лиц нет — по центру кадра.

        Раньше «нет лиц» выражалось запасными рядами eyes=0.36·H и head_top=0.05·H,
        то есть лицом во весь кадр. Правило макушки в `place_window` на таком ряде
        выигрывало и тащило окно ВВЕРХ: на 1080x2340 оно вставало на y=21 при
        центре 210, и низ кадра просто терялся — при том, что и комментарий, и
        строка лога обещали центр.
        """
        if no_faces:
            x = cx[i] - seg.win_w / 2
            return (
                _clamp(x, 0, info.width - seg.win_w),
                max(0.0, (info.height - seg.win_h) / 2),
            )
        return place_window(
            cx[i], eyes[i], heads[i],
            win_w=seg.win_w, win_h=seg.win_h,
            frame_w=info.width, frame_h=info.height, crop=crop,
        )

    def samples_in(start: float, end: float) -> list[int]:
        idx = [i for i, t in enumerate(track.times) if start - 1e-6 <= t < end - 1e-6]
        if not idx and track.times:
            # Кусок короче шага семплирования — берём ближайшую пробу.
            idx = [min(range(len(track.times)), key=lambda i: abs(track.times[i] - start))]
        return idx

    segments: list[Segment] = []
    for start, end, framing in spans:
        idx = samples_in(start, end)
        face_h = track.face_height(idx[0], idx[-1] + 1) if idx else 0.0
        eye_to_head = track.eye_to_head(idx[0], idx[-1] + 1) if idx else 0.0
        win_h = window_height(
            face_h or all_face_h,
            eye_to_head or all_eye_to_head,
            ratio=ratios.get(framing, crop.face_height_ratio),
            crop=crop,
            lo=lo,
            hi=hi,
        )
        win_h_px = min(_even(win_h), _even(info.height))
        win_w_px = min(_even(win_h_px * out_w / out_h), _even(info.width))
        # Ширина упёрлась в кадр — соотношение держим высотой: выход всё равно
        # масштабируется в 9:16, и растянутое лицо заметнее срезанного низа.
        win_h_px = min(win_h_px, _even(win_w_px * out_h / out_w))
        segments.append(
            Segment(
                start=start, end=end, framing=framing,
                win_w=win_w_px, win_h=win_h_px, face_h=face_h or all_face_h,
            )
        )

    segments = _merge_segments(segments)
    covered = segments[-1].end - segments[0].start if segments else 0.0
    # n кусков занимают n−1 интервалов между сменами — отсюда «+1», см. комментарий
    # к MIN_SECONDS_PER_FRAMING. Без него потолок бил по плану самого режиссёра на
    # роликах 10-20 с.
    allowed = max(MIN_FRAMING_SEGMENTS, int(covered / MIN_SECONDS_PER_FRAMING) + 1)
    if len(segments) > allowed:
        log.warning(
            "крупность меняется %d раз на %.0f с (потолок %d) — свожу к одному окну",
            len(segments), covered, allowed,
        )
        widest = max(segments, key=lambda s: s.win_h)
        segments = [
            Segment(
                0.0, segments[-1].end, widest.framing, widest.win_w, widest.win_h,
                face_h=all_face_h,
            )
        ]

    for seg in segments:
        idx = samples_in(seg.start, seg.end)
        if not idx:
            # Проб нет вообще (пустой трек): ни лица, ни движения. Ставим окно по
            # центру кадра. Прежде здесь стояли числа 0.4 и 0.15 вместо геометрии
            # пресета — но брать её тут не из чего, лица нет.
            x, y = (info.width - seg.win_w) / 2, (info.height - seg.win_h) / 2
            seg.times = [seg.start]
            seg.xs = clamp_window([x], seg.win_w, info.width)
            seg.ys = clamp_window([y], seg.win_h, info.height)
            continue

        raw = [place_at(i, seg) for i in idx]
        xs = apply_deadzone(
            moving_average([p[0] for p in raw], window), seg.win_w * deadzone_ratio
        )
        ys = apply_deadzone(
            moving_average([p[1] for p in raw], window), seg.win_h * deadzone_ratio
        )
        seg.times = [track.times[i] for i in idx]
        seg.times[0] = seg.start  # первая команда — ровно на резе, не позже него
        seg.xs = clamp_window(xs, seg.win_w, info.width)
        seg.ys = clamp_window(ys, seg.win_h, info.height)
    return segments


# --------------------------------------------------------------------------- трек как данные


def window_at(seg: Segment, t: float, *, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Окно кропа на момент `t` внутри куска: (x, y, w, h) в пикселях исходника.

    Повторяет то, что реально сделает ffmpeg, а не то, что задумано. Отсюда две
    неочевидности:

    1. Окно во весь кадр — это ОТСУТСТВИЕ фильтра `crop` (`build_crop_filter`
       возвращает пустую строку), то есть окно (0, 0, кадр). Взять `seg.xs[0]`
       было бы неверно: там лежит зажатая клампом позиция, которой в графе нет.
    2. `sendcmd` — ступенька, а не интерполяция: значение держится от своей
       команды до следующей. Поэтому здесь поиск последней команды до `t`, а не
       линейное смешивание соседних.
    """
    if seg.win_w >= frame_w and seg.win_h >= frame_h:
        return 0, 0, frame_w, frame_h
    if not seg.times:
        return seg.xs[0], seg.ys[0], seg.win_w, seg.win_h
    i = 0
    for j, cmd_t in enumerate(seg.times):
        if cmd_t <= t + 1e-9:
            i = j
        else:
            break
    return seg.xs[i], seg.ys[i], seg.win_w, seg.win_h


def project_track(
    track: Track,
    segments: Sequence[Segment],
    *,
    info: MediaInfo,
    fps: int,
    duration: float | None = None,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
) -> FaceTrack:
    """Трек в пикселях исходника → нормализованный трек выходного кадра (М1).

    Порядок обязателен и он ровно такой, как в MOTION_PLAN: сначала заполняются
    пропуски детекции, потом ряд пересчитывается на кадровую сетку, и только
    после этого сглаживается. Сглаживать до пересчёта нельзя — one euro ведёт
    срез от частоты, а у проб она в пять раз ниже кадровой.

    Про резы отдельно: их здесь нет и быть не должно. `autocrop` читает уже
    нарезанное видео, поэтому `track.times` с самого начала лежат в выходном
    времени — та самая ловушка «применить трек до резов» закрыта не кодом, а
    местом вызова.

    Пустой результат (лиц не нашли вовсе) — законный: возвращается трек без
    точек, а не выдуманный центр кадра.
    """
    if not track.times or not segments:
        return FaceTrack(fps=fps, detector=track.detector, analyzed=track.analyzed)
    if not any(f is not None for f in track.faces):
        log.info("трек не сохранён: лиц не найдено ни в одной пробе")
        return FaceTrack(fps=fps, detector=track.detector, analyzed=track.analyzed)

    def segment_at(t: float) -> Segment:
        for seg in segments:
            if seg.start - 1e-6 <= t < seg.end - 1e-6:
                return seg
        return segments[-1] if t >= segments[-1].start else segments[0]

    # --- 1. пробы в нормализованные координаты выходного кадра
    cx: list[float | None] = []
    cy: list[float | None] = []
    fh: list[float | None] = []
    conf: list[float] = []
    for t, face in zip(track.times, track.faces):
        if face is None:
            cx.append(None)
            cy.append(None)
            fh.append(None)
            conf.append(MISSING_CONF)
            continue
        seg = segment_at(t)
        wx, wy, ww, wh = window_at(seg, t, frame_w=info.width, frame_h=info.height)
        cx.append((face.x - wx) / ww)
        cy.append((face.y - wy) / wh)
        fh.append(face.h / wh)
        conf.append(_clamp(face.score, 0.0, 1.0))

    # --- 2. пропуски: внутри разрыва линейно, по краям удержанием
    cx_f, cy_f, fh_f = fill_track_gaps(cx), fill_track_gaps(cy), fill_track_gaps(fh)

    # --- 3. на кадровую сетку выходного ролика
    total = duration if duration is not None else info.duration
    frames = max(1, int(round(total * fps)))
    at = [i / fps for i in range(frames)]
    cx_r = resample(track.times, cx_f, at)
    cy_r = resample(track.times, cy_f, at)
    fh_r = resample(track.times, fh_f, at)
    conf_r = resample(track.times, conf, at)

    # --- 4. сглаживание. Только геометрия: уверенность не сигнал положения, и
    # размазывать провал в дырке значит прятать его от предохранителей М2.
    # `scale` обязателен: `beta` умножается на скорость, поэтому в долях кадра
    # стартовое 0.007 не работает вовсе — см. REF_WIDTH в src/core/track.py.
    euro = {"fps": fps, "min_cutoff": min_cutoff, "beta": beta}
    cx_s = smooth(cx_r, scale=REF_WIDTH, **euro)
    cy_s = smooth(cy_r, scale=REF_HEIGHT, **euro)
    fh_s = smooth(fh_r, scale=REF_HEIGHT, **euro)

    points = [
        TrackPoint(
            f=i,
            cx=_clamp(cx_s[i], -1.0, 2.0),
            cy=_clamp(cy_s[i], -1.0, 2.0),
            h=_clamp(fh_s[i], 0.0, 4.0),
            conf=_clamp(conf_r[i], 0.0, 1.0),
        )
        for i in range(frames)
    ]
    weak = sum(1 for p in points if p.conf < 0.5)
    log.info(
        "трек лица: %d точек при %d fps, слабых (conf<0.5) %d (%.0f %%), детектор=%s на %dx%d",
        len(points), fps, weak, 100 * weak / max(1, len(points)),
        track.detector, track.analyzed[0], track.analyzed[1],
    )
    return FaceTrack(
        fps=fps, points=points, detector=track.detector, analyzed=track.analyzed
    )


# --------------------------------------------------------------------------- ffmpeg


def build_sendcmd(
    times: Sequence[float],
    xs: Sequence[int],
    ys: Sequence[int],
    *,
    target: str = "crop",
    static_x: bool = False,
    static_y: bool = False,
) -> str:
    """Скрипт для sendcmd: перестановка окна кропа по времени.

    Адрес команды — имя экземпляра фильтра (`crop@c0`), потому что кусков разной
    крупности несколько и каждый живёт в своей ветке графа.
    """
    lines: list[str] = []
    prev: tuple[int, int] | None = None
    for i, t in enumerate(times):
        pair = (xs[i], ys[i])
        if pair == prev:
            continue
        end = times[i + 1] if i + 1 < len(times) else t + 1.0
        cmds = []
        if not static_x:
            cmds.append(f"{target} x {xs[i]}")
        if not static_y:
            cmds.append(f"{target} y {ys[i]}")
        if cmds:
            lines.append(f"{max(0.0, t):.3f}-{end:.3f} " + ", ".join(cmds) + ";")
        prev = pair
    return "\n".join(lines) + ("\n" if lines else "")


def grade_filter(grade: Grade) -> str:
    """`preset.grade` → кусок цепочки ffmpeg. Пустая строка — цветокор выключен.

    Телефонное видео приходит тёмным и плоским; `eq` вытягивает его ровно
    настолько, чтобы этого не было видно. Насыщенность выше 1.2 на лицах уходит в
    красноту — потолок держит пресет, а не код, но молчать про его нарушение
    нельзя: краснота на коже с одного кадра не читается, а на ролике целиком уже
    поздно.

    Виньетка — тот же случай, и она НЕ входит в «этого не видно»: замеренный по
    серой карте профиль (см. VIGNETTE_VISIBLE) показывает, что на значении из
    пресета glass верх и низ 9:16-кадра гаснут до 45 % от центра. Решение
    художественное, поэтому код его не меняет, но и молчать про него не должен.
    """
    if not grade.enabled:
        return ""
    if grade.saturation > SATURATION_RED_SKIN:
        log.warning(
            "цветокор: saturation=%.2f выше %.2f — на лицах это уходит в красноту",
            grade.saturation, SATURATION_RED_SKIN,
        )
    if grade.vignette > VIGNETTE_VISIBLE:
        log.warning(
            "цветокор: vignette=%.2f выше %.2f — это уже видимая рамка, а не «лёгкий "
            "цветокор»: на 9:16 верх и низ кадра гаснут примерно до половины яркости "
            "центра, вместе с макушкой и полосой субтитров",
            grade.vignette, VIGNETTE_VISIBLE,
        )
    parts: list[str] = []
    if (grade.contrast, grade.saturation, grade.gamma) != (1.0, 1.0, 1.0):
        parts.append(
            f"eq=contrast={grade.contrast:.4g}"
            f":saturation={grade.saturation:.4g}"
            f":gamma={grade.gamma:.4g}"
        )
    if grade.vignette > 0:
        # angle у ffmpeg в радианах, дефолт PI/5 ≈ 0.628 — то же, что в пресете.
        parts.append(f"vignette=angle={grade.vignette:.4g}")
    return ",".join(parts)


def _escape_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def build_crop_filter(
    segments: Sequence[Segment],
    *,
    frame_w: int,
    frame_h: int,
    out_w: int,
    out_h: int,
    grade: str = "",
    cmd_file: Path | None = None,
) -> str:
    """filter_complex: своя крупность на каждом куске, склейка, цветокор.

    Размер окна `crop` менять на ходу нельзя: `sendcmd` с командами `w`/`h`
    подвешивает ffmpeg 6.1 намертво (проверено на 6.1.1). Поэтому каждый кусок —
    отдельная ветка графа с собственным фильтром постоянного размера, а `trim`
    режет их по выходному времени встык. Границы непрерывны, а `trim` не включает
    свой конец, значит каждый кадр попадает ровно в одну ветку: ни дублей, ни
    потерь, и звук не съезжает относительно картинки.

    Цветокор — один раз после склейки, а не в каждой ветке: он одинаковый для
    всего ролика, а `eq` на каждом кадре стоит денег.

    `setsar=1` стоит В КАЖДОЙ ветке, а не только в хвосте. Окна разной крупности
    округляются до чётных пикселей и потому не идеально 9:16; `scale` доводит
    пропорцию не размером, а SAR, и ветки приходят в `concat` с разным SAR
    (1:1 против 4384:4383). `concat` такое не склеивает, а падает: «Input link
    parameters do not match» — то есть ровно смена крупности, ради которой ветки
    и заведены, роняла весь этап.
    """
    parts: list[str] = []
    head = "[0:v]"
    if cmd_file is not None:
        head += f"sendcmd=f='{_escape_path(cmd_file)}',"

    def crop_of(i: int, seg: Segment) -> str:
        if seg.win_w >= frame_w and seg.win_h >= frame_h:
            return ""  # окно во весь кадр — резать нечего
        return f"crop@c{i}=w={seg.win_w}:h={seg.win_h}:x={seg.xs[0]}:y={seg.ys[0]},"

    scale = f"scale={out_w}:{out_h}:flags=lanczos,setsar=1"
    if len(segments) == 1:
        parts.append(f"{head}{crop_of(0, segments[0])}{scale}[vcat]")
    else:
        n = len(segments)
        parts.append(head + f"split={n}" + "".join(f"[b{i}]" for i in range(n)))
        for i, seg in enumerate(segments):
            rng = f"trim=start={seg.start:.4f}"
            if i < n - 1:  # у последнего конец не ставим: хвост дороже точности
                rng += f":end={seg.end:.4f}"
            parts.append(f"[b{i}]{rng},setpts=PTS-STARTPTS,{crop_of(i, seg)}{scale}[s{i}]")
        parts.append("".join(f"[s{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vcat]")

    tail = f"{grade}," if grade else ""
    parts.append(f"[vcat]{tail}setsar=1[vout]")
    return ";".join(parts)


# --------------------------------------------------------------------------- вход


def warn_missing_inputs(
    preset: Preset | str | None, framing: Sequence[Span] | None, cfg: Settings
) -> None:
    """Молчаливая деградация Ф3 и Ф6 — в лог, а не в тишину.

    Оба входа необязательные, и обе подстановки по умолчанию тихо отменяют целую
    фазу: без `preset` цветокор берётся из `cfg.default_preset`, а не из пресета
    задачи (а пресеты различаются именно цветокором — glass даёт виньетку, bold
    нет), без `framing` весь ролик снимается одной крупностью. Сегодня вызовы в
    `pipeline/job.py` и `worker/tasks.py` не передают ни того, ни другого.
    """
    if preset is None:
        log.warning(
            "автокроп: пресет не передан — беру %s из конфига. Цветокор задачи (Ф3) "
            "при этом теряется: пресеты различаются именно им",
            cfg.default_preset,
        )
    if framing is None:
        log.warning(
            "автокроп: крупность не передана — весь ролик снимается одной, Ф6 "
            "выключена. Ожидается framing=framing_spans(edl, applied_cuts=...)"
        )


def autocrop(
    src: str | Path,
    dst: str | Path,
    *,
    info: MediaInfo | None = None,
    out_width: int | None = None,
    out_height: int | None = None,
    preset: Preset | str | None = None,
    framing: Sequence[Span] | None = None,
    smooth_seconds: float = 1.2,
    sample_fps: float = 6.0,
    deadzone_ratio: float = 0.012,
    crf: int = 18,
    cfg: Settings | None = None,
    work_dir: str | Path | None = None,
    track_out: str | Path | None = None,
) -> Path:
    """Горизонтальное (или любое) видео → вертикальное 9:16 с лицом в кадре.

    `framing` — интервалы крупности в выходном времени, см. `framing_spans`.
    Не передали — весь ролик снимается одной крупностью `wide`.

    `track_out` — куда положить трек лица (М1). Трек считается здесь не потому,
    что это удобно, а потому, что только здесь известны окна кропа: без них
    координаты выходного кадра не посчитать. Не передали — трек не сохраняется,
    поведение как до М1.

    Оба необязательных входа — `preset` и `framing` — умеют деградировать молча, и
    это уже стоило продукту всей Ф6: вызовы в `pipeline/job.py` и `worker/tasks.py`
    не передают ни того, ни другого, поэтому крупность до кадра не доезжает, а
    цветокор берётся из `cfg.default_preset`, а не из пресета задачи (ломая §0.4
    «сохранённый EDL + другой пресет = новое видео»). Правка вызовов — за
    владельцем job.py/tasks.py, но деградация обязана быть видимой отсюда.
    """
    cfg = cfg or default_settings
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = info or probe(src)
    out_w = out_width or cfg.render_width
    out_h = out_height or cfg.render_height
    warn_missing_inputs(preset, framing, cfg)
    if preset is None or isinstance(preset, str):
        preset = load_preset(preset or cfg.default_preset)
    grade = grade_filter(preset.grade)

    spans = normalize_spans(framing, info.duration)
    lo, hi = window_bounds(info.width, info.height, out_w, out_h, preset.crop.max_scale)
    base_w, base_h = crop_size(info.width, info.height, out_w / out_h)
    # Выбора нет, когда окно зажато клампом апскейла и при этом равно кадру: ни
    # сузить (запрещено), ни подвинуть (некуда). Тогда трек нужен только чтобы
    # честно записать в лог, что получилось, — хватит одной пробы в секунду.
    frozen = hi - lo < 1.0 and base_w >= info.width and base_h >= info.height
    track = track_attention(src, info=info, sample_fps=1.0 if frozen else sample_fps)

    if frozen:
        segments = [Segment(0.0, info.duration, "wide", base_w, base_h, [0.0], [0], [0])]
    else:
        segments = plan_segments(
            track, spans, info=info, out_w=out_w, out_h=out_h, preset=preset,
            smooth_seconds=smooth_seconds, sample_fps=sample_fps,
            deadzone_ratio=deadzone_ratio,
        )

    work = Path(work_dir) if work_dir else dst.parent
    work.mkdir(parents=True, exist_ok=True)
    script = "".join(
        build_sendcmd(
            seg.times, seg.xs, seg.ys, target=f"crop@c{i}",
            static_x=seg.static_x or seg.win_w >= info.width,
            static_y=seg.static_y or seg.win_h >= info.height,
        )
        for i, seg in enumerate(segments)
    )
    cmd_file: Path | None = None
    if script.strip():
        cmd_file = work / f"{dst.stem}.crop.sendcmd"
        cmd_file.write_text(script, encoding="utf-8")

    vf = build_crop_filter(
        segments, frame_w=info.width, frame_h=info.height,
        out_w=out_w, out_h=out_h, grade=grade, cmd_file=cmd_file,
    )
    if track_out is not None:
        out_fps = int(round(info.fps)) if info.fps and info.fps > 0 else cfg.render_fps
        face_track = project_track(
            track, segments, info=info, fps=out_fps,
            min_cutoff=preset.crop.track_min_cutoff, beta=preset.crop.track_beta,
        )
        Path(track_out).parent.mkdir(parents=True, exist_ok=True)
        Path(track_out).write_text(
            face_track.model_dump_json(indent=1), encoding="utf-8"
        )

    _log_framing(track, segments, preset.crop, info=info)
    log.info(
        "автокроп: кусков %d [%s] → %dx%d, цветокор=%s, детектор=%s, проб=%d",
        len(segments),
        " ".join(f"{s.framing} {s.win_w}x{s.win_h}" for s in segments),
        out_w, out_h, grade or "выкл", track.detector, len(track),
    )

    args = ["-i", str(src), "-filter_complex", vf, "-map", "[vout]"]
    if info.has_audio:
        args += ["-map", "0:a?", "-c:a", "copy"]
    args += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    ffmpeg(args, timeout=7200)
    return dst


def _log_framing(
    track: Track, segments: Sequence[Segment], crop: Crop, *, info: MediaInfo
) -> None:
    """Что получилось по крупности — числом, а не на глаз.

    Отдельно про «не сделали хуже»: если исходник снят крупнее цели, доля лица
    останется выше коридора, и это не дефект кадрирования, а его физический
    предел. Такое обязано быть видно в логе, иначе разбор упрётся в «почему 47 %».

    Число печатается вместе с растром анализа: доля лица — свойство пары «кадр +
    растр», и без него её не с чем сравнивать (samples/README.md).
    """
    overall = track.face_height()
    if overall <= 0:
        # Две разные ветки, и раньше обе описывались одной строкой про центр кадра.
        # По горизонтали окно ведёт центр масс движения и запросто проходит сотню
        # пикселей мимо центра (замер: 507 → 655 при центре 656), по вертикали —
        # действительно центр кадра. Врать про это нельзя: разбор «почему кадр
        # уехал» начинается ровно с этой строки.
        xs = [x for seg in segments for x in seg.xs]
        spread = (max(xs) - min(xs)) if xs else 0
        if any(m is not None for m in track.motion_x):
            log.info(
                "кадрирование: лиц не найдено, по горизонтали окно ведёт центр масс "
                "движения (ход %d px), по вертикали — центр кадра", spread,
            )
        else:
            log.info("кадрирование: лиц и движения не найдено, окно по центру кадра")
        return
    for seg in segments:
        target = crop.tight_face_height_ratio if seg.framing == "tight" else crop.face_height_ratio
        face_h = seg.face_h or overall
        log.info(
            "кадрирование %.1f-%.1f с (%s): доля лица %.2f при цели %.2f, окно %dx%d "
            "(детектор %s на %dx%d)",
            seg.start, seg.end, seg.framing, seg.face_share(face_h), target,
            seg.win_w, seg.win_h, track.detector, track.analyzed[0], track.analyzed[1],
        )
    widest = max(segments, key=lambda s: s.face_share(s.face_h or overall))
    share = widest.face_share(widest.face_h or overall)
    if share > crop.face_height_ratio * 1.15:
        log.warning(
            "исходник %dx%d снят крупнее цели: доля лица %.2f вместо %.2f. Кроп умеет "
            "только сужать кадр, а кламп max_scale=%.2f запрещает добирать разницу "
            "апскейлом — крупность оставлена как снято",
            info.width, info.height, share, crop.face_height_ratio, crop.max_scale,
        )


if __name__ == "__main__":  # python -m src.pipeline.crop in.mp4 out.mp4
    import sys

    from src.core.logging import setup_logging

    setup_logging(default_settings.log_level)
    print(autocrop(sys.argv[1], sys.argv[2]))
