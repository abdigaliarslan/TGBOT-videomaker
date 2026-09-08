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
from src.pipeline.probe import MediaInfo, probe

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "assets" / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

TARGET_AR = 9 / 16

# Бокс YuNet тесный: примерно от бровей до подбородка, без лба, волос и ушей.
# tests/face_share.py оценивает всю голову в 1.4 высоты бокса — те же 0.4 сверху,
# отложенные от верхней границы. На этой оценке стоит правило «макушка не обрезана».
HEAD_ABOVE_BOX = 0.4
# Запасная линия глаз, когда у детектора нет ландмарок (Haar): глаза чуть ниже
# верха бокса брови-подбородок.
EYES_BELOW_BOX_TOP = 0.3
# Потолок числа кусков разной крупности. Больше — это уже не монтаж, а мельтешение;
# такой EDL сводим к одному окну на весь ролик.
MAX_FRAMING_SEGMENTS = 12


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
    # Линия глаз из ландмарок детектора. None — считаем оценкой по боксу.
    eye_y: float | None = None

    @property
    def top(self) -> float:
        """Верхняя граница бокса детектора (примерно брови)."""
        return self.y - self.h / 2

    @property
    def head_top(self) -> float:
        """Оценка макушки: волосы и лоб выше тесного бокса."""
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
            # f[4:6] и f[6:8] — правый и левый глаз. Линия глаз из ландмарок честнее
            # любой доли от высоты бокса: на наклоне головы бокс едет, глаза нет.
            eye_y = (float(f[5]) + float(f[7])) / 2 if len(f) >= 15 else None
            out.append(
                Detection(
                    x=float(fx) + float(fw) / 2,
                    y=float(fy) + float(fh) / 2,
                    score=float(f[-1]),
                    area=float(fw) * float(fh),
                    w=float(fw),
                    h=float(fh),
                    eye_y=eye_y,
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
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48))
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


def track_attention(
    path: str | Path,
    *,
    info: MediaInfo | None = None,
    sample_fps: float = 6.0,
    detector: Detector | None = None,
    analyze_width: int = 640,
) -> Track:
    """Кадры → трек лица в пикселях ИСХОДНИКА.

    Детекция идёт на кадре, ужатом до `analyze_width`: у YuNet ограниченный
    диапазон размеров лиц, и на крупном селфи в нативном разрешении он цепляется
    за часть лица вместо целого. tests/face_share.py меряет тем же способом и с
    той же шириной — иначе доли лица для файлов разного разрешения несравнимы.
    """
    path = Path(path)
    info = info or probe(path)
    detector = detector or make_detector()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV не открыл файл: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or info.fps or 30.0
    step = max(1, int(round(src_fps / max(0.5, sample_fps))))
    scale = min(1.0, analyze_width / max(1, info.width))

    track = Track(detector=detector.name)
    prev_gray: np.ndarray | None = None
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

        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
        t = (idx - 1) / src_fps
        dets = detector.detect(small)
        if dets:
            best = max(dets, key=lambda d: d.area * (0.5 + d.score))
            track.faces.append(best.rescaled(1.0 / scale))
            track.motion_x.append(None)
            faces_found += 1
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
        "трекинг: %d проб, лицо найдено в %d (%.0f%%), детектор=%s",
        len(track), faces_found, 100 * faces_found / max(1, len(track)), detector.name,
    )
    return track


def fill_gaps(values: Sequence[float | None], default: float) -> list[float]:
    """Пропуски — линейной интерполяцией между известными, края — удержанием."""
    out: list[float] = [default] * len(values)
    known = [i for i, v in enumerate(values) if v is not None]
    if not known:
        return out
    for i in range(len(values)):
        v = values[i]
        if v is not None:
            out[i] = float(v)
            continue
        before = [k for k in known if k < i]
        after = [k for k in known if k > i]
        if before and after:
            a, b = before[-1], after[0]
            va, vb = float(values[a]), float(values[b])  # type: ignore[arg-type]
            out[i] = va + (vb - va) * (i - a) / (b - a)
        elif before:
            out[i] = float(values[before[-1]])  # type: ignore[arg-type]
        else:
            out[i] = float(values[after[0]])  # type: ignore[arg-type]
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
    """Камера стоит, пока объект не ушёл дальше порога.

    Без этого центр дрожит на пиксель-другой каждый кадр, и картинка «плывёт».
    """
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(v if abs(v - out[-1]) > threshold else out[-1])
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


def normalize_spans(spans: Sequence[Span] | None, duration: float) -> list[Span]:
    """Интервалы крупности → непрерывное покрытие [0, duration] без дырок."""
    if duration <= 0:
        return [(0.0, 0.0, "wide")]
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


def _merge_segments(segments: Sequence[Segment]) -> list[Segment]:
    """Соседние куски с одинаковым окном — один кусок: резать граф незачем."""
    out: list[Segment] = []
    for seg in segments:
        if out and (out[-1].win_w, out[-1].win_h) == (seg.win_w, seg.win_h):
            out[-1].end = seg.end
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

    # Ряды без пропусков. Дефолты — на случай, когда лиц не было вовсе: тогда окно
    # встаёт примерно по центру кадра, как и до Ф6.
    cx = fill_gaps(
        [f.x if f is not None else m for f, m in zip(track.faces, track.motion_x)],
        info.width / 2,
    )
    eyes = fill_gaps(
        [f.eyes if f is not None else None for f in track.faces],
        info.height * crop.eye_line_ratio,
    )
    heads = fill_gaps(
        [f.head_top if f is not None else None for f in track.faces],
        info.height * crop.headroom_ratio,
    )

    all_face_h = track.face_height()
    all_eye_to_head = track.eye_to_head()
    window = max(3, int(round(smooth_seconds * sample_fps)) | 1)

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
            Segment(start=start, end=end, framing=framing, win_w=win_w_px, win_h=win_h_px)
        )

    segments = _merge_segments(segments)
    if len(segments) > MAX_FRAMING_SEGMENTS:
        log.info("крупность меняется %d раз — свожу к одному окну", len(segments))
        widest = max(segments, key=lambda s: s.win_h)
        segments = [
            Segment(0.0, segments[-1].end, widest.framing, widest.win_w, widest.win_h)
        ]

    for seg in segments:
        idx = samples_in(seg.start, seg.end)
        if not idx:
            x, y = place_window(
                info.width / 2, info.height * 0.4, info.height * 0.15,
                win_w=seg.win_w, win_h=seg.win_h,
                frame_w=info.width, frame_h=info.height, crop=crop,
            )
            seg.times = [seg.start]
            seg.xs = clamp_window([x], seg.win_w, info.width)
            seg.ys = clamp_window([y], seg.win_h, info.height)
            continue

        raw = [
            place_window(
                cx[i], eyes[i], heads[i],
                win_w=seg.win_w, win_h=seg.win_h,
                frame_w=info.width, frame_h=info.height, crop=crop,
            )
            for i in idx
        ]
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
    красноту — потолок держит пресет, а не код, поэтому здесь только формат.
    """
    if not grade.enabled:
        return ""
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
    """
    parts: list[str] = []
    head = "[0:v]"
    if cmd_file is not None:
        head += f"sendcmd=f='{_escape_path(cmd_file)}',"

    def crop_of(i: int, seg: Segment) -> str:
        if seg.win_w >= frame_w and seg.win_h >= frame_h:
            return ""  # окно во весь кадр — резать нечего
        return f"crop@c{i}=w={seg.win_w}:h={seg.win_h}:x={seg.xs[0]}:y={seg.ys[0]},"

    scale = f"scale={out_w}:{out_h}:flags=lanczos"
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
) -> Path:
    """Горизонтальное (или любое) видео → вертикальное 9:16 с лицом в кадре.

    `framing` — интервалы крупности в выходном времени, см. `framing_spans`.
    Не передали — весь ролик снимается одной крупностью `wide`.
    """
    cfg = cfg or default_settings
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = info or probe(src)
    out_w = out_width or cfg.render_width
    out_h = out_height or cfg.render_height
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
    """
    face_h = track.face_height()
    if face_h <= 0:
        log.info("кадрирование: лиц не найдено, окно стоит по центру кадра")
        return
    for seg in segments:
        target = crop.tight_face_height_ratio if seg.framing == "tight" else crop.face_height_ratio
        log.info(
            "кадрирование %.1f-%.1f с (%s): доля лица %.2f при цели %.2f, окно %dx%d",
            seg.start, seg.end, seg.framing, seg.face_share(face_h), target,
            seg.win_w, seg.win_h,
        )
    widest = max(segments, key=lambda s: s.win_h)
    if widest.face_share(face_h) > crop.face_height_ratio * 1.15:
        log.warning(
            "исходник %dx%d снят крупнее цели: доля лица %.2f вместо %.2f. Кроп умеет "
            "только сужать кадр, а кламп max_scale=%.2f запрещает добирать разницу "
            "апскейлом — крупность оставлена как снято",
            info.width, info.height, widest.face_share(face_h),
            crop.face_height_ratio, crop.max_scale,
        )


if __name__ == "__main__":  # python -m src.pipeline.crop in.mp4 out.mp4
    import sys

    from src.core.logging import setup_logging

    setup_logging(default_settings.log_level)
    print(autocrop(sys.argv[1], sys.argv[2]))
