"""M6: автокроп в 9:16 с трекингом центра внимания.

Схема: детекция по кадрам → заполнение пропусков → сглаживание скользящим
средним → мёртвая зона → динамический crop через `sendcmd`.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from src.core.config import Settings, settings as default_settings
from src.core.proc import ffmpeg
from src.pipeline.probe import MediaInfo, probe

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "assets" / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

TARGET_AR = 9 / 16


@dataclass(slots=True)
class Detection:
    x: float  # центр по горизонтали, пиксели исходника
    y: float
    score: float
    area: float
    # Размеры бокса нужны отдельно от площади: доля лица меряется по ВЫСОТЕ кадра,
    # а из area её не вытащить. Значения по умолчанию — чтобы старый код и тесты,
    # конструирующие Detection только по центру и площади, продолжали работать.
    w: float = 0.0
    h: float = 0.0


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
            out.append(
                Detection(
                    x=float(fx) + float(fw) / 2,
                    y=float(fy) + float(fh) / 2,
                    score=float(f[-1]),
                    area=float(fw) * float(fh),
                    w=float(fw),
                    h=float(fh),
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
) -> tuple[list[float], list[float | None], list[float | None], str]:
    """Кадры → (времена, центр X, центр Y, имя детектора). None — не нашли."""
    path = Path(path)
    info = info or probe(path)
    detector = detector or make_detector()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV не открыл файл: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or info.fps or 30.0
    step = max(1, int(round(src_fps / max(0.5, sample_fps))))
    scale = min(1.0, analyze_width / max(1, info.width))

    times: list[float] = []
    xs: list[float | None] = []
    ys: list[float | None] = []
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
            xs.append(best.x / scale)
            ys.append(best.y / scale)
            faces_found += 1
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            mx = _motion_center(prev_gray, gray) if prev_gray is not None else None
            xs.append(mx / scale if mx is not None else None)
            ys.append(None)
            prev_gray = gray
        if dets:
            prev_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        times.append(t)

    cap.release()
    log.info(
        "трекинг: %d проб, лицо найдено в %d (%.0f%%), детектор=%s",
        len(times), faces_found, 100 * faces_found / max(1, len(times)), detector.name,
    )
    return times, xs, ys, detector.name


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


def clamp_track(centers: Sequence[float], window: int, frame: int) -> list[int]:
    """Центр → левая граница окна, прижатая внутрь кадра."""
    limit = max(0, frame - window)
    return [int(round(min(max(c - window / 2, 0), limit))) for c in centers]


def build_sendcmd(
    times: Sequence[float],
    xs: Sequence[int],
    ys: Sequence[int],
    *,
    static_x: bool = False,
    static_y: bool = False,
) -> str:
    """Скрипт для sendcmd: перестановка окна кропа по времени."""
    lines: list[str] = []
    prev: tuple[int, int] | None = None
    for i, t in enumerate(times):
        pair = (xs[i], ys[i])
        if pair == prev:
            continue
        end = times[i + 1] if i + 1 < len(times) else t + 1.0
        cmds = []
        if not static_x:
            cmds.append(f"crop x {xs[i]}")
        if not static_y:
            cmds.append(f"crop y {ys[i]}")
        if cmds:
            lines.append(f"{max(0.0, t):.3f}-{end:.3f} " + ", ".join(cmds) + ";")
        prev = pair
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- вход


def autocrop(
    src: str | Path,
    dst: str | Path,
    *,
    info: MediaInfo | None = None,
    out_width: int | None = None,
    out_height: int | None = None,
    smooth_seconds: float = 1.2,
    sample_fps: float = 6.0,
    deadzone_ratio: float = 0.012,
    crf: int = 18,
    cfg: Settings | None = None,
    work_dir: str | Path | None = None,
) -> Path:
    """Горизонтальное (или любое) видео → вертикальное 9:16 с лицом в кадре."""
    cfg = cfg or default_settings
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = info or probe(src)
    out_w = out_width or cfg.render_width
    out_h = out_height or cfg.render_height

    win_w, win_h = crop_size(info.width, info.height, out_w / out_h)
    static_x = win_w >= info.width
    static_y = win_h >= info.height

    if static_x and static_y:
        # Кадр уже нужного соотношения — только масштабируем.
        log.info("кадр уже %dx%d, кроп не нужен", info.width, info.height)
        ffmpeg(
            ["-i", str(src), "-vf", f"scale={out_w}:{out_h}:flags=lanczos",
             "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(dst)],
            timeout=7200,
        )
        return dst

    times, raw_x, raw_y, detector_name = track_attention(src, info=info, sample_fps=sample_fps)

    if not times:
        centers_x = [info.width / 2]
        centers_y = [info.height / 2]
        times = [0.0]
    else:
        centers_x = fill_gaps(raw_x, info.width / 2)
        centers_y = fill_gaps(raw_y, info.height / 2)
        window = max(3, int(round(smooth_seconds * sample_fps)) | 1)
        centers_x = moving_average(centers_x, window)
        centers_y = moving_average(centers_y, window)
        centers_x = apply_deadzone(centers_x, info.width * deadzone_ratio)
        centers_y = apply_deadzone(centers_y, info.height * deadzone_ratio)

    # Лицо приятнее держать в верхней трети, а не строго по центру.
    if not static_y:
        centers_y = [c - win_h * 0.06 for c in centers_y]

    xs = clamp_track(centers_x, win_w, info.width)
    ys = clamp_track(centers_y, win_h, info.height)

    work = Path(work_dir) if work_dir else dst.parent
    work.mkdir(parents=True, exist_ok=True)
    cmd_file = work / f"{dst.stem}.crop.sendcmd"
    cmd_file.write_text(
        build_sendcmd(times, xs, ys, static_x=static_x, static_y=static_y), encoding="utf-8"
    )

    escaped = str(cmd_file).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    vf = (
        f"sendcmd=f='{escaped}',"
        f"crop=w={win_w}:h={win_h}:x={xs[0]}:y={ys[0]},"
        f"scale={out_w}:{out_h}:flags=lanczos,setsar=1"
    )
    log.info(
        "автокроп: окно %dx%d → %dx%d, детектор=%s, проб=%d",
        win_w, win_h, out_w, out_h, detector_name, len(times),
    )
    args = ["-i", str(src), "-vf", vf, "-c:v", "libx264", "-preset", "medium",
            "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if info.has_audio:
        args += ["-c:a", "copy"]
    args += ["-movflags", "+faststart", str(dst)]
    ffmpeg(args, timeout=7200)
    return dst


if __name__ == "__main__":  # python -m src.pipeline.crop in.mp4 out.mp4
    import sys

    from src.core.logging import setup_logging

    setup_logging(default_settings.log_level)
    print(autocrop(sys.argv[1], sys.argv[2]))
