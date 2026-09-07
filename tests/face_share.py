"""Замер доли лица по высоте кадра (Фаза 0 «замерить базу», ориентир 20-25 %).

Не тест, а инструмент: семплирует кадры по всей длительности, на каждом берёт
самое крупное лицо и печатает распределение share = высота бокса / высота кадра.
Нужен для A/B между прогонами пайплайна: числа сравниваются друг с другом, а
глазами проверяются по --annotate.

    python tests/face_share.py data/work/<job>/result.mp4
    python tests/face_share.py data/work/<job>/source.mp4 --fps-sample 2 \
        --at 5 --annotate /tmp/at5.png --json /tmp/face_share.json

Про честность числа. YuNet отдаёт ТЕСНЫЙ бокс лица: примерно от бровей до
подбородка, без лба, волос и ушей. Реальная голова по высоте примерно в 1.4 раза
больше бокса (HEAD_FACTOR), поэтому share — не «доля головы» из ТЗ монтажёра, а
воспроизводимый прокси. Сравнивать его осмысленно с таким же числом другого
прогона, а не с абстрактным идеалом; коридор 20-25 % в отчёте — это коридор
именно для share YuNet, и он же переносится на Haar только приблизительно
(каскад Хаара даёт бокс заметно крупнее).

Про разрешение анализа (важно, иначе число врёт). Детекция идёт на кадре,
ужатом до --analyze-width (по умолчанию 640, как в `track_attention`), а не на
исходном. У YuNet ограниченный диапазон размеров лиц: на кадре 1080x1920 лицо
высотой под тысячу пикселей он перестаёт видеть целиком и цепляется за его
часть — на эталонном result.mp4 нативная детекция даёт медиану 25 %, тогда как
на том же кадре в 640 px — 49 %, и второе совпадает с замером по 720p-исходнику
(47 %). Доля — отношение, от масштаба она не зависит; фиксированная ширина
анализа нужна, чтобы числа для файлов разного разрешения были сопоставимы и
чтобы мерилось ровно то, что видит трекер автокропа.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.crop import Detection, make_detector  # noqa: E402
from src.pipeline.probe import probe  # noqa: E402

# Целевой коридор доли лица по высоте кадра (доли единицы, не проценты).
TARGET_LOW = 0.20
TARGET_HIGH = 0.25

# Ширина кадра для детекции. Совпадает с analyze_width в track_attention:
# меряем то же, что видит трекер, и получаем сравнимые числа для 720p и 1080p.
ANALYZE_WIDTH = 640

# Во сколько раз голова (с волосами и лбом) выше тесного бокса YuNet. Оценка,
# нужна только чтобы в сводке было видно порядок величины «доли головы».
HEAD_FACTOR = 1.4


@dataclass(slots=True)
class Sample:
    """Один семпл: запрошенное время, фактическое время кадра и самое крупное лицо."""

    t: float
    t_actual: float | None  # None — кадр не прочитан (например, запрос за концом файла)
    share: float | None  # None — лицо не найдено
    box: list[float] | None  # [x, y, w, h] в пикселях кадра
    score: float | None


def _seek(cap: cv2.VideoCapture, t: float) -> tuple[np.ndarray | None, float | None]:
    """Кадр в момент t по времени, а не по номеру.

    Позиционирование по CAP_PROP_POS_MSEC переживает переменный fps и вырезанные
    паузы (после cut.mp4 номер кадра уже ничего не значит), но попадает не точно
    в запрошенную секунду — поэтому возвращаем и фактическое время кадра.

    Позицию читаем строго ПОСЛЕ read(): сразу после set() бэкенд FFMPEG отдаёт
    мусор (доли секунды вместо запрошенной), а после чтения — метку выданного
    кадра. За концом файла read() отдаёт False и сбрасывает позицию в 0 — выдавать
    такое за фактическое время нельзя, поэтому там None.
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, None
    return frame, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0


def _downscale(frame: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """Кадр, ужатый до заданной ширины, и применённый масштаб. width<=0 — как есть."""
    w = frame.shape[1]
    if width <= 0 or width >= w:
        return frame, 1.0
    scale = width / w
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def _largest(dets: list[Detection]) -> Detection | None:
    """Самое крупное лицо. Именно площадь, без веса score: меряем геометрию, не уверенность."""
    valid = [d for d in dets if d.h > 0]
    return max(valid, key=lambda d: d.area) if valid else None


def measure(
    cap: cv2.VideoCapture, detector, t: float, analyze_width: int
) -> tuple[Sample, np.ndarray | None]:
    """Замер на одном кадре. Возвращает семпл и исходный кадр (нужен для --annotate).

    Детекция — на ужатой копии, а доля считается от высоты той же копии: масштаб
    сокращается. Бокс в семпле пересчитан обратно в пиксели исходного кадра.
    """
    frame, actual = _seek(cap, t)
    if frame is None:
        return Sample(t=t, t_actual=actual, share=None, box=None, score=None), None
    small, scale = _downscale(frame, analyze_width)
    best = _largest(detector.detect(small))
    if best is None:
        return Sample(t=t, t_actual=actual, share=None, box=None, score=None), frame
    sample = Sample(
        t=t,
        t_actual=actual,
        share=best.h / small.shape[0],
        box=[
            (best.x - best.w / 2) / scale,
            (best.y - best.h / 2) / scale,
            best.w / scale,
            best.h / scale,
        ],
        score=best.score,
    )
    return sample, frame


def stats(shares: list[float], n_samples: int) -> dict:
    """Сводка по найденным лицам. Пустой список — все поля None, но coverage честный."""
    if not shares:
        return {
            "n_samples": n_samples,
            "n_faces": 0,
            "coverage": 0.0,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
            "mean": None,
        }
    arr = np.asarray(shares, dtype=np.float64)
    return {
        "n_samples": n_samples,
        "n_faces": len(shares),
        "coverage": len(shares) / max(1, n_samples),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def verdict(value: float | None) -> str:
    if value is None:
        return "нет данных"
    if value < TARGET_LOW:
        return "ниже"
    if value > TARGET_HIGH:
        return "выше"
    return "в коридоре"


def annotate(frame: np.ndarray, sample: Sample, dst: Path) -> None:
    """Кадр с боксом и подписью — чтобы число проверялось глазами, а не на веру.

    Подписи латиницей: шрифты Хаара в OpenCV не умеют кириллицу и рисуют «?».
    """
    img = frame.copy()
    h, w = img.shape[:2]
    stamp = f"t={sample.t_actual:.2f}s" if sample.t_actual is not None else f"t~{sample.t:.2f}s"
    if sample.box is not None and sample.share is not None:
        x, y, bw, bh = (int(round(v)) for v in sample.box)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 220, 0), max(2, w // 400))
        # Горизонтали через весь кадр: видно, какую его часть занимает бокс по высоте.
        for line_y in (y, y + bh):
            cv2.line(img, (0, line_y), (w, line_y), (0, 220, 0), max(1, w // 900))
        label = f"share={sample.share * 100:.1f}% h={bh}px/{h}px {stamp}"
    else:
        label = f"no face {stamp}"

    scale = w / 1000.0
    thick = max(1, int(round(2 * scale)))
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = int(8 * scale)
    cv2.rectangle(img, (0, 0), (tw + 2 * pad, th + base + 2 * pad), (0, 0, 0), -1)
    cv2.putText(
        img, label, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick,
        cv2.LINE_AA,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), img):
        raise RuntimeError(f"не записался кадр: {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(description="доля лица по высоте кадра")
    ap.add_argument("video", help="путь к видео")
    ap.add_argument("--fps-sample", type=float, default=2.0, help="семплов в секунду (по умолч. 2)")
    ap.add_argument(
        "--at", type=float, action="append", default=None,
        help="секунда для отдельного замера, можно повторять (по умолч. 5.0)",
    )
    ap.add_argument(
        "--analyze-width", type=int, default=ANALYZE_WIDTH,
        help=f"ширина кадра для детекции, 0 — нативная (по умолч. {ANALYZE_WIDTH})",
    )
    ap.add_argument("--annotate", help="PNG с боксом на кадре первого --at")
    ap.add_argument("--json", dest="json_path", help="файл для полного отчёта")
    args = ap.parse_args()

    at = args.at if args.at else [5.0]
    path = Path(args.video)
    info = probe(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"OpenCV не открыл файл: {path}", file=sys.stderr)
        return 1
    detector = make_detector()

    # Размеры берём из реального кадра, а не из ffprobe: OpenCV отдаёт картинку
    # уже с применённым поворотом, и именно её видит детектор.
    first, _ = _seek(cap, 0.0)
    if first is None:
        print(f"не читается ни один кадр: {path}", file=sys.stderr)
        cap.release()
        return 1
    frame_h, frame_w = first.shape[:2]
    analyzed = _downscale(first, args.analyze_width)[0].shape[:2]

    step = 1.0 / max(0.05, args.fps_sample)
    n = max(1, int(info.duration / step))
    samples: list[Sample] = [
        measure(cap, detector, i * step, args.analyze_width)[0] for i in range(n)
    ]

    at_samples: list[Sample] = []
    for i, t in enumerate(at):
        sample, frame = measure(cap, detector, t, args.analyze_width)
        at_samples.append(sample)
        if args.annotate and i == 0:
            if frame is None:
                print(f"кадр на {t:.2f} c не прочитан, --annotate пропущен", file=sys.stderr)
            else:
                annotate(frame, sample, Path(args.annotate))
    cap.release()

    shares = [s.share for s in samples if s.share is not None]
    st = stats(shares, len(samples))
    med = st["median"]
    report = {
        "video": str(path.resolve()),
        "detector": detector.name,
        "frame": {"width": frame_w, "height": frame_h},
        "analyzed_frame": {"width": analyzed[1], "height": analyzed[0]},
        "duration": info.duration,
        "fps_sample": args.fps_sample,
        "stats": st,
        "target": {
            "low": TARGET_LOW,
            "high": TARGET_HIGH,
            "metric": "median",
            "verdict": verdict(med),
            "head_factor": HEAD_FACTOR,
            "head_share_estimate": None if med is None else med * HEAD_FACTOR,
        },
        "at": [asdict(s) for s in at_samples],
        "samples": [asdict(s) for s in samples],
        "note": (
            "share = высота бокса детектора / высота кадра; бокс YuNet тесный "
            f"(брови-подбородок), голова примерно в {HEAD_FACTOR} раза выше. "
            f"Детекция на кадре шириной {analyzed[1]} px — числа сравнимы между "
            "файлами разного разрешения только при одинаковом analyze_width"
        ),
    }

    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def pct(v: float | None) -> str:
        return "  --  " if v is None else f"{v * 100:5.1f}%"

    print(
        f"{path.name}: {frame_w}x{frame_h}, {info.duration:.1f} c, детектор={detector.name} "
        f"на {analyzed[1]}x{analyzed[0]}, семплов={st['n_samples']} @ {args.fps_sample} fps"
    )
    print(
        f"  share по высоте: медиана {pct(med)}  p10 {pct(st['p10'])}  p90 {pct(st['p90'])}  "
        f"min {pct(st['min'])}  max {pct(st['max'])}  среднее {pct(st['mean'])}"
    )
    print(
        f"  лицо найдено в {st['coverage'] * 100:.0f}% кадров "
        f"({st['n_faces']}/{st['n_samples']})"
    )
    for s in at_samples:
        if s.t_actual is None:
            print(f"  на {s.t:.2f} c: кадр не прочитан (за концом файла?)")
            continue
        extra = "" if s.share is None else f" (голова ~{pct(s.share * HEAD_FACTOR).strip()})"
        print(f"  на {s.t:.2f} c (кадр {s.t_actual:.2f} c): {pct(s.share)}{extra}")
    print(
        f"  коридор {TARGET_LOW * 100:.0f}-{TARGET_HIGH * 100:.0f}% по медиане: "
        f"{report['target']['verdict']}"
    )
    if args.annotate:
        print(f"  кадр с боксом: {args.annotate}")
    if args.json_path:
        print(f"  отчёт: {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
