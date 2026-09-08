"""DoD М1: попадает ли сохранённый трек в лицо на готовом ролике.

Не тест, а инструмент. Тесты в `tests/test_track.py` проверяют нашу арифметику
на синтетике; здесь проверяется утверждение, которое синтетикой не проверить:
что точка из `track.json` действительно лежит на лице в кадре РЕЗУЛЬТАТА.

Проверка независимая: детектор запускается заново, уже по выходному видео, и
сравнивается не с собой, а с записанным треком. Если пересчёт из координат
исходника в координаты выходного кадра где-то врёт — окном не того куска,
пропущенной командой `sendcmd`, забытым масштабом — точка промахнётся мимо
бокса, и это будет видно числом.

    python tests/face_track_check.py data/work/<job>/vertical.mp4 \\
        data/work/<job>/track.json --json /tmp/track-check.json
    python tests/face_track_check.py out.mp4 track.json --annotate /tmp/at5.png --at 5

Два числа на выходе:

- **доля кадров, где точка внутри бокса лица** — цель 100 % на кадрах, где лицо
  вообще нашлось. Кадры без детекции считаются отдельно и в знаменатель не
  идут: их промах — свойство детектора на этом кадре, а не трека.
- **дрожание центра на статичном плане, px между соседними кадрами** — цель
  меньше 2 px. «Статичный» определяется по самому треку: кадры, вокруг которых
  средняя скорость за полсекунды ниже `--static-speed` px/кадр. Почему именно
  так, а не по шагу и не по смещению, — в докстринге `jitter`: оба простых
  определения врут, и каждое по-своему.

Растр детекции здесь тот же, что у `tests/face_share.py` и у трекера автокропа
(высота 1138): доля лица — свойство пары «кадр + растр», и мерить второй раз
другим растром значит сравнивать несравнимое.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.track import FaceTrack, REF_HEIGHT, REF_WIDTH  # noqa: E402
from src.pipeline.crop import Detection, make_detector, pick_face  # noqa: E402

ANALYZE_HEIGHT = 1138


def _detect(frame: np.ndarray, detector, prev: Detection | None) -> Detection | None:
    """Лицо на кадре в координатах САМОГО кадра, а не растра анализа."""
    h, w = frame.shape[:2]
    scale = ANALYZE_HEIGHT / h if h else 1.0
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=interp)
    best = pick_face(detector.detect(small), prev)
    return best.rescaled(1.0 / scale) if best is not None else None


def check(video: Path, track: FaceTrack, *, every: int = 1) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV не открыл файл: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = make_detector()

    hits = 0
    checked = 0
    blind = 0
    misses: list[dict] = []
    prev: Detection | None = None
    frame_idx = -1

    while True:
        if not cap.grab():
            break
        frame_idx += 1
        if frame_idx % every:
            continue
        point = track.at(frame_idx)
        if point is None:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue
        face = _detect(frame, detector, prev)
        if face is None:
            blind += 1
            continue
        prev = face
        checked += 1
        px, py = point.cx * width, point.cy * height
        inside = (
            abs(px - face.x) <= face.w / 2 and abs(py - face.y) <= face.h / 2
        )
        if inside:
            hits += 1
        else:
            misses.append(
                {
                    "frame": frame_idx,
                    "track_px": [round(px, 1), round(py, 1)],
                    "face_px": [round(face.x, 1), round(face.y, 1)],
                    "off_x_px": round(px - face.x, 1),
                    "off_y_px": round(py - face.y, 1),
                    "box": [round(face.w, 1), round(face.h, 1)],
                }
            )
    cap.release()

    return {
        "video": str(video),
        "frame": [width, height],
        "analyzed_height": ANALYZE_HEIGHT,
        "detector": detector.name,
        "frames_checked": checked,
        "frames_without_face": blind,
        "hits": hits,
        "hit_rate": (hits / checked) if checked else None,
        "misses": misses[:20],
    }


def jitter(track: FaceTrack, *, static_travel: float) -> dict:
    """Дрожание центра на статичных отрезках, в пикселях опорного кадра.

    Опорный кадр — тот же 1080x1920, в котором MOTION_PLAN задаёт и порог «меньше
    2 px», и порог скорости для предохранителей М2. Числа трека нормализованные,
    поэтому без общего опорного кадра «2 px» не значат ничего.

    «Статичный» определяется по СРЕДНЕЙ СКОРОСТИ за полсекунды вокруг кадра, и
    оба более простых определения пришлось выбросить, потому что каждое врало
    по-своему:

    1. «Шаг меньше порога» — тавтология: отобрав шаги меньше двух пикселей,
       максимум по ним получишь меньше двух при любом качестве трека. Первый
       замер так и выдал 1.993 px при пороге 2.0.
    2. «Смещение за окно меньше порога» — ловится на РАЗВОРОТЕ. Голова шла вниз
       по 7.7 px/кадр, на кадре 409 сменила направление и вернулась: смещение за
       окно 3.2 px, то есть «стоит», а шаг в вершине 3.09 px объявлялся дрожью.
       Это не дрожь, это движение.

    Средняя скорость за окно от разворота не страдает: длина пути в вершине
    большая, кадр из статики выпадает — и в статике остаётся только то, что ею
    и является.
    """
    pts = track.points
    if len(pts) < 3:
        return {"samples": 0}
    xs = [p.cx * REF_WIDTH for p in pts]
    ys = [p.cy * REF_HEIGHT for p in pts]
    steps = [
        ((xs[i + 1] - xs[i]) ** 2 + (ys[i + 1] - ys[i]) ** 2) ** 0.5
        for i in range(len(pts) - 1)
    ]

    half = max(1, track.fps // 2)
    static_steps: list[float] = []
    for i in range(len(steps)):
        lo, hi = max(0, i - half), min(len(steps), i + half + 1)
        window = steps[lo:hi]
        if sum(window) / len(window) < static_travel:
            static_steps.append(steps[i])

    return {
        "samples": len(steps),
        "window_frames": half * 2,
        "static_speed_px_per_frame": static_travel,
        "static_samples": len(static_steps),
        "static_share": len(static_steps) / len(steps) if steps else None,
        "static_median_px": round(median(static_steps), 3) if static_steps else None,
        "static_max_px": round(max(static_steps), 3) if static_steps else None,
        "overall_median_px": round(median(steps), 3) if steps else None,
        "overall_max_px": round(max(steps), 3) if steps else None,
    }


def annotate(video: Path, track: FaceTrack, at: float, dst: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or track.fps
    frame_idx = int(round(at * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"не прочитался кадр на {at} с")
    point = track.at(frame_idx)
    if point is None:
        raise SystemExit(f"в треке нет кадра {frame_idx}")
    h, w = frame.shape[:2]
    cx, cy = int(point.cx * w), int(point.cy * h)
    half = int(point.h * h / 2)
    cv2.circle(frame, (cx, cy), 12, (0, 220, 255), -1)
    cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), (0, 220, 255), 3)
    cv2.putText(
        frame, f"f={frame_idx} conf={point.conf:.2f} h={point.h:.3f}",
        (24, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 3,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), frame)


def main() -> int:
    ap = argparse.ArgumentParser(description="проверка трека лица на готовом ролике")
    ap.add_argument("video", help="выходное видео (vertical.mp4 или result.mp4)")
    ap.add_argument("track", help="track.json")
    ap.add_argument("--every", type=int, default=3, help="проверять каждый n-й кадр (по умолч. 3)")
    ap.add_argument(
        "--static-speed", type=float, default=1.5,
        help="средняя скорость за полсекунды, ниже которой план статичен, px/кадр (по умолч. 1.5)",
    )
    ap.add_argument("--at", type=float, help="секунда для --annotate")
    ap.add_argument("--annotate", help="PNG с точкой трека на кадре")
    ap.add_argument("--json", dest="json_path", help="файл для полного отчёта")
    args = ap.parse_args()

    video = Path(args.video)
    track = FaceTrack.model_validate_json(Path(args.track).read_text(encoding="utf-8"))
    if not track.points:
        print("трек пустой: лиц не найдено ни в одной пробе")
        return 1

    hit = check(video, track, every=max(1, args.every))
    jit = jitter(track, static_travel=args.static_speed)
    report = {"hits": hit, "jitter": jit, "fps": track.fps, "points": len(track.points)}

    if args.annotate:
        annotate(video, track, args.at if args.at is not None else 0.0, Path(args.annotate))
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    rate = hit["hit_rate"]
    print(f"трек: {len(track.points)} точек при {track.fps} fps, детектор {hit['detector']}")
    print(
        f"попадание в лицо: {hit['hits']}/{hit['frames_checked']} "
        f"({'—' if rate is None else f'{rate * 100:.1f} %'}), "
        f"кадров без детекции {hit['frames_without_face']}"
    )
    print(
        f"дрожание на статике: медиана {jit.get('static_median_px')} px, "
        f"максимум {jit.get('static_max_px')} px "
        f"(статичных {jit.get('static_samples')}/{jit.get('samples')})"
    )
    if hit["misses"]:
        m = hit["misses"][0]
        print(f"первый промах: кадр {m['frame']}, мимо на {m['off_x_px']}, {m['off_y_px']} px")
    ok = rate is not None and rate >= 0.99 and (jit.get("static_max_px") or 0) < 2.0
    print("DoD М1:", "выполнен" if ok else "НЕ выполнен")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
