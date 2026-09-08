"""Замер доли лица по высоте кадра (Фаза 0 «замерить базу», ориентир 20-25 %).

Не тест, а инструмент: семплирует кадры по всей длительности, на каждом берёт
самое крупное лицо и печатает распределение share = высота бокса / высота кадра
анализа. Нужен для A/B между прогонами пайплайна: числа сравниваются друг с
другом, а глазами проверяются по --annotate.

    python tests/face_share.py data/work/<job>/result.mp4
    python tests/face_share.py data/work/<job>/source.mp4 --fps-sample 2 \
        --at 5 --annotate /tmp/at5.png --json /tmp/face_share.json

Что именно меряется (и чего число НЕ значит). Величина — не свойство видео, а
свойство пары «видео + растр анализа». Бокс YuNet сильно зависит от пиксельного
размера лица: на одном и том же кадре result.mp4 (1080x1920) медиана share идёт
25.2 % (нативно) → 28.1 % (960) → 40.5 % (800) → 49.0 % (640) → 56.2 % (480) →
60.2 % (320) — разброс в 2.4 раза без единого изменения в содержимом. Поэтому
сравнивать можно только замеры с ОДИНАКОВЫМ analyzed_frame, и он печатается в
первой строке сводки как часть идентичности замера. Совпадение чисел исходника
и выхода («47 % против 49 %») — следствие общего растра, а не подтверждение
«правильного» ответа: на растре 720 та же пара даёт 46.8 % против 45.2 %.

Почему нормируем по ВЫСОТЕ. Метрика — доля по высоте, значит пиксельная высота
лица на анализе = share × analyze_height. Фиксируем высоту — и лицо с одной и
той же истинной долей всегда даёт детектору один и тот же пиксельный размер,
независимо от соотношения сторон. При фиксации ШИРИНЫ 16:9-файл ушёл бы на
растр 640x360, а 9:16 — на 640x1138, и один и тот же кадр, добитый до 16:9
чёрными полями (доля по высоте не изменилась), давал бы 56.2 % вместо 48.4 %.
С фиксацией высоты 1138 та же пара даёт 49.0 % и 48.3 %. Дефолт 1138 — это
высота 9:16-кадра шириной 640, то есть тот же рабочий режим, что у трекера
автокропа в `track_attention`; для вертикальных роликов растр не изменился.
`--analyze-width` оставлен для свипов, но между разными пропорциями его числа
несравнимы.

Кадр приводится к запрошенному растру в ОБЕ стороны, включая апскейл: иначе
исходник уже запрошенного молча анализировался бы нативно (720p → 50.0 %,
тот же контент в 480p → 59.5 %). Апскейл убирает эту дыру, но не возвращает
утраченную детализацию: на общем растре 640x1138 720p даёт 50.9 %, а его же
480p-копия 46.2 %. То есть даже при одинаковом analyzed_frame остаётся зависимость
от резкости источника ~5 п.п. — числа разных исходников сравнивать осторожно.

Про HEAD_FACTOR. Раньше в докстринге стояло, что бокс YuNet идёт «от бровей до
подбородка». На дефолтном растре это неверно: на кадре t=5 из result.mp4 бокс
занимает 37.5-84.1 % высоты, тогда как брови ~44-46 %, а подбородок ~88 %, то
есть верх бокса — в волосах, а низ выше подбородка. Множитель 1.4 повторно
добавляет уже входящий в бокс лоб и не компенсирует срезанный подбородок,
поэтому «доля головы» осталась только в JSON и помечена как грубая прикидка.

Коридор 20-25 % — коридор для share YuNet на каноничном растре. Для Haar он не
переносится (на одном и том же файле YuNet даёт 47 %, Haar — 24.7 %, и вердикт
переворачивается), поэтому для не-YuNet вердикт не считается вовсе.
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

# Высота кадра для детекции. 1138 = высота 9:16-кадра шириной 640, то есть тот
# же рабочий режим, что у analyze_width=640 в track_attention, но нормировка по
# высоте — она не зависит от соотношения сторон (см. докстринг).
ANALYZE_HEIGHT = 1138

# Во сколько раз голова (с волосами и лбом) выше бокса YuNet. Некалиброванная
# прикидка, живёт только в JSON: на каноничном растре бокс уже включает лоб.
HEAD_FACTOR = 1.4

# Расширения, которые OpenCV умеет писать. Проверяем ДО замера: imwrite на
# неизвестном расширении кидает cv2.error, и полчаса семплирования пропадают.
ANNOTATE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Вердикт по коридору осмыслен только для детектора, на котором коридор мерился.
VERDICT_DETECTOR = "yunet"


@dataclass(slots=True)
class Sample:
    """Один семпл: запрошенное время, фактическое время кадра и самое крупное лицо."""

    t: float
    t_actual: float | None  # None — кадр не прочитан (обрыв потока, битый кадр)
    share: float | None  # None — лицо не найдено
    box: list[float] | None  # [x, y, w, h] в пикселях кадра
    score: float | None


def _seek(cap: cv2.VideoCapture, t: float) -> tuple[np.ndarray | None, float | None]:
    """Кадр в момент t по времени, а не по номеру. Только для точечных --at.

    Позиционирование по CAP_PROP_POS_MSEC переживает переменный fps и вырезанные
    паузы (после cut.mp4 номер кадра уже ничего не значит), но попадает не точно
    в запрошенную секунду — поэтому возвращаем и фактическое время кадра.

    Позицию читаем строго ПОСЛЕ read(): сразу после set() бэкенд FFMPEG отдаёт
    мусор (доли секунды вместо запрошенной), а после чтения — метку выданного
    кадра. За концом файла read() отдаёт False и сбрасывает позицию в 0 — выдавать
    такое за фактическое время нельзя, поэтому там None.

    Сам seek не проверяет, что попал куда просили: на файле без индекса OpenCV
    молча отдаёт первый кадр с ok=True. Расхождение ловится снаружи по t_actual.
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, None
    return frame, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0


def _rescale(frame: np.ndarray, *, height: int = 0, width: int = 0) -> tuple[np.ndarray, float]:
    """Кадр, приведённый РОВНО к запрошенному растру, и применённый масштаб.

    Масштабируем в обе стороны: и вниз, и вверх. Ранний выход «источник уже
    меньше запрошенного — оставим как есть» означал бы, что растр анализа зависит
    от разрешения входа, а вместе с ним и число (720p → 50.0 %, тот же контент в
    480p → 59.5 % при одинаковых флагах). height/width <= 0 — нативный растр.
    """
    h, w = frame.shape[:2]
    if height > 0:
        scale = height / h
    elif width > 0:
        scale = width / w
    else:
        return frame, 1.0
    if abs(scale - 1.0) < 1e-9:
        return frame, 1.0
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=interp), scale


def _largest(dets: list[Detection]) -> Detection | None:
    """Самое крупное лицо. Именно площадь, без веса score: меряем геометрию, не уверенность."""
    valid = [d for d in dets if d.h > 0]
    return max(valid, key=lambda d: d.area) if valid else None


def _degenerate(dets: list[Detection]) -> bool:
    """Детектор что-то вернул, но без размеров бокса — мерить нечего."""
    return bool(dets) and not any(d.h > 0 for d in dets)


def measure_frame(
    frame: np.ndarray, detector, t: float, t_actual: float | None, *, height: int, width: int
) -> tuple[Sample, bool]:
    """Замер на готовом кадре. Возвращает семпл и признак «бокс без размеров».

    Детекция — на приведённой копии, доля считается от высоты той же копии:
    масштаб сокращается. Бокс в семпле пересчитан в пиксели исходного кадра.
    """
    small, scale = _rescale(frame, height=height, width=width)
    dets = detector.detect(small)
    best = _largest(dets)
    if best is None:
        return Sample(t=t, t_actual=t_actual, share=None, box=None, score=None), _degenerate(dets)
    sample = Sample(
        t=t,
        t_actual=t_actual,
        share=best.h / small.shape[0],
        box=[
            (best.x - best.w / 2) / scale,
            (best.y - best.h / 2) / scale,
            best.w / scale,
            best.h / scale,
        ],
        score=best.score,
    )
    return sample, False


def measure_at(
    cap: cv2.VideoCapture, detector, t: float, *, height: int, width: int
) -> tuple[Sample, np.ndarray | None]:
    """Точечный замер по времени (для --at). Возвращает семпл и исходный кадр."""
    frame, actual = _seek(cap, t)
    if frame is None:
        return Sample(t=t, t_actual=None, share=None, box=None, score=None), None
    sample, _ = measure_frame(frame, detector, t, actual, height=height, width=width)
    return sample, frame


def sample_sequence(
    cap: cv2.VideoCapture, detector, *, fps_sample: float, src_fps: float, height: int, width: int
) -> tuple[list[Sample], int]:
    """Проход по видеопотоку последовательным grab/retrieve с шагом по кадрам.

    Точечный seek на каждый семпл (set(POS_MSEC) + read()) стоит примерно в шесть
    раз дороже: на 185-секундном ролике 369 кадров это 63 с против 11 с. Заодно
    последовательный проход чинит три ловушки seek-семплирования:

    - конец видеопотока виден сразу (grab() отдаёт False), поэтому не бывает
      «непрочитанных» семплов из хвоста, которых нет в видео, но есть в
      длительности контейнера (после `-c copy`-нарезки звук бывает длиннее);
    - на файле без индекса seek молча отдавал первый кадр снова и снова, и
      инструмент печатал уверенную статистику по одному кадру;
    - шаг меряется в кадрах, поэтому запросить семплов больше, чем есть кадров,
      физически нельзя — дублей в выборке не возникает.

    Фактическое время берём из POS_MSEC после retrieve(), а не из номера кадра:
    на файлах с вырезанными паузами номер уже ничего не значит.
    """
    step = max(1, int(round(src_fps / max(0.05, fps_sample))))
    out: list[Sample] = []
    degenerate = 0
    idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            t_nominal = idx / src_fps
            if not ok or frame is None:
                out.append(Sample(t=t_nominal, t_actual=None, share=None, box=None, score=None))
            else:
                t_actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                sample, bad = measure_frame(
                    frame, detector, t_nominal, t_actual, height=height, width=width
                )
                out.append(sample)
                degenerate += bad
        idx += 1
    return out, degenerate


def stats(samples: list[Sample]) -> dict:
    """Сводка. Три числа считаются раздельно: всего семплов, прочитано, с лицом.

    Раньше coverage делилась на все семплы сразу, и непрочитанные кадры выдавались
    за провал детектора: на файле с 5-секундным видео в 21-секундном контейнере
    выходило «лицо найдено в 23 % кадров», хотя из 10 прочитанных лицо нашлось
    в 10. Теперь coverage — доля ПРОЧИТАННЫХ кадров с лицом, а непрочитанные
    видны отдельным полем.
    """
    n_samples = len(samples)
    read = [s for s in samples if s.t_actual is not None]
    shares = [s.share for s in read if s.share is not None]
    times = [s.t_actual for s in read if s.t_actual is not None]
    base = {
        "n_samples": n_samples,
        "n_read": len(read),
        "n_unread": n_samples - len(read),
        "n_faces": len(shares),
        "coverage": len(shares) / len(read) if read else 0.0,
        "covered_from": min(times) if times else None,
        "covered_to": max(times) if times else None,
    }
    if not shares:
        return base | {"median": None, "p10": None, "p90": None,
                       "min": None, "max": None, "mean": None}
    arr = np.asarray(shares, dtype=np.float64)
    return base | {
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def verdict(value: float | None, detector_name: str) -> str:
    """Вердикт по коридору. Для чужого детектора коридор не переносится."""
    if detector_name != VERDICT_DETECTOR:
        return f"не применимо (детектор {detector_name})"
    if value is None:
        return "нет данных"
    if value < TARGET_LOW:
        return "ниже"
    if value > TARGET_HIGH:
        return "выше"
    return "в коридоре"


def annotate(frame: np.ndarray, sample: Sample, dst: Path, analyzed: tuple[int, int]) -> None:
    """Кадр с боксом и подписью — чтобы число проверялось глазами, а не на веру.

    Подписи латиницей: шрифты Хаара в OpenCV не умеют кириллицу и рисуют «?».
    Растр анализа тоже в подписи: без него по картинке не понять, к какому замеру
    она относится, а от растра число зависит вдвое.
    """
    img = frame.copy()
    h, w = img.shape[:2]
    stamp = f"t={sample.t_actual:.2f}s" if sample.t_actual is not None else f"t~{sample.t:.2f}s"
    raster = f"@{analyzed[1]}x{analyzed[0]}"
    if sample.box is not None and sample.share is not None:
        x, y, bw, bh = (int(round(v)) for v in sample.box)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 220, 0), max(2, w // 400))
        # Горизонтали через весь кадр: видно, какую его часть занимает бокс по высоте.
        for line_y in (y, y + bh):
            cv2.line(img, (0, line_y), (w, line_y), (0, 220, 0), max(1, w // 900))
        label = f"share={sample.share * 100:.1f}% h={bh}px/{h}px {stamp} {raster}"
    else:
        label = f"no face {stamp} {raster}"

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
    try:
        ok = cv2.imwrite(str(dst), img)
    except cv2.error as e:  # неизвестное расширение и прочие отказы кодека
        raise RuntimeError(f"не записался кадр {dst}: {e}") from e
    if not ok:
        raise RuntimeError(f"не записался кадр: {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(description="доля лица по высоте кадра")
    ap.add_argument("video", help="путь к видео")
    ap.add_argument("--fps-sample", type=float, default=2.0, help="семплов в секунду (по умолч. 2)")
    ap.add_argument(
        "--at", type=float, action="append", default=None,
        help="секунда для отдельного замера, можно повторять (по умолч. 5.0)",
    )
    raster = ap.add_mutually_exclusive_group()
    # default=None, а не 0: иначе явное «--analyze-width 0» (нативный растр) было
    # бы не отличить от «ключ не задан» и молча превращалось бы в высоту по умолчанию.
    raster.add_argument(
        "--analyze-height", type=int, default=None,
        help=f"высота кадра для детекции, 0 — нативная (по умолч. {ANALYZE_HEIGHT})",
    )
    raster.add_argument(
        "--analyze-width", type=int, default=None,
        help="ширина кадра для детекции вместо высоты; между разными пропорциями несравнимо",
    )
    ap.add_argument("--annotate", help="PNG с боксом на кадре первого --at")
    ap.add_argument("--json", dest="json_path", help="файл для полного отчёта")
    args = ap.parse_args()

    # Расширение проверяем до замера: иначе cv2.error в конце убивает и --json тоже.
    if args.annotate and Path(args.annotate).suffix.lower() not in ANNOTATE_SUFFIXES:
        ap.error(
            f"--annotate: OpenCV не пишет '{Path(args.annotate).suffix}', "
            f"нужно одно из {', '.join(sorted(ANNOTATE_SUFFIXES))}"
        )

    at = args.at if args.at else [5.0]
    if args.analyze_width is not None:
        width, height = args.analyze_width, 0
    else:
        width = 0
        height = ANALYZE_HEIGHT if args.analyze_height is None else args.analyze_height
    path = Path(args.video)
    info = probe(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"OpenCV не открыл файл: {path}", file=sys.stderr)
        return 1
    detector = make_detector()
    if detector.name != VERDICT_DETECTOR:
        print(
            f"детектор {detector.name}, а не {VERDICT_DETECTOR}: коридор {TARGET_LOW:.0%}-"
            f"{TARGET_HIGH:.0%} на него не переносится, вердикт не считается",
            file=sys.stderr,
        )

    # Размеры берём из реального кадра, а не из ffprobe: OpenCV отдаёт картинку
    # уже с применённым поворотом, и именно её видит детектор.
    first, _ = _seek(cap, 0.0)
    if first is None:
        print(f"не читается ни один кадр: {path}", file=sys.stderr)
        cap.release()
        return 1
    frame_h, frame_w = first.shape[:2]
    analyzed = _rescale(first, height=height, width=width)[0].shape[:2]

    # Шаг семплирования меряется в кадрах, поэтому частота должна быть правдоподобной:
    # часть контейнеров отдаёт в CAP_PROP_FPS таймбазу (90000) вместо частоты, и шаг
    # вырос бы до десятков тысяч кадров — то есть ролик остался бы почти не измерен.
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not 1.0 <= src_fps <= 240.0:
        src_fps = info.fps if 1.0 <= info.fps <= 240.0 else 30.0
    if args.fps_sample > src_fps:
        print(
            f"--fps-sample {args.fps_sample} выше частоты кадров источника ({src_fps:.2f}); "
            f"семплируем каждый кадр, лишние семплы не появятся",
            file=sys.stderr,
        )
    cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)  # проход начинаем с начала, а не с кадра после пробного
    samples, degenerate = sample_sequence(
        cap, detector, fps_sample=args.fps_sample, src_fps=src_fps, height=height, width=width
    )
    effective_fps = src_fps / max(1, int(round(src_fps / max(0.05, args.fps_sample))))

    # Точечные --at идут отдельным seek: там время задано человеком, а не сеткой.
    seek_tol = max(0.5, 2.0 / src_fps)
    at_samples: list[Sample] = []
    at_frames: list[np.ndarray | None] = []
    for t in at:
        sample, frame = measure_at(cap, detector, t, height=height, width=width)
        at_samples.append(sample)
        at_frames.append(frame)
    cap.release()

    if degenerate:
        print(
            f"детектор {detector.name} вернул {degenerate} боксов без размеров (w/h): "
            "мерить долю по высоте нечем",
            file=sys.stderr,
        )

    st = stats(samples)
    med = st["median"]
    report = {
        "video": str(path.resolve()),
        "detector": detector.name,
        "frame": {"width": frame_w, "height": frame_h},
        # Растр анализа — часть идентичности замера: от него число зависит вдвое.
        "analyzed_frame": {"width": analyzed[1], "height": analyzed[0]},
        "analyze_height": height,
        "analyze_width": width,
        "duration": info.duration,
        "fps_sample": args.fps_sample,
        "fps_effective": effective_fps,
        "source_fps": src_fps,
        "n_degenerate_boxes": degenerate,
        "stats": st,
        "target": {
            "low": TARGET_LOW,
            "high": TARGET_HIGH,
            "metric": "median",
            "verdict": verdict(med, detector.name),
            "head_factor": HEAD_FACTOR,
            "head_share_estimate": None if med is None else med * HEAD_FACTOR,
            "head_share_note": (
                "грубая прикидка, не калибровано: на каноничном растре бокс уже включает "
                "лоб и часть волос, но обрезан выше подбородка"
            ),
        },
        "at": [asdict(s) for s in at_samples],
        "samples": [asdict(s) for s in samples],
        "note": (
            "share = высота бокса детектора / высота кадра АНАЛИЗА "
            f"({analyzed[1]}x{analyzed[0]}). Число зависит от растра анализа не слабее, "
            "чем от содержимого, поэтому сравнимы только прогоны с одинаковым "
            "analyzed_frame и одинаковым detector; при одинаковом растре остаётся "
            "зависимость от резкости источника (~5 п.п. между 720p и его 480p-копией)"
        ),
    }

    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Аннотация — после JSON: отказ кодека не должен уносить с собой весь замер.
    annotated: str | None = None
    if args.annotate:
        if at_frames[0] is None:
            print(f"кадр на {at[0]:.2f} c не прочитан, --annotate пропущен", file=sys.stderr)
        else:
            annotate(at_frames[0], at_samples[0], Path(args.annotate), analyzed)
            annotated = args.annotate

    def pct(v: float | None) -> str:
        return "  --  " if v is None else f"{v * 100:5.1f}%"

    print(
        f"{path.name}: {frame_w}x{frame_h}, {info.duration:.1f} c, детектор={detector.name} "
        f"на {analyzed[1]}x{analyzed[0]}, семплов={st['n_samples']} @ {effective_fps:.2f} fps"
    )
    print(
        f"  share по высоте: медиана {pct(med)}  p10 {pct(st['p10'])}  p90 {pct(st['p90'])}  "
        f"min {pct(st['min'])}  max {pct(st['max'])}  среднее {pct(st['mean'])}"
    )
    print(
        f"  лицо найдено в {st['coverage'] * 100:.0f}% ПРОЧИТАННЫХ кадров "
        f"({st['n_faces']}/{st['n_read']})"
    )
    if st["n_unread"]:
        print(f"  кадров не прочитано: {st['n_unread']} из {st['n_samples']}")
    if st["covered_to"] is not None:
        covered = st["covered_to"] - st["covered_from"]
        tail = "" if covered >= info.duration - 1.0 else "  ← измерена только часть ролика"
        print(
            f"  покрыто {st['covered_from']:.1f}-{st['covered_to']:.1f} c "
            f"из {info.duration:.1f} c{tail}"
        )
    if degenerate:
        print(f"  боксов без размеров: {degenerate} (детектор не заполняет w/h)")
    for s in at_samples:
        if s.t_actual is None:
            print(f"  на {s.t:.2f} c: кадр не прочитан (за концом файла?)")
            continue
        off = "" if abs(s.t_actual - s.t) <= seek_tol else "  ← seek промахнулся"
        print(f"  на {s.t:.2f} c (кадр {s.t_actual:.2f} c): {pct(s.share)}{off}")
    print(
        f"  коридор {TARGET_LOW * 100:.0f}-{TARGET_HIGH * 100:.0f}% по медиане на "
        f"{analyzed[1]}x{analyzed[0]}: {report['target']['verdict']}"
    )
    if annotated:
        print(f"  кадр с боксом: {annotated}")
    if args.json_path:
        print(f"  отчёт: {args.json_path}")
    # Ненулевой код, когда числу верить нельзя: обёртка должна отличать «померили»
    # от «данных нет». Второе — это и пустая выборка, и «прочли огрызок ролика»
    # (битый файл без индекса, видеопоток короче контейнера).
    if med is None:
        return 2
    covered = st["covered_to"] - st["covered_from"]
    if info.duration >= 1.0 and covered < info.duration / 2:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
