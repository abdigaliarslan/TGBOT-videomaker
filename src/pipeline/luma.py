"""Яркость фона под субтитром: замер, которому уступает стиль (Ф9).

Панель фиксированной прозрачности не читается над светлым фоном: `opacity 0.34`
поверх снега даёт под белым текстом почти белое пятно (AUDIT §8.3). Прозрачность
нельзя выбрать один раз в пресете — её обязано выбирать измерение. Здесь считается
это измерение и кладётся в `CaptionBlock.bg_luma`; как из него получается
прозрачность — дело рендерера, тут нет ни одного решения о стиле.

Что именно меряется. Область — `geometry.caption_rect`, то есть прямоугольник
панели вместе с паддингами, ровно тот, который рендерер займёт под блок. Время —
ВЫХОДНОЙ таймлайн: видео на входе уже нарезано и кадрировано, под субтитром
окажется именно этот кадр, а не кадр исходника с тем же таймкодом. Прямоугольники
и времена берутся не пересчётом «примерно так же», а тем же `build_plan`, который
через минуту соберёт настоящий план: иначе замер и рендер разъедутся на первой же
правке геометрии.

Единица — ОТНОСИТЕЛЬНАЯ ЯРКОСТЬ ПО WCAG: каналы sRGB разгамливаются, потом
свёртка `0.2126 R + 0.7152 G + 0.0722 B`. Так требует потребитель: Captions.tsx
считает `contrastRatio` по WCAG, а формула контраста определена только на
линейном свете. Число едет в рендерер как есть, и все переводы туда-обратно
(смешать панель с фоном надо в sRGB-кодах, потому что Chromium композитит
`rgba()` там) рендерер делает сам — `toSrgb(bg_luma)` на входе в смесь и
`toLinear` перед сравнением. Пороги пресета (`luma_dark`/`luma_light`) читают то
же число, то есть калиброваться обязаны на линейной шкале, где средний серый —
0.216, а не 0.5; текущие 0.2/0.8 ставились, когда единица нигде не была названа,
и их стоит перепроверить на стороне рендерера.

Кадры берутся ffmpeg-ом, а не `cv2.VideoCapture`, и это не вкусовщина. OpenCV
игнорирует тег матрицы потока и раскладывает yuv420p всегда по BT.601, а
staged-видео помечено bt709, и рендерер (Chromium, как и ffmpeg) декодирует его
по тегу. На насыщенном цвете расхождение доходит до 0.1 по яркости — уровень, на
котором порог прозрачности панели переключается. Мерить надо тот растр, который
увидит рендерер, поэтому декодирует его тот же декодер.

Цена этапа — O(длины ролика), а не O(числа блоков): файл декодируется подряд до
последнего нужного кадра, `-frames:v` обрывает проход сразу после него.
Меньше секунды — это характеристика 18-секундного эталона, а не этапа: на входе
в разрешённые `max_video_minutes = 3` тот же проход стоит примерно на порядок
больше, и выключателя у него нет. Перемотки на каждый семпл дешевле всё равно не
выходят: на шестиминутном файле 36 семплов стоят 28.9 с сплошным проходом против
31.7 с отдельным `-ss` на каждый — декодер каждый раз отматывается к опорному
кадру и догоняет заново, а сверху ложится запуск процесса.

Замер описывает ВИДЕО, а не композит: b-roll с анимацией `fullscreen_to_inset`
на старте лежит поверх всего кадра (см. `geometry.resting_anchor` — с субтитром
разводится только конечный угол), и под таким блоком реальный фон — картинка, а
не кадр. Прямоугольники b-roll и фаза анимации есть только у рендерера, ему это
и решать; здесь это известная дыра, а не незамеченная.

Деградация везде тихая: не открылось видео, не посчитался прямоугольник, не
пришёл кадр — `bg_luma` остаётся `None`, и ролик едет как раньше. Именно `None`,
а не число из прошлого прогона: поле обнуляется до замера, потому что EDL
приходит извне (`process_from_edl`) и может уже нести значения, снятые с другого
кропа и другого пресета.
"""

from __future__ import annotations

import logging
import math
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.config import Settings, settings as default_settings
from src.core.edl import EDL
from src.core.geometry import Rect
from src.core.preset import Preset
from src.core.proc import ProcError
from src.core.render_plan import build_plan
from src.pipeline.probe import probe

log = logging.getLogger(__name__)

# Сколько кадров на блок. Три — компромисс из QUALITY_PLAN Ф9: одного мало (блок
# живёт секунду, за секунду фон успевает поменяться), больше четырёх не меняет
# среднее, но каждый кадр стоит распаковки.
SAMPLES_PER_BLOCK = 3

# Коэффициенты яркости Rec.709 в порядке каналов rgb24, который отдаёт ffmpeg.
# Перепутанный порядок здесь не падает, а тихо врёт: синий весит 0.07, красный
# 0.21 — на телесных тонах разница в разы больше допуска любого порога.
LUMA_RGB = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# sRGB-код → линейный свет. Кодов всего 256, поэтому таблица вместо возведения в
# степень на каждом пикселе: разгамливание нелинейно, среднее по кодам ему не
# равно, а честно линеаризовать миллион пикселей на кадр — секунды.
_CODES = np.arange(256, dtype=np.float64) / 255.0
TO_LINEAR = np.where(_CODES <= 0.04045, _CODES / 12.92, ((_CODES + 0.055) / 1.055) ** 2.4)


@dataclass(frozen=True, slots=True)
class LumaReport:
    """Что удалось измерить. Нужен для логов и тестов, в EDL не едет."""

    blocks: int  # всего блоков субтитров
    measured: int  # для скольких заполнен bg_luma
    frames: int  # сколько кадров реально распаковано
    seconds: float
    values: tuple[float | None, ...]

    @property
    def mean(self) -> float | None:
        got = [v for v in self.values if v is not None]
        return round(sum(got) / len(got), 4) if got else None


def frame_luma(frame: np.ndarray, rect: Rect) -> float | None:
    """Относительная яркость по WCAG в прямоугольнике кадра, 0..1.

    `frame` — кадр rgb24 (или серый), `None` на выходе — мерить нечего.
    Прямоугольник обрезается по кадру: блок у нижней кромки после округления
    вылезает на пиксель-другой, и это не повод терять замер целиком.
    """
    height, width = frame.shape[:2]
    x0 = max(0, int(math.floor(rect.x)))
    y0 = max(0, int(math.floor(rect.y)))
    x1 = min(width, int(math.ceil(rect.right)))
    y1 = min(height, int(math.ceil(rect.bottom)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None

    patch = TO_LINEAR[frame[y0:y1, x0:x1]]
    if patch.ndim == 2:  # серый кадр: все каналы равны, свёртка даёт то же самое
        value = float(patch.mean())
    else:
        # Среднее по каналам, потом свёртка: яркость линейна по каналам ПОСЛЕ
        # разгамливания, поэтому средняя яркость равна яркости средних.
        value = float(patch.reshape(-1, patch.shape[-1])[:, :3].mean(axis=0) @ LUMA_RGB)
    return min(1.0, max(0.0, value))


def _sample_times(start: float, end: float, count: int) -> list[float]:
    """Моменты замера внутри блока: центры равных долей, без краёв.

    Края брать нельзя: на границе блока обычно стоит рез, и кадр там принадлежит
    уже соседнему куску.
    """
    if count < 1:
        count = 1
    span = end - start
    if span <= 0:
        return [start]
    return [start + span * (i + 0.5) / count for i in range(count)]


def _decode(path: Path, indices: list[int], width: int, height: int, cfg: Settings):
    """Кадры с номерами `indices` (по возрастанию) как массивы rgb24.

    Один проход одним процессом: `select` пропускает нужные номера, `-frames:v`
    обрывает декодирование сразу после последнего. Кадров может прийти меньше,
    чем просили (файл кончился раньше плана) — тогда недостающие просто не
    выдаются, и их блоки остаются без замера.
    """
    expr = "+".join(rf"eq(n\,{i})" for i in indices)
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error"]
    if cfg.ffmpeg_threads > 0:
        cmd += ["-threads", str(cfg.ffmpeg_threads)]
    cmd += [
        "-i", str(path),
        "-vf", f"select='{expr}'",
        "-vsync", "0",
        "-frames:v", str(len(indices)),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    size = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(size)
            if buf is None or len(buf) < size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        if proc.poll() is None:
            proc.kill()
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                pipe.close()
        proc.wait()
        if err.strip():
            log.debug("ffmpeg при замере яркости: %s", err.strip().splitlines()[-1])


def measure_bg_luma(
    edl: EDL,
    video: str | Path,
    *,
    preset: Preset | None = None,
    apply_cuts: bool = True,
    samples: int = SAMPLES_PER_BLOCK,
    cfg: Settings | None = None,
) -> LumaReport:
    """Заполнить `CaptionBlock.bg_luma` по уже нарезанному и кадрированному видео.

    Меняет `edl` на месте: сначала сбрасывает поле у всех блоков, потом
    заполняет измеренные. Размер кадра и fps берутся из самого файла: замер
    обязан идти по тому растру, который увидит рендерер, а не по тому, который
    просили у кропа.
    """
    cfg = cfg or default_settings
    t0 = time.perf_counter()
    path = Path(video)
    blocks = len(edl.captions)

    def report(measured: int, frames: int) -> LumaReport:
        return LumaReport(
            blocks=blocks,
            measured=measured,
            frames=frames,
            seconds=round(time.perf_counter() - t0, 3),
            values=tuple(b.bg_luma for b in edl.captions),
        )

    if not edl.captions:
        return report(0, 0)

    # Старое значение — не замер: EDL мог прийти с прошлого прогона, где были
    # другой кроп и другой пресет. Чего не измерили, того нет.
    for block in edl.captions:
        block.bg_luma = None

    try:
        info = probe(path)
    except (OSError, ValueError, ProcError) as e:
        log.warning("яркость фона не измерена: не читается видео %s (%s)", path, e)
        return report(0, 0)

    width, height, fps = info.width, info.height, info.fps
    if width <= 0 or height <= 0:
        log.warning("яркость фона не измерена: у %s не читается размер кадра", path)
        return report(0, 0)
    if not (fps > 0):
        fps = float(cfg.render_fps)

    plan = build_plan(
        edl,
        str(path),
        fps=int(round(fps)) or cfg.render_fps,
        width=width,
        height=height,
        preset=preset,
        apply_cuts=apply_cuts,
        cfg=cfg,
    )

    # Слово знает свой блок: `build_plan` выкидывает блоки, у которых все
    # слова вырезаны, поэтому по номеру в списке блоки плана и EDL не
    # совпадают, а по индексу слова совпадают всегда.
    owner: dict[int, int] = {}
    for bi, block in enumerate(edl.captions):
        for cw in block.words:
            owner[cw.word_idx] = bi

    jobs: dict[int, list[tuple[int, Rect]]] = defaultdict(list)
    for pb in plan.blocks:
        if pb.rect is None:
            continue
        owners = [owner[w.word_idx] for w in pb.words if w.word_idx in owner]
        if not owners:
            continue
        bi = owners[0]
        for t in _sample_times(pb.start, pb.end, samples):
            # Семпл за концом файла не клэмпится к последнему кадру: чужой
            # кадр — это не «измерили хуже», это неверное число.
            jobs[max(0, int(round(t * fps)))].append((bi, pb.rect))

    if not jobs:
        log.info("яркость фона: мерить нечего — ни у одного блока нет прямоугольника")
        return report(0, 0)

    acc: dict[int, list[float]] = defaultdict(list)
    read = 0
    indices = sorted(jobs)
    for idx, frame in zip(indices, _decode(path, indices, width, height, cfg)):
        read += 1
        for bi, rect in jobs[idx]:
            value = frame_luma(frame, rect)
            if value is not None:
                acc[bi].append(value)

    for bi, values in acc.items():
        edl.captions[bi].bg_luma = round(sum(values) / len(values), 4)

    out = report(len(acc), read)
    log.info(
        "яркость фона: %d/%d блоков, %d кадров, среднее %s, %.2f с",
        out.measured, out.blocks, out.frames, out.mean, out.seconds,
    )
    return out
