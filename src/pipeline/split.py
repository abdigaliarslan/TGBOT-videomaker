"""Режим «Два видео»: кадр, поделённый сверху и снизу.

Сверху — второе видео пользователя, без звука и без склеек: оно иллюстрация, а
не речь, поэтому идёт непрерывно в выходном времени. Снизу — говорящий, уже
нарезанный по паузам и кадрированный по лицу в половину кадра. Когда верхнему
видео больше нечего показывать, нижнее мгновенно раскрывается на весь экран, и
дальше ролик обычный.

Кадр собирается ЗДЕСЬ, до рендера, в подготовленное видео 1080×1920. Рендерер
решений не принимает (§0): он получает одно видео и рисует поверх него план, как
у любого другого ролика. Момент раскрытия лежит в листе (`EDL.split_until`),
чтобы перерендер собрал ту же склейку.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

from src.core.config import Settings, settings as default_settings
from src.core.proc import ffmpeg
from src.pipeline.crop import autocrop
from src.pipeline.probe import probe

log = logging.getLogger(__name__)

# Короче этого разделённый кадр не ставится вовсе: половины, мелькнувшие на
# полсекунды в начале, читаются как сбой, а не как приём.
MIN_SPLIT_SECONDS = 1.0


def choose_split_until(
    top_duration: float, out_duration: float, windows: Sequence[tuple[float, float]]
) -> float | None:
    """До какой секунды ролика кадр поделён. `None` — разделения не будет.

    Верхнее видео не короче ролика — поделён весь ролик, хвост верхнего
    обрезается. Иначе склейка встаёт на границе субтитров (начало или конец
    блока), не позже конца верхнего: позже значило бы стоп-кадр наверху, а
    посреди блока — субтитр, который прыгает с шва вниз на полуслове.
    """
    if top_duration >= out_duration:
        return out_duration
    bounds = {0.0} | {a for a, _ in windows} | {b for _, b in windows}
    best = max((b for b in bounds if b <= top_duration), default=0.0)
    return best if best >= MIN_SPLIT_SECONDS else None


def top_window(
    src: Path, dst: Path, *, width: int, height: int, fps: float, duration: float
) -> Path:
    """Верхнее видео в окно половины: заполнить и обрезать по центру, без звука.

    Поворот с телефона ffmpeg применяет сам при чтении. Последний кадр
    придерживается на полсекунды: граница склейки считается по длительности из
    контейнера, а кадров в нём бывает на один меньше, и `vstack` тогда кончился
    бы раньше нижней половины.
    """
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,fps={fps:g},"
        "tpad=stop_mode=clone:stop_duration=0.5"
    )
    ffmpeg(
        ["-i", str(src), "-an", "-vf", vf, "-t", f"{duration:.3f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(dst)],
        timeout=3600,
    )
    return dst


def stage_split(
    bottom_cut: Path,
    top_src: Path,
    dst: Path,
    *,
    until: float,
    out_duration: float,
    work_dir: Path,
    cfg: Settings | None = None,
    preset: Any = None,
    framing: Any = None,
    track_out: Path | None = None,
    crop: Callable[..., Path] = autocrop,
) -> Path:
    """Подготовленное видео 1080×1920 из двух: половины до `until`, дальше нижнее на весь экран.

    Нижнее кадрируется дважды — в половину и, если раскрытие наступает раньше
    конца ролика, в полный кадр. Трек лица берётся у полного прохода: только
    там голова лежит в координатах кадра, поверх которого рисуются субтитры;
    у половины его нет, и привязку к голове план на это время снимает сам.
    Звук — дорожка нарезанного нижнего, ровно как у обычного ролика.
    """
    cfg = cfg or default_settings
    work_dir = Path(work_dir)
    width, height = cfg.render_width, cfg.render_height
    half_h = height // 2
    fps = round(probe(bottom_cut).fps) or cfg.render_fps

    half = crop(
        bottom_cut, work_dir / "split-bottom.mp4", out_width=width, out_height=half_h,
        preset=preset, framing=framing, cfg=cfg, work_dir=work_dir, track_out=None,
    )
    top = top_window(
        top_src, work_dir / "split-top.mp4", width=width, height=half_h, fps=fps, duration=until,
    )
    full = None
    if until < out_duration:
        full = crop(
            bottom_cut, work_dir / "split-full.mp4", out_width=width, out_height=height,
            preset=preset, framing=framing, cfg=cfg, work_dir=work_dir, track_out=track_out,
        )

    inputs = ["-i", str(top), "-i", str(half)]
    fc = (
        f"[0:v]trim=0:{until:.3f},setpts=PTS-STARTPTS,setsar=1[t];"
        f"[1:v]trim=0:{until:.3f},setpts=PTS-STARTPTS,setsar=1[h];"
        "[t][h]vstack=inputs=2[s]"
    )
    audio_input = 2
    if full is not None:
        inputs += ["-i", str(full)]
        fc += (
            f";[2:v]trim=start={until:.3f}:end={out_duration:.3f},setpts=PTS-STARTPTS,setsar=1[f];"
            "[s][f]concat=n=2:v=1:a=0[v]"
        )
        audio_input = 3
    else:
        fc += ";[s]null[v]"
    inputs += ["-i", str(bottom_cut)]
    ffmpeg(
        [*inputs, "-filter_complex", fc, "-map", "[v]", "-map", f"{audio_input}:a?",
         "-t", f"{out_duration:.3f}", "-r", str(fps),
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
         "-movflags", "+faststart", str(dst)],
        timeout=7200,
    )
    log.info(
        "два видео: поделено до %.2f с из %.2f с%s",
        until, out_duration, ", дальше нижнее на весь экран" if full is not None else "",
    )
    return Path(dst)
