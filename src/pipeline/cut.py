"""M6: применение EDL.cuts.

Склейка делается фильтром `concat` с перекодированием, а не `-c copy`.
Без перекодирования джамп-каты поедут по ключевым кадрам: ffmpeg не умеет резать
между кейфреймами без пересжатия, и стык уползёт на десятые доли секунды —
ровно туда, где стоит первое слово следующего блока.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.config import Settings, settings as default_settings
from src.core.edl import EDL, Cut
from src.core.proc import ffmpeg
from src.pipeline.audio import loudnorm_filter, measure_loudness
from src.pipeline.probe import MediaInfo, probe

log = logging.getLogger(__name__)


def segments_of(edl: EDL) -> list[Cut]:
    return sorted(edl.kept_cuts, key=lambda c: c.src_in)


def build_filter(
    segments: list[Cut],
    *,
    has_audio: bool,
    fps: int,
    audio_filter: str | None = None,
) -> tuple[str, list[str]]:
    """filter_complex для нарезки и склейки + имена выходных лейблов."""
    parts: list[str] = []
    labels: list[str] = []
    for i, c in enumerate(segments):
        # fps=... перед trim выравнивает сетку кадров: иначе на стыках дублируются
        # или теряются кадры, и звук уезжает относительно картинки.
        parts.append(
            f"[0:v]trim=start={c.src_in:.4f}:end={c.src_out:.4f},"
            f"setpts=PTS-STARTPTS,fps={fps}[v{i}]"
        )
        labels.append(f"[v{i}]")
        if has_audio:
            parts.append(
                f"[0:a]atrim=start={c.src_in:.4f}:end={c.src_out:.4f},"
                f"asetpts=PTS-STARTPTS[a{i}]"
            )
            labels.append(f"[a{i}]")

    n = len(segments)
    if has_audio:
        order = "".join(f"[v{i}][a{i}]" for i in range(n))
        parts.append(f"{order}concat=n={n}:v=1:a=1[vc][ac]")
        vout, aout = "[vc]", "[ac]"
        if audio_filter:
            parts.append(f"[ac]{audio_filter}[aout]")
            aout = "[aout]"
        return ";".join(parts), [vout, aout]

    order = "".join(f"[v{i}]" for i in range(n))
    parts.append(f"{order}concat=n={n}:v=1:a=0[vc]")
    return ";".join(parts), ["[vc]"]


def apply_cuts(
    edl: EDL,
    src: str | Path,
    dst: str | Path,
    *,
    info: MediaInfo | None = None,
    normalize_audio: bool = True,
    crf: int = 18,
    preset_speed: str = "medium",
    cfg: Settings | None = None,
) -> Path:
    """Исходник + EDL → видео без пауз. Возвращает путь к результату."""
    cfg = cfg or default_settings
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = info or probe(src)
    segments = segments_of(edl)

    if not segments:
        raise ValueError("в EDL нет kept-кусков: нечего склеивать")

    # Резать нечего — весь ролик целиком. Копируем без второго пересжатия.
    if len(segments) == 1 and segments[0].src_in <= 0.01 and segments[0].src_out >= info.duration - 0.01:
        if not normalize_audio or not info.has_audio:
            log.info("нарезка не требуется, копирую поток")
            ffmpeg(["-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)], timeout=1800)
            return dst

    audio_filter = None
    if normalize_audio and info.has_audio:
        audio_filter = loudnorm_filter(measure_loudness(src))

    fps = int(round(info.fps)) or cfg.render_fps
    fc, outputs = build_filter(segments, has_audio=info.has_audio, fps=fps, audio_filter=audio_filter)

    args = ["-i", str(src), "-filter_complex", fc]
    for label in outputs:
        args += ["-map", label]
    args += [
        "-c:v", "libx264", "-preset", preset_speed, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-r", str(fps),
    ]
    if info.has_audio:
        args += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    args += ["-movflags", "+faststart", str(dst)]

    dropped = sum(c.duration for c in edl.cuts if not c.kept)
    log.info(
        "нарезка: %d кусков, вырезано %.1f с из %.1f с",
        len(segments), dropped, info.duration,
    )
    ffmpeg(args, timeout=7200)
    return dst


if __name__ == "__main__":  # python -m src.pipeline.cut in.mp4 edl.json out.mp4
    import json
    import sys

    from src.core.logging import setup_logging

    setup_logging(default_settings.log_level)
    source, edl_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    edl_obj = EDL(**json.loads(Path(edl_path).read_text(encoding="utf-8")))
    print(apply_cuts(edl_obj, source, out))
