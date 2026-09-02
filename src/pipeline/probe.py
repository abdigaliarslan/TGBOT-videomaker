"""M1: ffprobe → длительность, fps, разрешение, наличие звука."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path

from src.core.proc import run


@dataclass(slots=True)
class MediaInfo:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    audio_channels: int
    audio_rate: int
    rotation: int  # градусы из side data; после поворота ширина/высота уже поменяны местами
    video_codec: str
    bitrate: int

    @property
    def is_vertical(self) -> bool:
        return self.height >= self.width

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        val = stream.get(key)
        if not val or val in ("0/0", "0"):
            continue
        try:
            f = float(Fraction(val))
        except (ZeroDivisionError, ValueError):
            continue
        if f > 0:
            return round(f, 3)
    return 30.0


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            return int(float(tags["rotate"])) % 360
        except (TypeError, ValueError):
            pass
    return 0


def probe(path: str | Path) -> MediaInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    data = json.loads(out)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError(f"в файле нет видеодорожки: {path}")

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    rot = _rotation(video)
    # ffmpeg сам применяет поворот при декодировании; логические размеры — с учётом него.
    if rot in (90, 270):
        width, height = height, width

    duration = 0.0
    for src in (fmt.get("duration"), video.get("duration")):
        try:
            duration = float(src)
            break
        except (TypeError, ValueError):
            continue
    if duration <= 0:
        # Контейнер без длительности (например, сырой поток) — считаем по пакетам.
        got = run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "packet=pts_time",
                "-of",
                "csv=p=0",
                str(path),
            ]
        ).strip().splitlines()
        duration = float(got[-1]) if got else 0.0

    return MediaInfo(
        path=str(path),
        duration=round(duration, 3),
        width=width,
        height=height,
        fps=_fps(video),
        has_audio=audio is not None,
        audio_channels=int((audio or {}).get("channels") or 0),
        audio_rate=int((audio or {}).get("sample_rate") or 0),
        rotation=rot,
        video_codec=video.get("codec_name", "?"),
        bitrate=int(fmt.get("bit_rate") or 0),
    )


if __name__ == "__main__":  # python -m src.pipeline.probe file.mp4
    import sys

    print(json.dumps(probe(sys.argv[1]).to_dict(), indent=2, ensure_ascii=False))
