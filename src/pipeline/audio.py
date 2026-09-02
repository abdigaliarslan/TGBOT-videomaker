"""M1: извлечение звуковой дорожки (моно 16 кГц); M6: loudnorm и детекция тишины."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.core.proc import capture, ffmpeg

log = logging.getLogger(__name__)

ASR_SAMPLE_RATE = 16_000
LOUDNORM_BASE = "loudnorm=I=-14:TP=-1.5:LRA=11"


def extract_wav(src: str | Path, dst: str | Path, *, sample_rate: int = ASR_SAMPLE_RATE) -> Path:
    """Моно PCM-wav для ASR.

    Без нормализации: ASR к громкости нечувствителен, а loudnorm на этом шаге
    только добавил бы задержку фильтра и сдвинул тайминги.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(
        ["-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)]
    )
    return dst


def measure_loudness(src: str | Path) -> dict[str, float] | None:
    """Первый проход loudnorm: замер. None, если замерить не удалось."""
    code, out, err = capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(src),
            "-af",
            f"{LOUDNORM_BASE}:print_format=json",
            "-f",
            "null",
            "-",
        ]
    )
    if code != 0:
        log.warning("loudnorm-замер не удался (код %s)", code)
        return None
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", out + err, re.S)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    parsed: dict[str, float] = {}
    for k, v in raw.items():
        try:
            parsed[k] = float(v)
        except (TypeError, ValueError):
            continue
    return parsed or None


def loudnorm_filter(measured: dict[str, float] | None = None) -> str:
    """Строка фильтра. С замером — двухпроходный (точный), без — однопроходный."""
    need = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    if not measured or not all(k in measured for k in need):
        return LOUDNORM_BASE
    # -70 в input_i означает «тишина»: линейный режим на такой дорожке ломает звук.
    if measured["input_i"] <= -70.0:
        return LOUDNORM_BASE
    return (
        f"{LOUDNORM_BASE}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )


def detect_silence(
    src: str | Path, *, noise_db: float = -32.0, min_duration: float = 0.35
) -> list[tuple[float, float]]:
    """Интервалы тишины через silencedetect (M6: детекция на стадии восприятия).

    Дополняет паузы между словами из ASR: ловит тишину до первого слова,
    после последнего и внутри «слов», которые ASR склеил.
    """
    code, out, err = capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(src),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_duration}",
            "-f",
            "null",
            "-",
        ]
    )
    if code != 0:
        return []
    text = out + err
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", text)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", text)]
    spans: list[tuple[float, float]] = []
    for i, s in enumerate(starts):
        if i < len(ends) and ends[i] > s:
            spans.append((max(0.0, s), ends[i]))
        elif i == len(starts) - 1 and i >= len(ends):
            spans.append((max(0.0, s), float("inf")))  # тишина до конца файла
    return spans


if __name__ == "__main__":  # python -m src.pipeline.audio in.mp4 [out.wav]
    import sys

    logging.basicConfig(level=logging.INFO)
    print(extract_wav(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "audio.wav"))
