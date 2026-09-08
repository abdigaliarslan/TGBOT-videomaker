"""M1: извлечение звуковой дорожки (моно 16 кГц); M6: loudnorm и детекция тишины."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.core.proc import capture, ffmpeg

log = logging.getLogger(__name__)

ASR_SAMPLE_RATE = 16_000

# Цель по умолчанию для нарезки (M6). Звуковой пост (Ф2) просит свою из пресета:
# −16 LUFS — то, что ждут соцсети, −14 осталось историческим значением этапа cut.
LOUDNORM_TARGET_LUFS = -14.0
LOUDNORM_TP = -1.5  # потолок истинного пика, дБTP
LOUDNORM_LRA = 11.0


def loudnorm_base(target_lufs: float = LOUDNORM_TARGET_LUFS) -> str:
    return f"loudnorm=I={target_lufs:g}:TP={LOUDNORM_TP:g}:LRA={LOUDNORM_LRA:g}"


LOUDNORM_BASE = loudnorm_base()


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


def measure_loudness(
    src: str | Path, *, target_lufs: float = LOUDNORM_TARGET_LUFS
) -> dict[str, float] | None:
    """Первый проход loudnorm: замер. None, если замерить не удалось.

    Цель нужна уже здесь: loudnorm печатает `target_offset` относительно неё, и
    подставлять замер одной цели в фильтр с другой — значит промахнуться мимо
    обеих.
    """
    code, out, err = capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(src),
            # -vn обязателен: без него ffmpeg по умолчанию тащит в null-выход ещё и
            # видеопоток, то есть декодирует весь ролик ради замера звука. На
            # 20-секундном клипе это 1.58 с против 0.94 с, а замеров за джоб два.
            "-vn",
            "-af",
            f"{loudnorm_base(target_lufs)}:print_format=json",
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


def loudnorm_filter(
    measured: dict[str, float] | None = None, *, target_lufs: float = LOUDNORM_TARGET_LUFS
) -> str:
    """Строка фильтра. С замером — двухпроходный (точный), без — однопроходный."""
    base = loudnorm_base(target_lufs)
    need = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    if not measured or not all(k in measured for k in need):
        return base
    # -70 в input_i означает «тишина»: линейный режим на такой дорожке ломает звук.
    if measured["input_i"] <= -70.0:
        return base
    return (
        f"{base}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )


def measure_ebur128(src: str | Path) -> dict[str, float] | None:
    """Итоговая громкость дорожки по EBU R128: I (LUFS), LRA (LU), пик (dBFS).

    Отдельно от `measure_loudness`: тот считает вход для нормализации, а этот
    проверяет РЕЗУЛЬТАТ. Проверять нормализатор его же замером бессмысленно —
    нужен независимый счётчик, и это ровно та команда, которой DoD Ф2 меряется
    руками (`ffmpeg -i x -af ebur128 -f null -`).
    """
    code, out, err = capture(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(src),
            "-vn",  # мерим звук, видео декодировать незачем
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ]
    )
    if code != 0:
        log.warning("ebur128-замер не удался (код %s)", code)
        return None
    text = out + err
    # В логе ebur128 печатает и покадровые строки с тем же «I:», поэтому цепляемся
    # за блок Summary в конце, а не за первое совпадение.
    tail = text[text.rfind("Summary:") :] if "Summary:" in text else text
    values: dict[str, float] = {}
    for key, pattern in (
        ("i", r"\bI:\s*(-?[\d.]+)\s*LUFS"),
        ("lra", r"\bLRA:\s*(-?[\d.]+)\s*LU"),
        ("peak", r"Peak:\s*(-?[\d.]+)\s*dBFS"),
    ):
        m = re.search(pattern, tail)
        if m:
            try:
                values[key] = float(m.group(1))
            except ValueError:
                continue
    return values or None


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
            "-vn",  # мерим звук, видео декодировать незачем
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
