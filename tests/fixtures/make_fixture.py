"""Генератор тестового видео с ИЗВЕСТНЫМИ таймингами речи.

Без него DoD M1 («тайминги совпадают с речью в пределах 100 мс») непроверяем:
нужен эталон, а не «на слух похоже». Речь синтезируется ffmpeg-фильтром flite,
поэтому фикстура воспроизводима на любой машине без внешних файлов.

    python tests/fixtures/make_fixture.py

Кладёт рядом speech.mp4 и speech.truth.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent
VIDEO = FIXTURES / "speech.mp4"
TRUTH = FIXTURES / "speech.truth.json"

# (текст фразы, пауза ПОСЛЕ неё в секундах)
# Паузы разной длины: короткие остаются, длинные должен вырезать M6.
PHRASES: list[tuple[str, float]] = [
    ("this video shows how the editor works", 0.35),
    ("it cuts the long pauses automatically", 1.60),
    ("captions appear word by word", 0.40),
    ("and the important word gets bigger", 1.20),
    ("that is the whole idea", 0.60),
]
LEAD_IN = 0.5  # тишина в начале
FPS = 30
SIZE = (1920, 1080)
SAMPLE_RATE = 16_000


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-1500:]}")
    return p.stdout


def _duration(path: Path) -> float:
    return float(
        _run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
        ).strip()
    )


def _speech_span(path: Path, duration: float) -> tuple[float, float]:
    """Реальные границы речи внутри wav, а не границы файла.

    flite добавляет тишину в начале и в конце. Если считать эталоном границы
    файла, любой ASR будет «ошибаться» на эту тишину — и измерение станет
    бессмысленным. Границы речи берём по энергии сигнала.
    """
    import re
    import subprocess

    p = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-i", str(path),
            "-af", "silencedetect=noise=-45dB:d=0.03", "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    text = p.stdout + p.stderr
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", text)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", text)]

    speech_start = 0.0
    if starts and starts[0] <= 0.02 and ends:
        speech_start = ends[0]

    speech_end = duration
    if starts and starts[-1] > speech_start:
        # хвостовая тишина — та, что доходит до конца файла
        trailing_closes_at_eof = not ends or len(starts) > len(ends) or abs(ends[-1] - duration) < 0.05
        if trailing_closes_at_eof:
            speech_end = starts[-1]
    return round(speech_start, 3), round(speech_end, 3)


def build(out: Path = VIDEO, truth_path: Path = TRUTH) -> tuple[Path, dict]:
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / "_work"
    work.mkdir(exist_ok=True)

    parts: list[Path] = []
    truth: list[dict] = []
    cursor = LEAD_IN

    lead = work / "lead.wav"
    _run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
            "-t", f"{LEAD_IN}", "-c:a", "pcm_s16le", str(lead),
        ]
    )
    parts.append(lead)

    for i, (text, gap) in enumerate(PHRASES):
        speech = work / f"p{i}.wav"
        _run(
            [
                "ffmpeg", "-hide_banner", "-nostdin", "-y",
                "-f", "lavfi", "-i", f"flite=text='{text}':voice=slt",
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(speech),
            ]
        )
        dur = _duration(speech)
        sp_start, sp_end = _speech_span(speech, dur)
        truth.append(
            {
                "text": text,
                "words": text.split(),
                # границы РЕЧИ — эталон для DoD M1
                "start": round(cursor + sp_start, 3),
                "end": round(cursor + sp_end, 3),
                # границы файла — нужны только для отладки фикстуры
                "file_start": round(cursor, 3),
                "file_end": round(cursor + dur, 3),
                "gap_after": gap,
            }
        )
        parts.append(speech)
        cursor += dur

        if gap > 0:
            sil = work / f"s{i}.wav"
            _run(
                [
                    "ffmpeg", "-hide_banner", "-nostdin", "-y",
                    "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
                    "-t", f"{gap}", "-c:a", "pcm_s16le", str(sil),
                ]
            )
            parts.append(sil)
            cursor += gap

    total = round(cursor, 3)

    listing = work / "concat.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    audio = work / "audio.wav"
    _run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-y",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c:a", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", str(audio),
        ]
    )

    # Горизонтальная картинка с движущимся объектом: проверяет и автокроп 9:16
    # (есть за чем следить), и то, что рендер не растягивает кадр.
    w, h = SIZE
    fc = (
        "[0:v][1:v]overlay=x='(W-w)/2+380*sin(2*PI*t/6)':y='(H-h)/2'[bg];"
        "[bg]drawtext=text='REELBOT FIXTURE':fontcolor=white:fontsize=64"
        ":x=(w-text_w)/2:y=h-160[v]"
    )
    _run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"color=c=0x101418:s={w}x{h}:r={FPS}",
            "-f", "lavfi", "-i", f"color=c=0xE8734A:s=360x360:r={FPS}",
            "-i", str(audio),
            "-filter_complex", fc,
            "-map", "[v]", "-map", "2:a",
            "-t", f"{total}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-r", str(FPS),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            str(out),
        ]
    )

    meta = {
        "video": out.name,
        "duration": total,
        "fps": FPS,
        "width": w,
        "height": h,
        "language": "en",
        "lead_in": LEAD_IN,
        "phrases": truth,
    }
    truth_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    for p in work.iterdir():
        p.unlink()
    work.rmdir()
    return out, meta


def ensure() -> tuple[Path, dict]:
    """Собрать фикстуру, если её ещё нет. Используется тестами."""
    if VIDEO.exists() and TRUTH.exists():
        return VIDEO, json.loads(TRUTH.read_text(encoding="utf-8"))
    return build()


if __name__ == "__main__":
    path, meta = build()
    print(f"{path}  {meta['duration']:.2f}s", file=sys.stderr)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
