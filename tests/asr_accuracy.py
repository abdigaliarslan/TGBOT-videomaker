"""Измеритель точности ASR относительно эталона фикстуры (DoD M1).

Не тест, а инструмент: печатает отклонение границ фраз, чтобы выбор параметров
ASR был измерением, а не вкусовщиной (§7).

    python tests/asr_accuracy.py [--no-vad] [--model small]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fixtures.make_fixture import ensure  # noqa: E402


def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--no-vad", action="store_true")
    args = ap.parse_args()

    import os

    os.environ["WHISPER_LANGUAGE"] = "en"
    os.environ["WHISPER_MODEL"] = args.model

    from src.core.config import get_settings

    get_settings.cache_clear()
    from src.core.logging import setup_logging
    from src.pipeline import asr as asr_mod

    setup_logging("WARNING")

    if args.no_vad:
        original = asr_mod._transcribe_faster_whisper

        def patched(wav, *, language, model_name):
            from faster_whisper import WhisperModel

            key = ("fw-novad", model_name)
            model = asr_mod._MODEL_CACHE.get(key)
            if model is None:
                model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=8)
                asr_mod._MODEL_CACHE[key] = model
            segments, info = model.transcribe(
                str(wav),
                language=language,
                word_timestamps=True,
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            raw = [
                (w.word, w.start, w.end, float(getattr(w, "probability", 1.0) or 1.0))
                for seg in segments
                for w in (getattr(seg, "words", None) or [])
            ]
            return asr_mod.ASRResult(
                words=asr_mod._finalize(raw, duration=float(info.duration)),
                language=info.language,
                backend="faster_whisper(no-vad)",
                model=model_name,
                duration=float(info.duration),
            )

        asr_mod._transcribe_faster_whisper = patched  # type: ignore[assignment]
        _ = original

    video, truth = ensure()
    res = asr_mod.transcribe(video, language="en", model=args.model)

    words = res.words
    print(f"бэкенд={res.backend} модель={res.model} слов={len(words)}")

    cursor = 0
    deltas: list[tuple[str, float, float]] = []
    for ph in truth["phrases"]:
        expected = [norm(w) for w in ph["words"]]
        # ищем первое слово фразы, начиная с текущей позиции
        first = next(
            (i for i in range(cursor, len(words)) if norm(words[i].text) == expected[0]), None
        )
        if first is None:
            print(f"  !! не найдено начало фразы: {ph['text']!r}")
            continue
        last = first + len(expected) - 1
        if last >= len(words):
            print(f"  !! фраза обрезана: {ph['text']!r}")
            continue
        got_start, got_end = words[first].start, words[last].end
        d_start = got_start - ph["start"]
        d_end = got_end - ph["end"]
        deltas.append((ph["text"], d_start, d_end))
        print(
            f"  {ph['text'][:38]:38s} start {ph['start']:6.2f} → {got_start:6.2f} "
            f"({d_start:+.3f})   end {ph['end']:6.2f} → {got_end:6.2f} ({d_end:+.3f})"
        )
        cursor = last + 1

    if deltas:
        all_d = [abs(d) for _, ds, de in deltas for d in (ds, de)]
        print(f"\nсреднее |Δ| = {sum(all_d)/len(all_d)*1000:.0f} мс, максимум = {max(all_d)*1000:.0f} мс")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
