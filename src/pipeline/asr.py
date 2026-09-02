"""M1: ASR с пословным выравниванием → list[Word].

Бэкенд выбирается один раз и абстрагирован, потому что стек Whisper ломается
на мажорных апдейтах (§5 M1 ⚠), а версии зафиксированы в pyproject.toml.

- `whisperx`     — форсированное выравнивание wav2vec2, самые точные границы слов,
                   но требует torch + (практически) CUDA.
- `faster_whisper` — CTranslate2, работает на CPU, тайминги слов берутся из
                   cross-attention самой модели. Это рабочий бэкенд по умолчанию
                   на машинах без GPU.

CLI: python -m src.pipeline.asr <file> > words.json
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.core.config import settings
from src.core.edl import Word
from src.pipeline.audio import ASR_SAMPLE_RATE, extract_wav

log = logging.getLogger(__name__)

Backend = Literal["whisperx", "faster_whisper"]

# Модели тяжёлые (сотни МБ) — грузим один раз на процесс.
_MODEL_CACHE: dict[tuple, Any] = {}


@dataclass(slots=True)
class ASRResult:
    words: list[Word]
    language: str
    backend: str
    model: str
    duration: float = 0.0
    dropped_segments: int = 0  # отброшенные галлюцинации
    raw_text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "backend": self.backend,
            "model": self.model,
            "duration": self.duration,
            "dropped_segments": self.dropped_segments,
            "text": self.raw_text,
            "words": [w.model_dump() for w in self.words],
        }

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


# ------------------------------------------------------------------ выбор бэкенда


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 — отсутствие CUDA не должно ронять пайплайн
        return False


def _has_whisperx() -> bool:
    from importlib.util import find_spec

    return find_spec("whisperx") is not None


def pick_backend(requested: str = "auto") -> Backend:
    if requested == "whisperx":
        if not _has_whisperx():
            raise RuntimeError("asr_backend=whisperx, но пакет whisperx не установлен")
        return "whisperx"
    if requested == "faster_whisper":
        return "faster_whisper"
    # auto: whisperx, если установлен — даже на CPU.
    #
    # Измерено на фикстуре tests/fixtures/speech.mp4 (15.5 с, 12 ядер, без GPU,
    # модель small; эталон — границы речи по энергии сигнала):
    #
    #   faster_whisper   0.23x реалтайма, среднее отклонение 230 мс, максимум 716 мс
    #   whisperx         0.30x реалтайма, среднее отклонение  46 мс, максимум 101 мс
    #
    # +30% времени за пятикратную точность границ слов. Для пословных субтитров
    # это решающе: на 230 мс подсветка уезжает на слово вперёд.
    if _has_whisperx():
        return "whisperx"
    return "faster_whisper"


def _device_and_compute() -> tuple[str, str]:
    device = settings.whisper_device
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    compute = settings.whisper_compute_type
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


# ------------------------------------------------------------------ нормализация


def _clean(text: str) -> str:
    return text.strip()


def _finalize(raw: list[tuple[str, float, float, float]], *, duration: float) -> list[Word]:
    """Из сырых (text, start, end, prob) в list[Word]: чистка, монотонность, индексы.

    Индекс равен позиции в списке — на этот инвариант опирается весь EDL (§3).
    """
    words: list[Word] = []
    prev_end = 0.0
    for text, start, end, prob in raw:
        t = _clean(text)
        if not t:
            continue
        s = max(0.0, float(start))
        e = float(end)
        if e <= s:
            e = s + 0.06
        # ASR иногда отдаёт перекрывающиеся границы — выпрямляем, иначе субтитры дрожат.
        if s < prev_end:
            s = prev_end
            if e <= s:
                e = s + 0.06
        if duration > 0:
            s = min(s, duration)
            e = min(e, duration)
            if e <= s:
                continue
        words.append(Word(idx=len(words), text=t, start=round(s, 3), end=round(e, 3), prob=round(float(prob), 3)))
        prev_end = e
    return words


# ------------------------------------------------------------------ бэкенды


def _transcribe_faster_whisper(wav: Path, *, language: str | None, model_name: str) -> ASRResult:
    from faster_whisper import WhisperModel

    device, compute = _device_and_compute()
    key = ("fw", model_name, device, compute, settings.whisper_cpu_threads)
    model = _MODEL_CACHE.get(key)
    if model is None:
        log.info("загружаю faster-whisper %s (%s/%s)", model_name, device, compute)
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute,
            cpu_threads=settings.whisper_cpu_threads,
        )
        _MODEL_CACHE[key] = model

    segments, info = model.transcribe(
        str(wav),
        language=language or None,
        word_timestamps=True,
        beam_size=settings.whisper_beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        # Whisper без этого зацикливается и дописывает несказанное.
        condition_on_previous_text=False,
    )

    raw: list[tuple[str, float, float, float]] = []
    texts: list[str] = []
    dropped = 0
    for seg in segments:
        seg_words = list(getattr(seg, "words", None) or [])
        if not seg_words:
            continue
        probs = [float(getattr(w, "probability", 1.0) or 0.0) for w in seg_words]
        avg = sum(probs) / len(probs)
        # Галлюцинации на музыке/шуме: низкая средняя уверенность слов.
        if avg < settings.whisper_min_confidence:
            dropped += 1
            log.debug("отброшен сегмент (conf=%.2f): %r", avg, seg.text)
            continue
        texts.append(seg.text)
        for w in seg_words:
            raw.append(
                (w.word, w.start, w.end, float(getattr(w, "probability", 1.0) or 1.0))
            )

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    return ASRResult(
        words=_finalize(raw, duration=duration),
        language=getattr(info, "language", language or "") or "",
        backend="faster_whisper",
        model=model_name,
        duration=duration,
        dropped_segments=dropped,
        raw_text=" ".join(t.strip() for t in texts).strip(),
        meta={"device": device, "compute_type": compute},
    )


def _transcribe_whisperx(wav: Path, *, language: str | None, model_name: str) -> ASRResult:
    import whisperx  # type: ignore[import-not-found]

    device, compute = _device_and_compute()
    key = ("wx", model_name, device, compute)
    model = _MODEL_CACHE.get(key)
    if model is None:
        log.info("загружаю whisperx %s (%s/%s)", model_name, device, compute)
        model = whisperx.load_model(model_name, device, compute_type=compute)
        _MODEL_CACHE[key] = model

    audio = whisperx.load_audio(str(wav))
    result = model.transcribe(audio, batch_size=8, language=language or None)
    lang = result.get("language", language or "")

    align_key = ("wx-align", lang, device)
    aligned_model = _MODEL_CACHE.get(align_key)
    if aligned_model is None:
        aligned_model = whisperx.load_align_model(language_code=lang, device=device)
        _MODEL_CACHE[align_key] = aligned_model
    align_model, metadata = aligned_model

    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, device, return_char_alignments=False
    )

    raw: list[tuple[str, float, float, float]] = []
    dropped = 0
    for seg in aligned.get("segments", []):
        seg_words = seg.get("words") or []
        usable = [w for w in seg_words if w.get("start") is not None and w.get("end") is not None]
        if not usable:
            continue
        probs = [float(w.get("score", 1.0) or 0.0) for w in usable]
        if sum(probs) / len(probs) < settings.whisper_min_confidence:
            dropped += 1
            continue
        for w in usable:
            raw.append((w["word"], w["start"], w["end"], float(w.get("score", 1.0) or 1.0)))

    return ASRResult(
        words=_finalize(raw, duration=len(audio) / ASR_SAMPLE_RATE),
        language=lang,
        backend="whisperx",
        model=model_name,
        duration=len(audio) / ASR_SAMPLE_RATE,
        dropped_segments=dropped,
        raw_text=" ".join(s.get("text", "").strip() for s in aligned.get("segments", [])).strip(),
        meta={"device": device, "compute_type": compute},
    )


# ------------------------------------------------------------------ публичный вход


def transcribe(
    src: str | Path,
    *,
    language: str | None = None,
    model: str | None = None,
    backend: str | None = None,
) -> ASRResult:
    """Видео или wav → пословный транскрипт.

    Не-wav вход сначала прогоняется через ffmpeg (моно 16 кГц).
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)

    lang = settings.whisper_language if language is None else language
    model_name = model or settings.whisper_model
    chosen = pick_backend(backend or settings.asr_backend)

    tmp: Path | None = None
    wav = src
    if src.suffix.lower() != ".wav":
        tmp = Path(tempfile.mkdtemp(prefix="reelbot-asr-")) / "audio.wav"
        wav = extract_wav(src, tmp)
    try:
        fn = _transcribe_whisperx if chosen == "whisperx" else _transcribe_faster_whisper
        result = fn(wav, language=lang or None, model_name=model_name)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
            tmp.parent.rmdir()

    log.info(
        "ASR: %d слов, язык=%s, бэкенд=%s/%s, отброшено сегментов=%d",
        len(result.words),
        result.language,
        result.backend,
        result.model,
        result.dropped_segments,
    )
    return result


def words_from_json(payload: dict | list) -> list[Word]:
    """Обратная загрузка words.json (для перерендера из сохранённого EDL, §0.4)."""
    items = payload["words"] if isinstance(payload, dict) else payload
    return [Word(**w) for w in items]


if __name__ == "__main__":  # python -m src.pipeline.asr file.mp4 > words.json
    import sys

    from src.core.logging import setup_logging

    setup_logging(settings.log_level)
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m src.pipeline.asr <file> > words.json")
    res = transcribe(sys.argv[1])
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
