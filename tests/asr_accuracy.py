"""Измеритель точности ASR: отклонение границ слов от речи (DoD M1).

Не тест, а инструмент: печатает отклонение границ, время прогона и пиковую
память, чтобы выбор бэкенда и модели был измерением, а не вкусовщиной (§7).

Два режима, потому что эталоны бывают двух видов:

1. Фикстура с разметкой (`tests/fixtures/speech.mp4`) — синтезированная речь,
   границы фраз известны с точностью до кадра. Меряется |Δ| начала первой и
   конца последней границы каждой фразы. Речь английская: flite других голосов
   не знает, а другого синтезатора в окружении нет.

2. Любой ролик без разметки (`--media`) — эталоном служит ЭНЕРГИЯ сигнала:
   момент, когда речь возобновляется после паузы, слышно и без расшифровки.
   Это единственный способ проверить рабочий язык (русский), и он же ловит
   болезнь, которую фикстура не видит: слово, растянутое на паузу.

    python tests/asr_accuracy.py                         # фикстура, бэкенд из настроек
    python tests/asr_accuracy.py --backend whisperx --model medium
    python tests/asr_accuracy.py --media samples/phase-0-source.mp4 --language ru
"""

from __future__ import annotations

import argparse
import os
import re
import resource
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fixtures.make_fixture import ensure  # noqa: E402

# Окно поиска слова под точку эталона. Шире — и метрика начинает притягивать
# соседнее слово, показывая «точность» там, где бэкенд промахнулся на слово.
MATCH_WINDOW = 1.0
# Пауза, по возобновлению речи после которой считается отклонение. Короче — уже
# не пауза, а смычка внутри фразы, там граница речи размазана.
SILENCE_MIN = 0.35
# Порог тишины для silencedetect. −35 дБ отделяет речь от комнатного шума на
# записи с телефона; на синтезированной фикстуре сгодился бы любой.
SILENCE_NOISE = "-35dB"
# «Тишина доходит до конца файла» — с точностью до кадра огибающей silencedetect
# (10 мс) плюс запас.
EDGE_EPS = 0.05


def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _stats(deltas: list[float]) -> tuple[float, float] | None:
    """Среднее |Δ| и максимум в миллисекундах. None — мерить нечего.

    Именно None, а не (0, 0): пустой список значит «точек эталона не осталось»,
    и напечатанный ноль читался бы как идеальная точность — ровно наоборот.
    """
    a = [abs(d) for d in deltas]
    return (sum(a) / len(a) * 1000.0, max(a) * 1000.0) if a else None


def _fmt(stats: tuple[float, float] | None, label: str, n: int, missed: int) -> str:
    if stats is None:
        return f"{label} ({n}): мерить нечего — точек эталона не осталось"
    mean, worst = stats
    tail = f", без слова: {missed}" if missed else ""
    return f"{label} ({n}): среднее |Δ| = {mean:.0f} мс, максимум = {worst:.0f} мс{tail}"


# ------------------------------------------------------------------ режим фикстуры


def measure_fixture(words, truth) -> list[float]:
    """|Δ| границ фраз. Печатает каждую фразу, возвращает отклонения в секундах."""
    cursor = 0
    deltas: list[float] = []
    for ph in truth["phrases"]:
        expected = [norm(w) for w in ph["words"]]
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
        # Совпадение ОДНОГО первого слова ничего не гарантирует: стоит ASR
        # склеить или разбить слово, и `last` укажет на чужое слово, а измеритель
        # напечатает отклонение, к границе фразы отношения не имеющее.
        got = [norm(w.text) for w in words[first : last + 1]]
        if got != expected:
            print(f"  !! слова фразы {ph['text']!r} не совпали с транскриптом: {' '.join(got)}")
            continue
        got_start, got_end = words[first].start, words[last].end
        d_start = got_start - ph["start"]
        d_end = got_end - ph["end"]
        deltas += [d_start, d_end]
        print(
            f"  {ph['text'][:38]:38s} start {ph['start']:6.2f} → {got_start:6.2f} "
            f"({d_start:+.3f})   end {ph['end']:6.2f} → {got_end:6.2f} ({d_end:+.3f})"
        )
        cursor = last + 1
    return deltas


# ------------------------------------------------------------------ режим по энергии


def audio_duration(media: Path) -> float:
    """Длительность звуковой дорожки. 0.0 — если ffprobe не смог."""
    p = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=duration", "-of", "csv=p=0", str(media),
        ],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(p.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0.0


def speech_edges(media: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Паузы по энергии: (по концам которых меряются начала, по началам — концы).

    Возвращаются именно ПАУЗЫ (start, end), а не отдельные моменты: без второй
    границы не отличить слово, начавшееся ДО паузы и растянутое на неё, от
    первого слова после неё — а это две разные ошибки бэкенда.
    """
    p = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-i", str(media),
            "-af", f"silencedetect=noise={SILENCE_NOISE}:d={SILENCE_MIN}", "-f", "null", "-",
        ],
        capture_output=True, text=True, check=False,
    )
    text = p.stdout + p.stderr
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", text)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", text)]
    if not ends:
        return [], []
    # Тишина в начале файла — не «уход в паузу», а его отсутствие; тишина в конце
    # — не «возобновление». Считать по ним отклонение значит мерить длину файла.
    #
    # Отделять их надо ПО ВРЕМЕНИ, а не по порядку в списке: silence_start своей
    # паузы всегда идёт раньше её silence_end, поэтому сравнение starts[0] с
    # ends[0] истинно на любом файле и выбрасывало бы первую же точку эталона —
    # на ролике, который начинается с речи, мерить становилось нечего.
    # Ведущая тишина — та, до которой не было речи: либо файл начинается с неё
    # (starts[0] == 0), либо до неё звучало меньше, чем длится минимальная пауза
    # (щелчок, стук микрофона — фразой это быть не может).
    duration = audio_duration(media)
    leading = bool(starts) and starts[0] < SILENCE_MIN
    pauses = list(zip(starts, ends, strict=False))
    body = pauses[1:] if leading else pauses
    onsets = [
        (s, e)
        for s, e in body
        # хвостовой флаш silencedetect на конце потока — не возобновление речи
        if not (duration and duration - e <= EDGE_EPS)
    ]
    return onsets, body


def _match_in_order(words, pauses: list[tuple[float, float]], label: str, *, onset: bool):
    """Сопоставить паузы со словами ПО ПОРЯДКУ транскрипта.

    Ближайшее по времени слово брать нельзя: оно не обязано быть тем самым.
    Метрика тогда меряет не выравнивание, а ПЛОТНОСТЬ границ — фиктивный
    «бэкенд», печатающий слово каждые 100 мс и не слышавший звука, набирает на
    «ближайшем» 37 мс по началам и 23 мс по концам, то есть «точнее» настоящих
    (на сопоставлении по порядку он же даёт 946 мс, как и должен).

    Поэтому слово выбирается по порядку и ровно один раз:
    - `onset` — первое слово, НАЧАВШЕЕСЯ не раньше паузы; отклонение считается
      от её конца (момента, когда речь возобновилась);
    - иначе — последнее слово, КОНЧИВШЕЕСЯ не позже паузы; отклонение считается
      от её начала (момента, когда речь смолкла).

    Слово, начавшееся до паузы и растянутое за её конец, не подходит ни туда, ни
    сюда: точка остаётся без слова и попадает в счётчик промахов, а не в среднее.
    """
    deltas: list[float] = []
    missed = 0
    cursor = 0
    for s, e in pauses:
        t = e if onset else s
        chosen = None
        for i in range(cursor, len(words)):
            w = words[i]
            v = w.start if onset else w.end
            if v > t + MATCH_WINDOW:
                break
            if v < t - MATCH_WINDOW:
                continue
            if onset:
                if w.start >= s:  # слово из-за паузы, а не растянутое через неё
                    chosen = i
                    break
            elif w.end <= e:
                chosen = i  # последнее подходящее в окне
        if chosen is None:
            missed += 1
            print(f"  {label} {t:6.2f}: слова в ±{MATCH_WINDOW:.0f} с нет")
            continue
        w = words[chosen]
        cursor = chosen + 1
        d = (w.start if onset else w.end) - t
        deltas.append(d)
        print(f"  {label} {t:6.2f} → {(w.start if onset else w.end):6.2f} ({d:+.3f})  {w.text!r}")
    return deltas, missed


def measure_media(words, media: Path) -> tuple[tuple[list[float], int], tuple[list[float], int]]:
    """|Δ| границ слов относительно энергии. Возвращает (по началам, по концам).

    Каждая половина — (отклонения в секундах, число точек эталона без слова).
    Число промахов возвращается вместе с отклонениями не для красоты: окно
    ±MATCH_WINDOW обрезает распределение сверху, и промах больше секунды иначе
    просто исчез бы из статистики, улучшив её.
    """
    onsets, offsets = speech_edges(media)
    if not onsets and not offsets:
        print("  !! silencedetect не нашёл пауз — на этом ролике мерить нечего")
        return ([], 0), ([], 0)

    # Возобновление речи — событие однозначное, и именно на нём видно, что
    # подсветка уехала. Уход в паузу мягче: хвост слова затухает постепенно, и
    # порог срабатывает раньше, чем слово физически кончилось. Поэтому числа
    # печатаются раздельно, а решение принимается по началам.
    on = _match_in_order(words, onsets, "возобновление", onset=True)
    off = _match_in_order(words, offsets, "уход в паузу ", onset=False)
    return on, off


# ------------------------------------------------------------------ прогон


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="small")
    ap.add_argument(
        "--backend", default="", help="auto|whisperx|faster_whisper (по умолчанию — из настроек)"
    )
    ap.add_argument(
        "--media", type=Path, help="ролик без разметки: эталон берётся из энергии сигнала"
    )
    ap.add_argument("--language", default="", help="язык (для --media; для фикстуры всегда en)")
    ap.add_argument("--no-trim", action="store_true", help="без обрезки тишины на краях слов")
    ap.add_argument("--no-vad", action="store_true", help="faster_whisper без VAD-фильтра")
    args = ap.parse_args()

    language = args.language or ("ru" if args.media else "en")
    # Настройки читаются из окружения ДО импорта конфига: Settings создаётся при
    # импорте модуля, и правка после него уже ни на что не влияет.
    os.environ["WHISPER_LANGUAGE"] = language
    os.environ["WHISPER_MODEL"] = args.model
    if args.backend:
        os.environ["ASR_BACKEND"] = args.backend
    if args.no_trim:
        os.environ["ASR_TRIM_WORD_SILENCE"] = "false"

    from src.core.logging import setup_logging
    from src.pipeline import asr as asr_mod

    setup_logging("WARNING")

    if args.no_vad:
        def patched(wav, *, language, model_name):
            """faster_whisper без VAD: проверка, не VAD ли двигает границы."""
            from faster_whisper import WhisperModel

            from src.core.config import settings

            key = ("fw-novad", model_name)
            model = asr_mod._MODEL_CACHE.get(key)
            if model is None:
                model = WhisperModel(
                    model_name, device="cpu", compute_type="int8",
                    cpu_threads=settings.whisper_cpu_threads,
                )
                asr_mod._MODEL_CACHE[key] = model
            segments, info = model.transcribe(
                str(wav),
                language=language,
                word_timestamps=True,
                beam_size=settings.whisper_beam_size,
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

    if args.media:
        media, truth = args.media, None
    else:
        media, truth = ensure()

    t0 = time.perf_counter()
    res = asr_mod.transcribe(
        media, language=language, model=args.model, backend=args.backend or None
    )
    elapsed = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    words = res.words
    print(
        f"бэкенд={res.backend} модель={res.model} язык={res.language} слов={len(words)} "
        f"обрезано={res.meta.get('trimmed_words', 0)}"
    )
    print(f"время {elapsed:.1f} с (холодный старт, модели грузятся), пик RSS {peak_mb:.0f} МБ\n")

    if truth is not None:
        deltas = measure_fixture(words, truth)
        stats = _stats(deltas)
        print("\n" + (f"среднее |Δ| = {stats[0]:.0f} мс, максимум = {stats[1]:.0f} мс"
                      if stats else "мерить нечего: ни одна фраза не сошлась с транскриптом"))
    else:
        (d_on, miss_on), (d_off, miss_off) = measure_media(words, media)
        print()
        print(_fmt(_stats(d_on), "возобновление речи", len(d_on), miss_on))
        print(_fmt(_stats(d_off), "уход в паузу     ", len(d_off), miss_off))

    # Не только точность: пропущенное слово или перекрытие ломает EDL целиком,
    # а по средним отклонениям этого не видно.
    bad = sum(1 for a, b in pairwise(words) if a.end > b.start + 1e-6)
    print(f"немонотонных пар: {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
