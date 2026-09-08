"""DoD M1: тайминги слов совпадают с речью в пределах 100 мс.

Тесты, которые грузят модели, помечены `slow` и по умолчанию пропускаются:
`pytest -m slow` запускает их явно. Выбор бэкенда и выпрямление таймингов —
чистая логика, она проверяется всегда.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from src.pipeline.probe import probe

TOLERANCE_MEAN = 0.100
TOLERANCE_MAX = 0.250


def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


# ------------------------------------------------------------------ выбор бэкенда


def test_auto_prefers_whisperx(monkeypatch):
    """Границы слов важнее скорости: whisperx промахивается на 46 мс, whisper — на 234."""
    from src.pipeline import asr

    monkeypatch.setattr(asr, "_has_whisperx", lambda: True)
    assert asr.pick_backend("auto") == "whisperx"


def test_auto_falls_back_to_faster_whisper(monkeypatch):
    from src.pipeline import asr

    monkeypatch.setattr(asr, "_has_whisperx", lambda: False)
    assert asr.pick_backend("auto") == "faster_whisper"


def test_explicit_backend_wins_over_auto(monkeypatch):
    from src.pipeline import asr

    monkeypatch.setattr(asr, "_has_whisperx", lambda: True)
    assert asr.pick_backend("faster_whisper") == "faster_whisper"


def test_asking_for_missing_whisperx_is_an_error(monkeypatch):
    """Молча подменять бэкенд нельзя: тайминги разъедутся, а причины не видно."""
    from src.pipeline import asr

    monkeypatch.setattr(asr, "_has_whisperx", lambda: False)
    with pytest.raises(RuntimeError):
        asr.pick_backend("whisperx")


# ------------------------------------------------------------------ инвариант таймингов


def test_finalize_straightens_overlaps():
    """Перекрытие границ = подсветка на двух словах разом. Выпрямляем на входе."""
    from src.pipeline.asr import _finalize

    words = _finalize([("раз", 0.0, 1.0, 0.9), ("два", 0.5, 1.4, 0.9)], duration=2.0)
    assert [w.idx for w in words] == [0, 1]
    assert words[0].end <= words[1].start


def test_finalize_reindexes_after_dropping_empty():
    """Индекс равен позиции — на этом стоит весь EDL (§3)."""
    from src.pipeline.asr import _finalize

    words = _finalize(
        [("раз", 0.0, 0.4, 0.9), ("  ", 0.4, 0.6, 0.9), ("два", 0.6, 1.0, 0.9)], duration=2.0
    )
    assert [(w.idx, w.text) for w in words] == [(0, "раз"), (1, "два")]


def test_finalize_gives_zero_length_words_a_duration():
    from src.pipeline.asr import _finalize

    words = _finalize([("раз", 1.0, 1.0, 0.9)], duration=2.0)
    assert words[0].end > words[0].start


# ------------------------------------------------------- обрезка тишины на краях
#
# Оба бэкенда растягивают последнее слово перед паузой на саму паузу, а нарезка
# считает паузы как зазор МЕЖДУ словами — и не видит спрятанную. Замер на
# эталонном ролике: без обрезки whisperx вырезает 0.97 с из 20.01, с обрезкой —
# 3.43 с (tests/asr_accuracy.py --media).


def _wav(path, spans, *, duration=2.0, sr=16000):
    """wav, где `spans` — отрезки (start, end) с тоном, остальное тишина."""
    import math
    import struct
    import wave

    total = int(duration * sr)
    data = [0.0] * total
    for start, end in spans:
        for i in range(int(start * sr), min(int(end * sr), total)):
            data[i] = 0.5 * math.sin(2 * math.pi * 220.0 * i / sr)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(b"".join(struct.pack("<h", int(v * 32767)) for v in data))
    return path


def test_trim_cuts_silence_swallowed_by_the_word(tmp_path):
    """Слово, растянутое на паузу, возвращается к концу речи."""
    from src.core.edl import Word
    from src.pipeline.asr import trim_word_silence

    wav = _wav(tmp_path / "a.wav", [(0.0, 0.5), (1.5, 2.0)])
    words = [
        Word(idx=0, text="раз", start=0.0, end=1.5),
        Word(idx=1, text="два", start=1.5, end=2.0),
    ]
    assert trim_word_silence(words, wav) == 1
    assert words[0].end == pytest.approx(0.5, abs=0.05)
    assert words[1].end == pytest.approx(2.0, abs=0.05)


def test_trim_moves_the_start_forward_over_silence(tmp_path):
    """Начало, поставленное в тишину, тоже неверно: подсветка загорается раньше речи."""
    from src.core.edl import Word
    from src.pipeline.asr import trim_word_silence

    wav = _wav(tmp_path / "a.wav", [(0.0, 0.3), (1.0, 2.0)])
    words = [
        Word(idx=0, text="раз", start=0.0, end=0.3),
        Word(idx=1, text="два", start=0.35, end=2.0),
    ]
    trim_word_silence(words, wav)
    assert words[1].start == pytest.approx(1.0, abs=0.05)


def test_trim_never_grows_a_word(tmp_path):
    """Обрезка не имеет права ДОБАВИТЬ времени: время речи не выдумывается."""
    from src.core.edl import Word
    from src.pipeline.asr import trim_word_silence

    wav = _wav(tmp_path / "a.wav", [(0.0, 2.0)])
    words = [Word(idx=0, text="раз", start=0.5, end=1.0)]
    trim_word_silence(words, wav)
    assert (words[0].start, words[0].end) == (0.5, 1.0)


def test_trim_ignores_short_silence(tmp_path):
    """Провал короче порога — это смычка внутри слова, а не пауза."""
    from src.core.edl import Word
    from src.pipeline.asr import trim_word_silence

    wav = _wav(tmp_path / "a.wav", [(0.0, 0.9), (0.98, 2.0)])
    words = [Word(idx=0, text="раз", start=0.0, end=1.05)]
    assert trim_word_silence(words, wav) == 0


def test_trim_keeps_timings_monotonic(tmp_path):
    """Границы едут только внутрь слова — инвариант EDL не может сломаться."""
    from src.core.edl import Word
    from src.pipeline.asr import trim_word_silence

    wav = _wav(tmp_path / "a.wav", [(0.0, 0.4), (1.6, 2.0)])
    words = [
        Word(idx=0, text="раз", start=0.0, end=1.0),
        Word(idx=1, text="два", start=1.0, end=1.7),
        Word(idx=2, text="три", start=1.7, end=2.0),
    ]
    trim_word_silence(words, wav)
    for a, b in pairwise(words):
        assert a.end <= b.start + 1e-6
        assert a.end > a.start


def test_trim_survives_a_file_it_cannot_read(tmp_path):
    """Не PCM, не wav, нет файла — обрезки просто не будет, задача не падает."""
    from src.core.edl import Word
    from src.pipeline.asr import trim_word_silence

    junk = tmp_path / "a.wav"
    junk.write_bytes(b"not a wav at all")
    words = [Word(idx=0, text="раз", start=0.0, end=1.0)]
    assert trim_word_silence(words, junk) == 0
    assert trim_word_silence(words, tmp_path / "missing.wav") == 0
    assert (words[0].start, words[0].end) == (0.0, 1.0)


def test_trim_survives_a_truncated_pcm(tmp_path):
    """Усечённый wav (нечётное число байт) — не повод ронять готовый транскрипт.

    Раньше `wave` открывал такой файл штатно, а np.frombuffer падал уже за
    пределами try — то есть задача умирала ПОСЛЕ успешного распознавания.
    """
    from src.core.edl import Word
    from src.pipeline.asr import trim_word_silence

    wav = _wav(tmp_path / "a.wav", [(0.0, 0.5), (1.5, 2.0)])
    cut = tmp_path / "cut.wav"
    cut.write_bytes(wav.read_bytes()[:-1])
    words = [Word(idx=0, text="раз", start=0.0, end=1.0)]
    changed = trim_word_silence(words, cut)  # главное — что не бросает
    assert changed >= 0
    assert 0.0 <= words[0].start < words[0].end <= 1.0


def test_trim_gives_up_on_signal_without_dynamics(tmp_path):
    """Ровный уровень: порога «речь/тишина» нет, и выдумывать его нельзя."""
    from src.core.edl import Word
    from src.pipeline.asr import trim_word_silence

    wav = _wav(tmp_path / "a.wav", [(0.0, 2.0)])
    words = [Word(idx=0, text="раз", start=0.0, end=2.0)]
    assert trim_word_silence(words, wav) == 0


# ------------------------------------------------------------- сам измеритель
#
# Измеритель — часть доказательной базы выбора бэкенда, поэтому он проверяется
# так же, как код продукта: сломанный измеритель печатает «0 мс» и выглядит как
# успех.


def _has_ffmpeg() -> bool:
    import shutil

    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not _has_ffmpeg(), reason="нужен ffmpeg/ffprobe")
def test_speech_edges_finds_the_pause_in_a_clip_that_starts_with_speech(tmp_path):
    """Ролик БЕЗ ведущей тишины: единственная пауза обязана попасть в эталон.

    Отсев ведущей тишины раньше стоял на сравнении `starts[0] < ends[0]`, которое
    истинно всегда, и выбрасывал первое настоящее возобновление речи. На таком
    ролике измеритель оставался без точек и печатал идеальный ноль.
    """
    from tests.asr_accuracy import speech_edges

    wav = _wav(tmp_path / "a.wav", [(0.0, 2.0), (3.0, 5.0)], duration=5.0)
    onsets, offsets = speech_edges(wav)
    assert [round(e, 1) for _s, e in onsets] == [3.0]
    assert [round(s, 1) for s, _e in offsets] == [2.0]


@pytest.mark.skipif(not _has_ffmpeg(), reason="нужен ffmpeg/ffprobe")
def test_speech_edges_drops_leading_and_trailing_silence(tmp_path):
    """Тишина в начале файла — не уход в паузу, тишина в конце — не возобновление."""
    from tests.asr_accuracy import speech_edges

    wav = _wav(tmp_path / "a.wav", [(1.0, 2.0), (3.0, 4.0)], duration=5.0)
    onsets, offsets = speech_edges(wav)
    assert [round(e, 1) for _s, e in onsets] == [3.0]  # 1.0 — начало файла, 5.0 — конец
    assert [round(s, 1) for s, _e in offsets] == [2.0, 4.0]


def test_measure_media_does_not_reward_a_backend_that_never_heard_the_audio(tmp_path):
    """Сопоставление по порядку, а не «ближайшее слово».

    «Бэкенд», печатающий слово каждые 100 мс, на ближайшем слове набирал 37 мс и
    выглядел точнее настоящих: метрика мерила плотность границ, а не выравнивание.
    """
    if not _has_ffmpeg():
        pytest.skip("нужен ffmpeg/ffprobe")
    from src.core.edl import Word
    from tests.asr_accuracy import _stats, measure_media

    wav = _wav(tmp_path / "a.wav", [(0.0, 2.0), (3.0, 5.0)], duration=5.0)
    dense = [
        Word(idx=i, text="x", start=round(i * 0.1, 3), end=round(i * 0.1 + 0.09, 3))
        for i in range(50)
    ]
    (d_on, _miss_on), _off = measure_media(dense, wav)
    stats = _stats(d_on)
    assert stats is not None
    assert stats[0] > 0.030 * 1000, "плотная сетка слов не должна выглядеть точной"


def test_stats_says_nothing_to_measure_instead_of_zero():
    """Пустой список — это «мерить нечего», а не «отклонение 0 мс»."""
    from tests.asr_accuracy import _stats

    assert _stats([]) is None
    assert _stats([0.1])[0] == pytest.approx(100.0)


# ------------------------------------------------------------------ на моделях (slow)


@pytest.fixture(scope="module")
def fixture_video():
    from tests.fixtures.make_fixture import ensure

    return ensure()


@pytest.mark.slow
def test_probe_matches_fixture(fixture_video):
    video, truth = fixture_video
    info = probe(video)
    assert info.width == truth["width"]
    assert info.height == truth["height"]
    assert info.has_audio is True
    assert info.duration == pytest.approx(truth["duration"], abs=0.15)
    assert info.fps == pytest.approx(truth["fps"], abs=0.1)


@pytest.mark.slow
def test_word_timings_match_speech(fixture_video):
    """DoD меряется на бэкенде, который код выбирает САМ (`auto`).

    Не на том, что стоит в .env: локальная настройка — это не свойство продукта,
    а faster_whisper на этой же фикстуре даёт 216 мс и завалил бы проверку по
    причине, к коду отношения не имеющей. Но и молчать о расхождении нельзя:
    зелёная галочка на бэкенде, которым машина не пользуется, — худший вариант,
    поэтому расхождение поднимается предупреждением (pytest покажет его в
    сводке), а не печатается в поток, который никто не читает.
    """
    import warnings

    from src.core.config import settings
    from src.pipeline.asr import pick_backend, transcribe

    backend = pick_backend("auto")
    if backend != "whisperx":
        pytest.skip(
            "whisperx не установлен, а faster_whisper не проходит DoD "
            "(216 мс среднего отклонения на этой фикстуре): pip install whisperx"
        )
    configured = pick_backend(settings.asr_backend)
    if configured != backend:
        warnings.warn(
            f"ASR_BACKEND в настройках даёт {configured}, а DoD меряется на {backend}: "
            f"на этой машине продукт поедет на другом бэкенде, чем проверяет тест",
            UserWarning,
            stacklevel=2,
        )

    video, truth = fixture_video
    result = transcribe(video, language="en", model="small", backend=backend)

    assert result.words, "ASR не вернул ни одного слова"
    # индексы обязаны совпадать с позицией — на этом стоит весь EDL
    assert [w.idx for w in result.words] == list(range(len(result.words)))
    # тайминги монотонны и не перекрываются
    for a, b in zip(result.words, result.words[1:]):
        assert a.end <= b.start + 1e-6, f"перекрытие: {a.text!r} и {b.text!r}"
        assert a.end > a.start

    deltas: list[float] = []
    cursor = 0
    for phrase in truth["phrases"]:
        expected = [norm(w) for w in phrase["words"]]
        first = next(
            (
                i
                for i in range(cursor, len(result.words))
                if norm(result.words[i].text) == expected[0]
            ),
            None,
        )
        assert first is not None, f"фраза не найдена в транскрипте: {phrase['text']!r}"
        last = first + len(expected) - 1
        assert last < len(result.words), f"фраза обрезана: {phrase['text']!r}"
        # Совпадения одного первого слова мало: склей ASR два слова — и `last`
        # укажет на чужое, а «отклонение» посчитается от границы другой фразы.
        got = [norm(w.text) for w in result.words[first : last + 1]]
        assert got == expected, f"слова фразы {phrase['text']!r} не совпали: {' '.join(got)}"
        deltas.append(abs(result.words[first].start - phrase["start"]))
        deltas.append(abs(result.words[last].end - phrase["end"]))
        cursor = last + 1

    mean = sum(deltas) / len(deltas)
    worst = max(deltas)
    print(f"\nотклонение: среднее {mean * 1000:.0f} мс, максимум {worst * 1000:.0f} мс")
    assert mean <= TOLERANCE_MEAN, (
        f"среднее отклонение {mean * 1000:.0f} мс > {TOLERANCE_MEAN * 1000:.0f} мс"
    )
    assert worst <= TOLERANCE_MAX, f"максимум {worst * 1000:.0f} мс > {TOLERANCE_MAX * 1000:.0f} мс"


@pytest.mark.slow
def test_asr_output_feeds_director(fixture_video):
    """Транскрипт из ASR должен без правок собираться в валидный EDL.

    Здесь бэкенд НАМЕРЕННО берётся из настроек, а не форсируется: это проверка
    того пути, по которому пойдёт задача на этой машине. Какой бэкенд сработал —
    печатается, чтобы «3 passed» не скрывало, что мерили и что гоняли.
    """
    from src.core.edl import validate_edl
    from src.pipeline.asr import transcribe
    from src.pipeline.director import deterministic_edl

    video, truth = fixture_video
    result = transcribe(video, language="en", model="small")
    print(f"\nEDL собран на бэкенде из настроек: {result.backend}/{result.model}")
    edl = deterministic_edl(result.words, truth["duration"])
    assert validate_edl(edl) == []
    assert edl.out_duration < truth["duration"], "паузы должны были сократить ролик"
