"""DoD M1: тайминги слов совпадают с речью в пределах 100 мс.

Тесты, которые грузят модели, помечены `slow` и по умолчанию пропускаются:
`pytest -m slow` запускает их явно. Выбор бэкенда и выпрямление таймингов —
чистая логика, она проверяется всегда.
"""

from __future__ import annotations

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
    from src.pipeline.asr import transcribe

    video, truth = fixture_video
    result = transcribe(video, language="en", model="small")

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
            (i for i in range(cursor, len(result.words)) if norm(result.words[i].text) == expected[0]),
            None,
        )
        assert first is not None, f"фраза не найдена в транскрипте: {phrase['text']!r}"
        last = first + len(expected) - 1
        assert last < len(result.words), f"фраза обрезана: {phrase['text']!r}"
        deltas.append(abs(result.words[first].start - phrase["start"]))
        deltas.append(abs(result.words[last].end - phrase["end"]))
        cursor = last + 1

    mean = sum(deltas) / len(deltas)
    worst = max(deltas)
    print(f"\nотклонение: среднее {mean * 1000:.0f} мс, максимум {worst * 1000:.0f} мс")
    assert mean <= TOLERANCE_MEAN, f"среднее отклонение {mean * 1000:.0f} мс > {TOLERANCE_MEAN * 1000:.0f} мс"
    assert worst <= TOLERANCE_MAX, f"максимум {worst * 1000:.0f} мс > {TOLERANCE_MAX * 1000:.0f} мс"


@pytest.mark.slow
def test_asr_output_feeds_director(fixture_video):
    """Транскрипт из ASR должен без правок собираться в валидный EDL."""
    from src.core.edl import validate_edl
    from src.pipeline.asr import transcribe
    from src.pipeline.director import deterministic_edl

    video, truth = fixture_video
    result = transcribe(video, language="en", model="small")
    edl = deterministic_edl(result.words, truth["duration"])
    assert validate_edl(edl) == []
    assert edl.out_duration < truth["duration"], "паузы должны были сократить ролик"
