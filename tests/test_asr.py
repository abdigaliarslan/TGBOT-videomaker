"""DoD M1: тайминги слов совпадают с речью в пределах 100 мс.

Тест медленный (грузит модели), поэтому помечен `slow` и по умолчанию
пропускается: `pytest -m slow` запускает его явно.
"""

from __future__ import annotations

import pytest

from src.pipeline.probe import probe

pytestmark = pytest.mark.slow

TOLERANCE_MEAN = 0.100
TOLERANCE_MAX = 0.250


def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


@pytest.fixture(scope="module")
def fixture_video():
    from tests.fixtures.make_fixture import ensure

    return ensure()


def test_probe_matches_fixture(fixture_video):
    video, truth = fixture_video
    info = probe(video)
    assert info.width == truth["width"]
    assert info.height == truth["height"]
    assert info.has_audio is True
    assert info.duration == pytest.approx(truth["duration"], abs=0.15)
    assert info.fps == pytest.approx(truth["fps"], abs=0.1)


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
