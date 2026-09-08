"""Поведение режиссёра вокруг LLM: приём, один ретрай, фолбэк (§3).

Ключевое требование: что бы модель ни вернула, наружу выходит валидный EDL.
"""

from __future__ import annotations

import pytest

from src.core.config import Settings
from src.core.edl import Word, validate_edl
from src.llm.client import LLMClient
from src.pipeline.director import direct
from tests.conftest import FakeProvider, result_of


def cfg() -> Settings:
    return Settings(
        _env_file=None,
        sub_max_words=4, sub_max_chars=40, sub_gap_split=0.6, cut_max_gap=0.8,
        llm_provider="deepseek", deepseek_api_key="test",
    )


def words(n: int = 9) -> list[Word]:
    return [Word(idx=i, text=f"сл{i}", start=i * 0.4, end=i * 0.4 + 0.3) for i in range(n)]


def good_plan() -> dict:
    return {
        "hook_end_word": 2,
        "drop_words": [],
        "blocks": [
            {"words": [0, 1, 2], "accent_word": 1, "accent": "accent", "position": "auto", "follow_face": False},
            {"words": [3, 4, 5], "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
            {"words": [6, 7, 8], "accent_word": 8, "accent": "shout", "position": "auto", "follow_face": False},
        ],
    }


def broken_plan() -> dict:
    """Слова 4 и 5 потеряны — валидатор обязан это поймать."""
    return {
        "hook_end_word": None,
        "drop_words": [],
        "blocks": [
            {"words": [0, 1, 2], "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
            {"words": [3, 6, 7, 8], "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
        ],
    }


def client_with(*payloads) -> LLMClient:
    provider = FakeProvider([result_of(p) for p in payloads])
    c = LLMClient(cfg=cfg(), provider=provider)
    c.fake = provider  # type: ignore[attr-defined]
    return c


async def test_good_plan_is_accepted():
    ws = words()
    llm = client_with(good_plan())
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert [cw.word_idx for b in edl.captions for cw in b.words] == list(range(9))
    assert any(cw.emphasis.value == "shout" for b in edl.captions for cw in b.words)
    assert len(llm.fake.calls) == 1


async def test_broken_plan_triggers_exactly_one_retry():
    ws = words()
    llm = client_with(broken_plan(), good_plan())
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert len(llm.fake.calls) == 2, "должен быть ровно один ретрай"


async def test_retry_carries_the_validation_error():
    ws = words()
    llm = client_with(broken_plan(), good_plan())
    await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    second = llm.fake.calls[1]["user"]
    assert "Предыдущий ответ не подошёл" in second
    assert "не попали ни в один блок" in second


async def test_two_bad_plans_fall_back_to_deterministic():
    """§3: второй провал → фолбэк, не третья попытка. Ролик всё равно соберётся."""
    ws = words()
    llm = client_with(broken_plan(), broken_plan())
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert len(llm.fake.calls) == 2
    # Фолбэк не ЧИТАЕТ смысл, но иерархию кегля держит: без неё блок выходит
    # стеной ровного крупного текста (С1 сделал кегль иерархией). Ровно одно
    # слово на блок, выбранное детерминированно, — см. `lead_word`.
    for block in edl.captions:
        assert sum(1 for cw in block.words if cw.emphasis.value != "none") == 1


async def test_llm_outage_falls_back_without_raising():
    from src.llm.providers import ProviderError

    ws = words()
    llm = LLMClient(cfg=cfg(), provider=FakeProvider(
        [ProviderError("сеть", retryable=True), ProviderError("сеть", retryable=True)]))
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []


async def test_no_llm_at_all_still_produces_edl():
    ws = words()
    edl = await direct(ws, 4.5, llm=None, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert [cw.word_idx for b in edl.captions for cw in b.words] == list(range(9))


async def test_llm_returning_garbage_indices_is_survivable():
    """Несуществующие индексы схему не нарушают — их отсеивает сборка EDL."""
    ws = words(6)
    plan = {
        "hook_end_word": 999,
        "drop_words": [-5, 999],
        "blocks": [
            {"words": [0, 1, 2, 999], "accent_word": 999, "accent": "shout", "position": "auto", "follow_face": False},
            {"words": [3, 4, 5], "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
        ],
    }
    llm = client_with(plan)
    edl = await direct(ws, 3.0, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert [cw.word_idx for b in edl.captions for cw in b.words] == [0, 1, 2, 3, 4, 5]


async def test_out_of_range_value_costs_a_retry():
    """Блок из шести слов схему нарушает — ловится ДО сборки EDL и лечится ретраем."""
    ws = words(6)
    good = {
        "hook_end_word": None, "drop_words": [],
        "blocks": [{"words": [0, 1, 2], "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
                   {"words": [3, 4, 5], "accent_word": None, "accent": None, "position": "auto", "follow_face": False}],
    }
    bad = {
        "hook_end_word": None, "drop_words": [],
        "blocks": [{"words": [0, 1, 2, 3, 4, 5], "accent_word": None, "accent": None,
                    "position": "auto", "follow_face": False}],
    }
    llm = client_with(bad, good)
    edl = await direct(ws, 3.0, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert len(llm.fake.calls) == 2
    assert "максимум 5" in llm.fake.calls[1]["user"]


async def test_dropped_words_are_excluded_from_captions_and_cuts():
    ws = words(9)
    plan = {
        "hook_end_word": 2,
        "drop_words": [4],
        "blocks": [
            {"words": [0, 1, 2], "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
            {"words": [3, 5], "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
            {"words": [6, 7, 8], "accent_word": None, "accent": None, "position": "auto", "follow_face": False},
        ],
    }
    llm = client_with(plan)
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    captioned = [cw.word_idx for b in edl.captions for cw in b.words]
    assert 4 not in captioned
    assert any(c.reason == "drop_filler" for c in edl.cuts)


async def test_words_table_is_sent_not_raw_text():
    """Модель обязана видеть индексы, иначе ей нечем ссылаться (§0.1)."""
    ws = words(4)
    llm = client_with(
        {
            "hook_end_word": None,
            "drop_words": [],
            "blocks": [{"words": [0, 1, 2, 3], "accent_word": None, "accent": None, "position": "auto", "follow_face": False}],
        }
    )
    await direct(ws, 2.0, llm=llm, language="ru", cfg=cfg())
    sent = llm.fake.calls[0]["user"]
    assert "   0 сл0" in sent
    assert "   3 сл3" in sent
