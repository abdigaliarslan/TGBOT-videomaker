"""Поведение режиссёра вокруг LLM: приём, один ретрай, фолбэк (§3).

Ключевое требование: что бы модель ни вернула, наружу выходит валидный EDL.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.core.config import Settings
from src.core.edl import Word, validate_edl
from src.llm.client import LLMClient
from src.pipeline.director import direct
from tests.test_llm import FakeClient, message_with


def cfg() -> Settings:
    return Settings(sub_max_words=4, sub_max_chars=40, sub_gap_split=0.6, cut_max_gap=0.8)


def words(n: int = 9) -> list[Word]:
    return [Word(idx=i, text=f"сл{i}", start=i * 0.4, end=i * 0.4 + 0.3) for i in range(n)]


def good_plan() -> dict:
    return {
        "hook_end_word": 2,
        "drop_words": [],
        "blocks": [
            {"words": [0, 1, 2], "accent_word": 1, "accent": "accent", "position": "auto"},
            {"words": [3, 4, 5], "accent_word": None, "accent": None, "position": "auto"},
            {"words": [6, 7, 8], "accent_word": 8, "accent": "shout", "position": "auto"},
        ],
        "camera": [],
    }


def broken_plan() -> dict:
    """Слова 4 и 5 потеряны — валидатор обязан это поймать."""
    return {
        "hook_end_word": None,
        "drop_words": [],
        "blocks": [
            {"words": [0, 1, 2], "accent_word": None, "accent": None, "position": "auto"},
            {"words": [3, 6, 7, 8], "accent_word": None, "accent": None, "position": "auto"},
        ],
        "camera": [],
    }


def client_with(*payloads) -> LLMClient:
    c = LLMClient(api_key="test")
    c._client = FakeClient([message_with(p) for p in payloads])
    return c


async def test_good_plan_is_accepted():
    ws = words()
    llm = client_with(good_plan())
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert [cw.word_idx for b in edl.captions for cw in b.words] == list(range(9))
    assert any(cw.emphasis.value == "shout" for b in edl.captions for cw in b.words)
    assert len(llm._client.messages.calls) == 1


async def test_broken_plan_triggers_exactly_one_retry():
    ws = words()
    llm = client_with(broken_plan(), good_plan())
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert len(llm._client.messages.calls) == 2, "должен быть ровно один ретрай"


async def test_retry_carries_the_validation_error():
    ws = words()
    llm = client_with(broken_plan(), good_plan())
    await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    second = llm._client.messages.calls[1]["messages"][0]["content"][-1]["text"]
    assert "не прошёл валидацию" in second
    assert "не попали ни в один блок" in second


async def test_two_bad_plans_fall_back_to_deterministic():
    """§3: второй провал → фолбэк, не третья попытка. Ролик всё равно соберётся."""
    ws = words()
    llm = client_with(broken_plan(), broken_plan())
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert len(llm._client.messages.calls) == 2
    # фолбэк не расставляет акценты
    assert all(cw.emphasis.value == "none" for b in edl.captions for cw in b.words)


async def test_llm_outage_falls_back_without_raising():
    from tests.test_llm import Retryable

    ws = words()
    llm = LLMClient(api_key="test")
    llm._client = FakeClient([Retryable("сеть"), Retryable("сеть")])
    edl = await direct(ws, 4.5, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []


async def test_no_llm_at_all_still_produces_edl():
    ws = words()
    edl = await direct(ws, 4.5, llm=None, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert [cw.word_idx for b in edl.captions for cw in b.words] == list(range(9))


async def test_llm_returning_garbage_indices_is_survivable():
    ws = words(6)
    plan = {
        "hook_end_word": 999,
        "drop_words": [-5, 999],
        "blocks": [
            {"words": [0, 1, 2, 999], "accent_word": 999, "accent": "shout", "position": "auto"},
            {"words": [3, 4, 5], "accent_word": None, "accent": None, "position": "auto"},
        ],
        "camera": [{"word_idx": 999, "type": "zoom_punch", "scale": 99.0}],
    }
    llm = client_with(plan)
    edl = await direct(ws, 3.0, llm=llm, language="ru", cfg=cfg())
    assert validate_edl(edl) == []
    assert edl.camera == []


async def test_dropped_words_are_excluded_from_captions_and_cuts():
    ws = words(9)
    plan = {
        "hook_end_word": 2,
        "drop_words": [4],
        "blocks": [
            {"words": [0, 1, 2], "accent_word": None, "accent": None, "position": "auto"},
            {"words": [3, 5], "accent_word": None, "accent": None, "position": "auto"},
            {"words": [6, 7, 8], "accent_word": None, "accent": None, "position": "auto"},
        ],
        "camera": [],
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
            "blocks": [{"words": [0, 1, 2, 3], "accent_word": None, "accent": None, "position": "auto"}],
            "camera": [],
        }
    )
    await direct(ws, 2.0, llm=llm, language="ru", cfg=cfg())
    sent = llm._client.messages.calls[0]["messages"][0]["content"][-1]["text"]
    assert "   0 сл0" in sent
    assert "   3 сл3" in sent
