"""Тесты LLM-клиента (§4): промпты-файлы, кеширование, учёт токенов, ретрай.

Сеть не трогаем: подставляем двойник SDK. Проверяется наша обвязка, а не Anthropic.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.core.config import Settings
from src.llm import client as client_mod
from src.llm.client import (
    LLMClient,
    LLMUnavailable,
    PRICES,
    Usage,
    load_prompt,
    render,
)
from src.llm.schemas import DIRECT_SCHEMA, SCHEMAS, TIERS


# ------------------------------------------------------------------ промпты


@pytest.mark.parametrize("name", ["direct", "clean_transcript", "plan_broll", "metadata"])
def test_every_prompt_file_parses(name):
    p = load_prompt(name)
    assert p.system and p.user
    assert "<!-- system -->" not in p.system


def test_prompt_files_exist_for_every_schema():
    for name in SCHEMAS:
        assert load_prompt(name).system


def test_every_prompt_has_a_tier():
    assert set(TIERS) == set(SCHEMAS)
    for tier in TIERS.values():
        assert tier in ("cheap", "smart", "vision")


def test_render_substitutes_variables():
    assert render("привет {{name}}!", {"name": "мир"}) == "привет мир!"


def test_render_leaves_json_braces_alone():
    """В промптах полно фигурных скобок из примеров JSON — их трогать нельзя."""
    template = '{"words": [0, 1]} и {{var}}'
    assert render(template, {"var": "X"}) == '{"words": [0, 1]} и X'


def test_render_raises_on_missing_variable():
    with pytest.raises(KeyError):
        render("нужен {{missing}}", {})


def test_direct_prompt_forbids_timecodes():
    """§0.1: модель не должна возвращать таймкоды — это должно быть в промпте."""
    p = load_prompt("direct")
    assert "таймкод" in p.system.lower()
    assert "индекс" in p.system.lower()


def test_broll_prompt_forbids_text_on_images():
    """§5 M7 ⚠: моделям запрещено рисовать текст на картинках."""
    assert "текст" in load_prompt("plan_broll").system.lower()


def test_direct_schema_has_no_time_fields():
    """Схема — последний рубеж: если поля времени нет, модель его и не вернёт."""
    blob = json.dumps(DIRECT_SCHEMA)
    for forbidden in ("start", "end", "src_in", "src_out", "seconds", "time"):
        assert f'"{forbidden}"' not in blob, f"в схеме direct есть поле времени: {forbidden}"


def test_schemas_are_closed():
    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
            for v in node.values():
                check(v)
        elif isinstance(node, list):
            for v in node:
                check(v)

    for schema in SCHEMAS.values():
        check(schema)


# ------------------------------------------------------------------ учёт токенов


def test_cost_math():
    u = Usage(
        prompt="direct", tier="smart", model="claude-opus-5",
        input_tokens=1_000_000, output_tokens=0,
    )
    assert u.cost_usd == pytest.approx(PRICES["claude-opus-5"][0])

    cached = Usage(
        prompt="direct", tier="smart", model="claude-opus-5",
        input_tokens=0, cache_read_tokens=1_000_000,
    )
    # чтение из кеша примерно в десять раз дешевле обычного входа
    assert cached.cost_usd == pytest.approx(PRICES["claude-opus-5"][0] * 0.1)


def test_cost_report_sums_calls():
    c = LLMClient(api_key="x")
    c.usages = [
        Usage(prompt="direct", tier="smart", model="claude-opus-5", input_tokens=1000, output_tokens=500),
        Usage(prompt="metadata", tier="cheap", model="claude-haiku-4-5", input_tokens=200, output_tokens=100),
    ]
    report = c.report()
    assert report["total_input_tokens"] == 1200
    assert report["total_output_tokens"] == 600
    assert report["total_cost_usd"] == pytest.approx(c.total_cost())


# ------------------------------------------------------------------ вызов


class FakeStream:
    def __init__(self, message):
        self._message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self._message


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return FakeStream(nxt)


class FakeClient:
    def __init__(self, script):
        self.messages = FakeMessages(script)


def message_with(payload: dict, **usage):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload, ensure_ascii=False))],
        usage=SimpleNamespace(
            input_tokens=usage.get("input_tokens", 100),
            output_tokens=usage.get("output_tokens", 50),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        ),
        stop_reason="end_turn",
    )


class Retryable(Exception):
    pass


Retryable.__name__ = "APIConnectionError"


async def test_complete_returns_parsed_json():
    c = LLMClient(api_key="test")
    c._client = FakeClient([message_with({"fixes": []})])
    result = await c.complete("clean_transcript", {"language": "ru", "words_table": "0 раз"},
                              SCHEMAS["clean_transcript"], "cheap")
    assert result.data == {"fixes": []}
    assert result.usage.prompt == "clean_transcript"
    assert result.usage.tier == "cheap"


async def test_system_prompt_is_cached():
    """§4: системный промпт одинаков между роликами → prompt caching обязателен."""
    c = LLMClient(api_key="test")
    c._client = FakeClient([message_with({"fixes": []})])
    await c.complete("clean_transcript", {"language": "ru", "words_table": "0 раз"},
                     SCHEMAS["clean_transcript"], "cheap")
    kwargs = c._client.messages.calls[0]
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    # волатильное (транскрипт) — в сообщении, а не в системной части
    assert "0 раз" not in kwargs["system"][0]["text"]


async def test_structured_output_is_requested():
    c = LLMClient(api_key="test")
    c._client = FakeClient([message_with({"fixes": []})])
    await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                     SCHEMAS["clean_transcript"], "cheap")
    fmt = c._client.messages.calls[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"] is SCHEMAS["clean_transcript"]


async def test_usage_is_recorded_and_sunk():
    seen: list[Usage] = []
    c = LLMClient(api_key="test", usage_sink=seen.append)
    c._client = FakeClient([message_with({"fixes": []}, input_tokens=300, cache_read_input_tokens=7000)])
    await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                     SCHEMAS["clean_transcript"], "cheap")
    assert len(seen) == 1
    assert seen[0].cache_read_tokens == 7000
    assert seen[0].input_tokens == 300
    assert c.usages == seen


async def test_exactly_one_retry_on_transport_error(monkeypatch):
    """§4: ретрай ровно один. Второй провал — фолбэк, не третья попытка."""
    monkeypatch.setattr(client_mod.asyncio, "sleep", lambda _: _noop())
    c = LLMClient(api_key="test")
    c._client = FakeClient([Retryable("сеть"), message_with({"fixes": []})])
    result = await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                              SCHEMAS["clean_transcript"], "cheap")
    assert result.data == {"fixes": []}
    assert result.usage.attempt == 2


async def test_second_failure_gives_up(monkeypatch):
    monkeypatch.setattr(client_mod.asyncio, "sleep", lambda _: _noop())
    c = LLMClient(api_key="test")
    c._client = FakeClient([Retryable("сеть"), Retryable("опять сеть")])
    with pytest.raises(LLMUnavailable):
        await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                         SCHEMAS["clean_transcript"], "cheap")
    assert len(c._client.messages.calls) == 2


async def test_non_retryable_error_is_not_retried():
    c = LLMClient(api_key="test")
    c._client = FakeClient([ValueError("плохой запрос")])
    with pytest.raises(LLMUnavailable):
        await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                         SCHEMAS["clean_transcript"], "cheap")
    assert len(c._client.messages.calls) == 1


async def test_no_key_means_unavailable():
    c = LLMClient(api_key="")
    assert c.ready is False
    with pytest.raises(LLMUnavailable):
        await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                         SCHEMAS["clean_transcript"], "cheap")


async def test_correction_goes_into_user_part_not_system():
    """Ретрай с текстом ошибки не должен ломать кеш системной части."""
    c = LLMClient(api_key="test")
    c._client = FakeClient([message_with({"fixes": []})])
    await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                     SCHEMAS["clean_transcript"], "cheap", correction="- блок пустой")
    kwargs = c._client.messages.calls[0]
    assert "блок пустой" not in kwargs["system"][0]["text"]
    user_text = kwargs["messages"][0]["content"][-1]["text"]
    assert "блок пустой" in user_text


def test_tier_maps_to_model_not_hardcoded():
    cfg = Settings(llm_tier_smart="model-A", llm_tier_cheap="model-B", llm_tier_vision="model-C")
    assert cfg.model_for_tier("smart") == "model-A"
    assert cfg.model_for_tier("cheap") == "model-B"
    assert cfg.model_for_tier("vision") == "model-C"
    with pytest.raises(ValueError):
        cfg.model_for_tier("magic")


async def _noop():
    return None
