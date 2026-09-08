"""Тесты провайдеров: что именно уходит в API и как читается расход.

Сеть подменяется httpx.MockTransport — проверяется форма запроса и разбор
ответа, а не чужой сервер.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.llm.providers import (
    GeminiProvider,
    OpenAICompatProvider,
    ProviderError,
    make_provider,
    price_for,
    schema_hint,
)
from src.llm.schemas import CLEAN_SCHEMA

SCHEMA = CLEAN_SCHEMA


def mock(provider, handler):
    provider._http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=provider.base_url,
        headers={"Authorization": "Bearer test", "x-goog-api-key": "test"},
    )
    return provider


# ------------------------------------------------------------------ фабрика


def test_factory_knows_providers():
    for name in ("anthropic", "deepseek", "gemini", "openai"):
        p = make_provider(name, "k", "http://x")
        assert p.name == name


def test_factory_rejects_unknown():
    with pytest.raises(ValueError) as e:
        make_provider("нетакого", "k", "http://x")
    assert "deepseek" in str(e.value)


def test_deepseek_cannot_enforce_schema():
    """Ключевое ограничение: схему придётся класть в промпт."""
    assert make_provider("deepseek", "k", "http://x").supports_schema is False
    assert make_provider("anthropic", "k", "http://x").supports_schema is True
    assert make_provider("gemini", "k", "http://x").supports_schema is True


def test_schema_hint_contains_the_schema():
    hint = schema_hint(SCHEMA)
    assert '"fixes"' in hint and "JSON" in hint


# ------------------------------------------------------------------ openai-совместимый


async def test_openai_compat_request_shape():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"fixes": []}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1500, "completion_tokens": 200,
                      "prompt_cache_hit_tokens": 1000, "prompt_cache_miss_tokens": 500},
        })

    p = mock(OpenAICompatProvider("k", "https://api.deepseek.com"), handler)
    res = await p.complete(model="deepseek-v4-flash", system="СИСТЕМА", user="ЗАДАЧА", schema=SCHEMA)
    await p.aclose()

    # системное сообщение первое и отдельное — на нём держится кеш
    assert seen["messages"][0] == {"role": "system", "content": "СИСТЕМА"}
    assert seen["messages"][1]["content"] == "ЗАДАЧА"
    assert seen["response_format"] == {"type": "json_object"}
    assert res.text == '{"fixes": []}'


async def test_openai_compat_does_not_double_count_cached_tokens():
    """prompt_tokens включает кешированные — иначе себестоимость завысится."""
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1500, "completion_tokens": 200,
                      "prompt_cache_hit_tokens": 1000},
        })

    p = mock(OpenAICompatProvider("k", "https://api.deepseek.com"), handler)
    res = await p.complete(model="deepseek-v4-flash", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert res.cache_read_tokens == 1000
    assert res.input_tokens == 500
    assert res.input_tokens + res.cache_read_tokens == 1500


async def test_openai_compat_reads_openai_style_cache_field():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 10,
                      "prompt_tokens_details": {"cached_tokens": 400}},
        })

    p = mock(OpenAICompatProvider("k", "https://api.openai.com/v1"), handler)
    res = await p.complete(model="gpt-x", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert res.cache_read_tokens == 400
    assert res.input_tokens == 500


async def test_openai_compat_sends_images_as_data_urls():
    seen: dict = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": {}})

    p = mock(OpenAICompatProvider("k", "https://api.deepseek.com"), handler)
    await p.complete(model="m", system="s", user="u", schema=SCHEMA, images=[b"\xff\xd8jpeg"])
    await p.aclose()
    parts = seen["messages"][1]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[-1] == {"type": "text", "text": "u"}


@pytest.mark.parametrize("status,retryable", [(429, True), (500, True), (503, True), (400, False), (401, False)])
async def test_openai_compat_error_classification(status, retryable):
    def handler(request):
        return httpx.Response(status, json={"error": {"message": "нет"}})

    p = mock(OpenAICompatProvider("k", "https://api.deepseek.com"), handler)
    with pytest.raises(ProviderError) as e:
        await p.complete(model="m", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert e.value.retryable is retryable
    assert e.value.status == status


# ------------------------------------------------------------------ gemini


async def test_gemini_request_shape():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        assert "generateContent" in str(request.url)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": '{"fixes": []}'}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 800, "candidatesTokenCount": 60,
                              "cachedContentTokenCount": 300},
        })

    p = mock(GeminiProvider("k", "https://generativelanguage.googleapis.com/v1beta"), handler)
    res = await p.complete(model="gemini-3.7-flash", system="СИСТЕМА", user="ЗАДАЧА", schema=SCHEMA)
    await p.aclose()

    assert seen["systemInstruction"]["parts"][0]["text"] == "СИСТЕМА"
    assert seen["generationConfig"]["responseJsonSchema"] == SCHEMA
    assert seen["generationConfig"]["responseMimeType"] == "application/json"
    assert res.cache_read_tokens == 300
    assert res.input_tokens == 500
    assert res.text == '{"fixes": []}'


async def test_gemini_thinking_is_off_by_default():
    """Размышления считаются в maxOutputTokens и платятся как выход (§4)."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [], "usageMetadata": {}})

    p = mock(GeminiProvider("k", "https://x"), handler)
    await p.complete(model="gemini-3.7-flash", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert seen["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


async def test_gemini_thinking_level_when_asked():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [], "usageMetadata": {}})

    p = mock(GeminiProvider("k", "https://x", thinking=True), handler)
    await p.complete(model="m", system="s", user="u", schema=SCHEMA, effort="low")
    await p.aclose()
    assert seen["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}


async def test_gemini_retries_without_thinking_config_on_400():
    """gemini-3.6-flash отвечает 400 на thinkingConfig — модель не должна пропасть."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "thinkingConfig" in body["generationConfig"]:
            return httpx.Response(400, json={"error": {"message": "invalid argument"}})
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2},
        })

    p = mock(GeminiProvider("k", "https://x"), handler)
    res = await p.complete(model="gemini-3.6-flash", system="s", user="u", schema=SCHEMA)
    assert res.text == "{}"
    assert len(seen) == 2

    # Урок запомнен: второй раз спорную настройку не шлём вовсе.
    await p.complete(model="gemini-3.6-flash", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert len(seen) == 3
    assert "thinkingConfig" not in seen[-1]["generationConfig"]


async def test_gemini_400_without_thinking_config_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "плохая схема"}})

    p = mock(GeminiProvider("k", "https://x", thinking=True), handler)
    with pytest.raises(ProviderError):
        await p.complete(model="m", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert len(calls) == 1


async def test_gemini_counts_thoughts_as_output():
    """`candidatesTokenCount` их не включает, а счёт нам приходит за всё."""
    def handler(request):
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 40,
                              "thoughtsTokenCount": 260},
        })

    p = mock(GeminiProvider("k", "https://x"), handler)
    res = await p.complete(model="m", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert res.output_tokens == 300


async def test_gemini_empty_candidates_is_not_a_crash():
    def handler(request):
        return httpx.Response(200, json={"candidates": [], "usageMetadata": {}})

    p = mock(GeminiProvider("k", "https://x"), handler)
    res = await p.complete(model="m", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert res.text == ""


# ------------------------------------------------------------------ цены


def test_cache_price_is_cheaper_than_miss_for_every_model():
    from src.llm.providers import PRICES

    for model, (miss, hit, out) in PRICES.items():
        assert hit < miss, f"{model}: кеш не дешевле промаха"
        assert out > 0


def test_deepseek_cache_is_dramatically_cheaper():
    miss, hit, _ = price_for("deepseek-v4-flash")
    assert miss / hit > 20  # ради этого системный промпт и держим неизменным
