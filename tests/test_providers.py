"""Тесты провайдеров: что именно уходит в API и как читается расход.

Сеть подменяется httpx.MockTransport — проверяется форма запроса и разбор
ответа, а не чужой сервер.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.llm.providers import (
    FalProvider,
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
    for name in ("anthropic", "deepseek", "gemini", "openai", "fal"):
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


# ------------------------------------------------- 2xx с телом не от модели (дефект №3)
#
# fal → OpenRouter → Google — две прокси, а прокси отдают HTML-страницу ошибки
# с кодом 200. `resp.json()` кидает на ней json.JSONDecodeError, то есть
# ValueError: не ProviderError, а значит ни ретрая (§4), ни фолбэка в
# `director`/`broll` — задача падает целиком вместо деградации.

HTML_ERROR_PAGE = "<html><head><title>502 Bad Gateway</title></head><body>nginx</body></html>"


def html_200(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=HTML_ERROR_PAGE, headers={"content-type": "text/html"})


async def test_openai_compat_html_body_on_200_is_a_retryable_provider_error():
    p = mock(OpenAICompatProvider("k", "https://api.deepseek.com"), html_200)
    with pytest.raises(ProviderError) as e:
        await p.complete(model="m", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert e.value.retryable is True
    assert "502 Bad Gateway" in str(e.value)  # оператору видно, кто именно ответил


async def test_fal_html_body_on_200_is_a_retryable_provider_error():
    """Ровно тот путь, который сейчас в проде: шлюз поверх шлюза."""
    p = mock(FalProvider("k", "https://fal.run/openrouter/router/openai/v1"), html_200)
    with pytest.raises(ProviderError) as e:
        await p.complete(model="google/gemini-3.5-flash", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert e.value.retryable is True


async def test_gemini_html_body_on_200_is_a_retryable_provider_error():
    p = mock(GeminiProvider("k", "https://x"), html_200)
    with pytest.raises(ProviderError) as e:
        await p.complete(model="m", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert e.value.retryable is True


@pytest.mark.parametrize("payload", ['["не", "объект"]', '"строка"', "42", "null"])
async def test_valid_json_that_is_not_an_object_is_also_a_provider_error(payload):
    """`data.get(...)` на списке даёт AttributeError — то же необработанное падение."""
    def handler(request):
        return httpx.Response(200, text=payload, headers={"content-type": "application/json"})

    p = mock(OpenAICompatProvider("k", "https://x"), handler)
    with pytest.raises(ProviderError) as e:
        await p.complete(model="m", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert e.value.retryable is True


def test_anthropic_sdk_json_failure_is_retryable_too():
    """Три провайдера — одно поведение: чужой разовый сбой повторяем."""
    from src.llm.providers import _sdk_retryable

    assert _sdk_retryable(json.JSONDecodeError("Expecting value", "<html>", 0)) is True

    class APIResponseValidationError(Exception):
        pass

    assert _sdk_retryable(APIResponseValidationError("ответ не той формы")) is True
    # А вот плохой запрос повторять по-прежнему нельзя.
    assert _sdk_retryable(ValueError("просто ошибка")) is False


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


async def test_gemini_hears_the_voice_before_reading_the_task():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}],
            "usageMetadata": {},
        })

    p = mock(GeminiProvider("k", "https://x"), handler)
    await p.complete(model="m", system="s", user="u", schema=SCHEMA, audio=(b"ID3mp3", "audio/mp3"))
    await p.aclose()
    parts = seen["contents"][0]["parts"]
    assert parts[0]["inlineData"]["mimeType"] == "audio/mp3"
    assert parts[-1] == {"text": "u"}


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


# ------------------------------------------------------------------ fal


def fal_ok(seen: list, *, cost: float = 0.0012):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"auth": request.headers.get("authorization"), "body": json.loads(request.content)})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"fixes": []}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 20, "cost": cost},
        })
    return handler


async def test_fal_speaks_openai_with_its_own_key_and_a_real_schema():
    seen: list = []
    p = mock(FalProvider("fal-key", "https://fal.run/openrouter/router/openai/v1"), fal_ok(seen))
    res = await p.complete(model="google/gemini-3.1-flash-lite", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    body = seen[0]["body"]
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["reasoning"] == {"enabled": False}
    assert res.billed_usd == pytest.approx(0.0012)


async def test_fal_authorizes_with_its_key_scheme():
    p = FalProvider("fal-key", "https://x")
    assert p._ensure().headers["authorization"] == "Key fal-key"
    await p.aclose()


async def test_fal_hands_the_voice_to_the_model():
    seen: list = []
    p = mock(FalProvider("k", "https://x"), fal_ok(seen))
    await p.complete(model="m", system="s", user="u", schema=SCHEMA, audio=(b"ID3", "audio/mp3"))
    await p.aclose()
    parts = seen[0]["body"]["messages"][1]["content"]
    assert parts[0]["type"] == "input_audio" and parts[0]["input_audio"]["format"] == "mp3"
    assert parts[-1] == {"type": "text", "text": "u"}


async def test_fal_falls_back_to_minimal_reasoning_once_and_remembers():
    """У flash размышления не отключаются: 400 «Reasoning is mandatory»."""
    seen: list = []
    ok = fal_ok(seen)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["reasoning"] == {"enabled": False}:
            seen.append({"auth": None, "body": body})
            return httpx.Response(400, json={"error": {"message": "Reasoning is mandatory for this endpoint and cannot be disabled."}})
        return ok(request)

    p = mock(FalProvider("k", "https://x"), handler)
    await p.complete(model="google/gemini-3.5-flash", system="s", user="u", schema=SCHEMA)
    await p.complete(model="google/gemini-3.5-flash", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert [c["body"]["reasoning"] for c in seen] == [
        {"enabled": False}, {"effort": "minimal"}, {"effort": "minimal"},
    ]


async def test_deepseek_is_not_handed_audio_it_cannot_hear():
    assert OpenAICompatProvider.supports_audio is False


# ------------------------------------------------- стоимость от шлюза (дефект №4)


@pytest.mark.parametrize("cost", [0, 0.0, -1.0, None, "0.01", True, False])
async def test_a_cost_the_gateway_did_not_really_report_falls_back_to_the_price_table(cost):
    """Ноль — это «шлюз не посчитал», а не «вызов бесплатный».

    Записанный нуль молча обнулял бы себестоимость ролика в отчёте §7 навсегда
    и без сигнала. А `isinstance(True, int)` — истина, и `cost: true` раньше
    превращался в $1.00.
    """
    seen: list = []
    p = mock(FalProvider("k", "https://x"), fal_ok(seen, cost=cost))
    res = await p.complete(model="google/gemini-3.5-flash", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert res.billed_usd is None


async def test_a_real_cost_from_the_gateway_wins_over_the_table():
    """Счёт шлюза точнее таблицы: в таблице нет наценки посредника."""
    seen: list = []
    p = mock(FalProvider("k", "https://x"), fal_ok(seen, cost=0.00875))
    res = await p.complete(model="google/gemini-3.5-flash", system="s", user="u", schema=SCHEMA)
    await p.aclose()
    assert res.billed_usd == pytest.approx(0.00875)
