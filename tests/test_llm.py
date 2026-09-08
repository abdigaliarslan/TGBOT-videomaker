"""Тесты LLM-клиента (§4): промпты-файлы, тиры, учёт токенов, ретрай, схема."""

from __future__ import annotations

import json

import pytest

from src.core.config import PROVIDER_MODELS, Settings
from src.llm import client as client_mod
from src.llm.client import (
    LLMBadOutput,
    LLMClient,
    LLMUnavailable,
    Usage,
    load_prompt,
    parse_json,
    render,
)
from src.llm.providers import ProviderError, price_for
from src.llm.schemas import DIRECT_SCHEMA, SCHEMAS, TIERS
from tests.conftest import FakeProvider, result_of


def cfg(**kw) -> Settings:
    # _env_file=None: тест проверяет код, а не .env конкретной машины.
    base = dict(llm_provider="deepseek", deepseek_api_key="test", llm_max_retries=1)
    base.update(kw)
    return Settings(_env_file=None, **base)


def client(script, *, supports_schema=True, **kw) -> tuple[LLMClient, FakeProvider]:
    provider = FakeProvider(script, supports_schema=supports_schema)
    return LLMClient(cfg=cfg(**kw), provider=provider), provider


# ------------------------------------------------------------------ промпты


@pytest.mark.parametrize("name", ["direct", "clean_transcript", "plan_broll", "metadata"])
def test_every_prompt_file_parses(name):
    p = load_prompt(name)
    assert p.system and p.user
    assert "<!-- system -->" not in p.system


def test_prompt_exists_for_every_schema():
    for name in SCHEMAS:
        assert load_prompt(name).system


def test_every_prompt_has_a_tier():
    assert set(TIERS) == set(SCHEMAS)
    assert set(TIERS.values()) <= {"cheap", "smart", "vision"}


def test_render_substitutes_variables():
    assert render("привет {{name}}!", {"name": "мир"}) == "привет мир!"


def test_render_leaves_json_braces_alone():
    assert render('{"words": [0]} и {{v}}', {"v": "X"}) == '{"words": [0]} и X'


def test_render_raises_on_missing_variable():
    with pytest.raises(KeyError):
        render("нужен {{missing}}", {})


def test_direct_prompt_forbids_timecodes():
    p = load_prompt("direct")
    assert "таймкод" in p.system.lower() and "индекс" in p.system.lower()


def test_broll_prompt_forbids_text_on_images():
    assert "текст" in load_prompt("plan_broll").system.lower()


def test_direct_schema_has_no_time_fields():
    blob = json.dumps(DIRECT_SCHEMA)
    for forbidden in ("start", "end", "src_in", "src_out", "seconds", "time"):
        assert f'"{forbidden}"' not in blob


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


# ------------------------------------------------------------------ разбор ответа


def test_parse_json_plain():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_unwraps_markdown_fence():
    """Модель оборачивает JSON в ```json даже когда просили не оборачивать."""
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('```\n{"a": 2}\n```') == {"a": 2}


# ------------------------------------------------------------------ тиры и провайдеры


def test_tier_maps_to_model_not_hardcoded():
    s = Settings(_env_file=None, llm_provider="deepseek", llm_tier_smart="model-A")
    assert s.model_for_tier("smart") == "model-A"
    with pytest.raises(ValueError):
        s.model_for_tier("magic")


def test_empty_tier_falls_back_to_provider_default():
    for provider, models in PROVIDER_MODELS.items():
        s = Settings(_env_file=None, llm_provider=provider)
        for tier, model in models.items():
            assert s.model_for_tier(tier) == model


def test_api_key_follows_provider():
    s = Settings(_env_file=None, llm_provider="deepseek", deepseek_api_key="D", anthropic_api_key="A")
    assert s.llm_api_key == "D"
    assert s.llm_ready is True
    assert Settings(_env_file=None, llm_provider="gemini", anthropic_api_key="A").llm_ready is False


def test_every_provider_has_default_models():
    assert set(PROVIDER_MODELS) >= {"anthropic", "deepseek", "gemini"}
    for models in PROVIDER_MODELS.values():
        assert set(models) == {"cheap", "smart", "vision"}


# ------------------------------------------------------------------ учёт токенов


def test_cost_uses_cache_price_for_cached_tokens():
    miss, hit, out = price_for("deepseek-v4-flash")
    assert hit < miss  # иначе кеш не имел бы смысла

    cold = Usage(prompt="p", tier="smart", model="deepseek-v4-flash", input_tokens=1_000_000)
    warm = Usage(prompt="p", tier="smart", model="deepseek-v4-flash", cache_read_tokens=1_000_000)
    assert cold.cost_usd == pytest.approx(miss)
    assert warm.cost_usd == pytest.approx(hit)
    assert warm.cost_usd < cold.cost_usd / 10


def test_price_matches_prefix_for_dated_model():
    assert price_for("claude-opus-5-20260101") == price_for("claude-opus-5")


def test_unknown_model_still_priced():
    assert price_for("совсем-новая-модель")[0] > 0


def test_report_sums_calls():
    c = LLMClient(cfg=cfg(), provider=FakeProvider([]))
    c.usages = [
        Usage(prompt="direct", tier="smart", model="deepseek-v4-pro", input_tokens=1000, output_tokens=500),
        Usage(prompt="metadata", tier="cheap", model="deepseek-v4-flash", input_tokens=200, output_tokens=100),
    ]
    r = c.report()
    assert r["total_input_tokens"] == 1200
    assert r["total_output_tokens"] == 600
    assert r["total_cost_usd"] == pytest.approx(c.total_cost())


# ------------------------------------------------------------------ вызов


async def test_complete_returns_parsed_json():
    c, p = client([result_of({"fixes": []})])
    res = await c.complete("clean_transcript", {"language": "ru", "words_table": "0 раз"},
                           SCHEMAS["clean_transcript"], "cheap")
    assert res.data == {"fixes": []}
    assert res.usage.prompt == "clean_transcript"
    assert res.usage.tier == "cheap"
    assert res.usage.provider == "fake"
    assert res.usage.model == "deepseek-v4-flash"


async def test_system_stays_a_stable_prefix():
    """§4: системная часть одинакова между роликами — на ней держится кеш."""
    c, p = client([result_of({"fixes": []}), result_of({"fixes": []})])
    await c.complete("clean_transcript", {"language": "ru", "words_table": "первый ролик"},
                     SCHEMAS["clean_transcript"], "cheap")
    await c.complete("clean_transcript", {"language": "ru", "words_table": "второй ролик"},
                     SCHEMAS["clean_transcript"], "cheap")
    assert p.calls[0]["system"] == p.calls[1]["system"]
    assert "первый ролик" not in p.calls[0]["system"]
    assert "первый ролик" in p.calls[0]["user"]


async def test_schema_goes_natively_when_provider_supports_it():
    c, p = client([result_of({"fixes": []})], supports_schema=True)
    await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                     SCHEMAS["clean_transcript"], "cheap")
    assert p.calls[0]["schema"] is SCHEMAS["clean_transcript"]
    assert "json_schema" not in p.calls[0]["user"]


async def test_schema_goes_into_prompt_when_provider_cannot_enforce_it():
    """DeepSeek не умеет json_schema — схему кладём в промпт."""
    c, p = client([result_of({"fixes": []})], supports_schema=False)
    await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                     SCHEMAS["clean_transcript"], "cheap")
    assert '"fixes"' in p.calls[0]["user"]


async def test_usage_is_recorded_and_sunk():
    seen: list[Usage] = []
    provider = FakeProvider([result_of({"fixes": []}, input_tokens=300, cache_read_tokens=7000)])
    c = LLMClient(cfg=cfg(), provider=provider, usage_sink=seen.append)
    await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                     SCHEMAS["clean_transcript"], "cheap")
    assert len(seen) == 1
    assert seen[0].cache_read_tokens == 7000
    assert c.usages == seen


async def test_exactly_one_retry_on_transport_error(monkeypatch):
    """§4: ретрай ровно один."""
    monkeypatch.setattr(client_mod.asyncio, "sleep", _noop)
    c, p = client([ProviderError("сеть", retryable=True), result_of({"fixes": []})])
    res = await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                           SCHEMAS["clean_transcript"], "cheap")
    assert res.data == {"fixes": []}
    assert res.usage.attempt == 2


async def test_second_transport_failure_gives_up(monkeypatch):
    monkeypatch.setattr(client_mod.asyncio, "sleep", _noop)
    c, p = client([ProviderError("сеть", retryable=True), ProviderError("опять", retryable=True)])
    with pytest.raises(LLMUnavailable):
        await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                         SCHEMAS["clean_transcript"], "cheap")
    assert len(p.calls) == 2


async def test_non_retryable_error_is_not_retried():
    c, p = client([ProviderError("плохой запрос", status=400, retryable=False)])
    with pytest.raises(LLMUnavailable):
        await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                         SCHEMAS["clean_transcript"], "cheap")
    assert len(p.calls) == 1


async def test_bad_json_raises_bad_output():
    c, p = client([result_of("это не json")])
    with pytest.raises(LLMBadOutput):
        await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                         SCHEMAS["clean_transcript"], "cheap")


async def test_wrong_shape_raises_bad_output():
    """json_object пропустит что угодно — форму ловим сами."""
    c, p = client([result_of({"fixes": "не список"})])
    with pytest.raises(LLMBadOutput) as e:
        await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                         SCHEMAS["clean_transcript"], "cheap")
    assert "форма" in str(e.value)


async def test_no_key_means_unavailable():
    c = LLMClient(cfg=Settings(_env_file=None, llm_provider="deepseek", deepseek_api_key=""))
    assert c.ready is False
    with pytest.raises(LLMUnavailable):
        await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                         SCHEMAS["clean_transcript"], "cheap")


async def test_correction_goes_into_user_part_not_system():
    c, p = client([result_of({"fixes": []})])
    await c.complete("clean_transcript", {"language": "ru", "words_table": "x"},
                     SCHEMAS["clean_transcript"], "cheap", correction="- блок пустой")
    assert "блок пустой" not in p.calls[0]["system"]
    assert "блок пустой" in p.calls[0]["user"]


async def test_images_dropped_for_text_only_provider():
    c, p = client([result_of({"items": []})], supports_schema=True)
    p.supports_images = False
    await c.complete("plan_broll",
                     {"language": "ru", "duration": "10", "max_items": "1", "words_table": "0 раз"},
                     SCHEMAS["plan_broll"], "vision", images=[b"jpegbytes"])
    assert p.calls[0]["images"] == []


async def _noop(*_args, **_kwargs):
    return None
