"""Абстрактный LLM-клиент (§4).

Вызывающий код знает тир (`cheap`/`smart`/`vision`), а не имя модели и не
провайдера. Промпты лежат в `prompts/*.md`, а не в питоновских строках.
Системная часть промпта одинакова между роликами → идёт в prompt cache, иначе
переплата (у DeepSeek кеш-попадание дешевле промаха в 30 раз).

Ретрай ровно один: второй провал → фолбэк, не третья попытка.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from src.core.config import Settings, settings as default_settings
from src.llm import jsonshape
from src.llm.providers import (
    Provider,
    ProviderError,
    make_provider,
    price_for,
    schema_hint,
)

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

SYSTEM_MARK = "<!-- system -->"
USER_MARK = "<!-- user -->"

# Модель иногда оборачивает JSON в markdown, даже когда просили не оборачивать.
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


class LLMUnavailable(RuntimeError):
    """Ключа нет или провайдер не отвечает. Вызывающий обязан уйти в фолбэк."""


class LLMBadOutput(ValueError):
    """Модель вернула не то: не JSON или не та форма. Лечится ретраем (§4)."""


class LLMBudgetExceeded(LLMBadOutput):
    """Ответ оборван по лимиту токенов. Ретрай с теми же настройками бесполезен."""


@dataclass(slots=True)
class Usage:
    prompt: str
    tier: str
    model: str
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    seconds: float = 0.0
    attempt: int = 1
    billed_usd: float | None = None

    @property
    def cost_usd(self) -> float:
        # Счёт шлюза точнее таблицы цен (в таблице нет наценки посредника),
        # поэтому он в приоритете. Но только ПОЛОЖИТЕЛЬНЫЙ: ноль значит «шлюз
        # не сказал», а не «вызов бесплатный» (дефект №4) — записанный нуль
        # молча обнулял бы себестоимость ролика в отчёте §7 без всякого сигнала.
        # `_billed_usd` в providers.py нули уже отсекает; здесь — чтобы и Usage,
        # собранный мимо провайдера, нельзя было обнулить.
        if self.billed_usd is not None and self.billed_usd > 0:
            return self.billed_usd
        miss, hit, out = price_for(self.model)
        return (
            self.input_tokens * miss
            + self.cache_read_tokens * hit
            + self.cache_creation_tokens * miss * 1.25
            + self.output_tokens * out
        ) / 1_000_000

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cost_usd"] = round(self.cost_usd, 6)
        return d


@dataclass(slots=True)
class LLMResult:
    data: dict[str, Any]
    usage: Usage
    raw: str = ""


@dataclass(slots=True)
class Prompt:
    name: str
    system: str
    user: str
    extra: dict[str, str] = field(default_factory=dict)


def load_prompt(name: str) -> Prompt:
    """Читает `prompts/<name>.md`, разделённый маркерами system/user."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"нет промпта {path}")
    text = path.read_text(encoding="utf-8")
    if SYSTEM_MARK not in text or USER_MARK not in text:
        raise ValueError(f"{path}: нужны маркеры {SYSTEM_MARK} и {USER_MARK}")
    _, rest = text.split(SYSTEM_MARK, 1)
    system, user = rest.split(USER_MARK, 1)
    return Prompt(name=name, system=system.strip(), user=user.strip())


_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def render(template: str, variables: dict[str, Any]) -> str:
    """Подстановка {{var}}. Не str.format: в промптах полно скобок из JSON."""
    missing: list[str] = []

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in variables:
            missing.append(key)
            return m.group(0)
        v = variables[key]
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

    out = _VAR_RE.sub(sub, template)
    if missing:
        raise KeyError(f"в промпте нет значений для: {sorted(set(missing))}")
    return out


def parse_json(text: str) -> Any:
    """JSON из ответа модели, с поправкой на markdown-обёртку."""
    stripped = text.strip()
    fenced = FENCE_RE.match(stripped)
    if fenced:
        stripped = fenced.group(1)
    return json.loads(stripped)


UsageSink = Callable[[Usage], Awaitable[None] | None]


class LLMClient:
    """Один вход для всех вызовов модели, каким бы ни был провайдер."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        usage_sink: UsageSink | None = None,
        cfg: Settings | None = None,
        provider: Provider | None = None,
    ) -> None:
        self.cfg = cfg or default_settings
        self._api_key = api_key if api_key is not None else self.cfg.llm_api_key
        self._usage_sink = usage_sink
        self._provider = provider
        self.usages: list[Usage] = []

    # ------------------------------------------------------------------ доступность
    @property
    def provider_name(self) -> str:
        return self._provider.name if self._provider else self.cfg.llm_provider

    @property
    def ready(self) -> bool:
        if self._provider is not None:
            return self.cfg.llm_enabled
        return bool(self._api_key) and self.cfg.llm_enabled

    def _ensure_provider(self) -> Provider:
        if not self.ready:
            raise LLMUnavailable(
                f"нет ключа для провайдера {self.cfg.llm_provider!r} "
                f"(или LLM_ENABLED=0) — работаем без LLM"
            )
        if self._provider is None:
            self._provider = make_provider(
                self.cfg.llm_provider,
                self._api_key,
                self.cfg.llm_base_url,
                self.cfg.llm_timeout_s,
                thinking=self.cfg.llm_thinking,
            )
        return self._provider

    # ------------------------------------------------------------------ основной вызов
    async def complete(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        schema: dict[str, Any],
        model_tier: str,
        *,
        images: Iterable[bytes] | None = None,
        audio: tuple[bytes, str] | None = None,
        correction: str | None = None,
        max_tokens: int = 16_000,
        effort: str | None = None,
    ) -> LLMResult:
        """Вызов модели со структурированным выводом. Ретрай — только транспорт.

        `correction` — текст ошибки для повторного вызова (§3): дописывается к
        пользовательской части, чтобы кеш системной уцелел.

        `audio` — (байты, mime). Провайдер, который звук не слушает, получает
        вызов без него: промпт обязан работать и так.
        """
        provider = self._ensure_provider()
        prompt = load_prompt(prompt_name)
        model = self.cfg.model_for_tier(model_tier)

        user_text = render(prompt.user, variables)
        if not provider.supports_schema:
            # Провайдер не умеет принуждать к схеме — кладём её в промпт,
            # а форму ответа проверим сами.
            user_text += schema_hint(schema)
        if correction:
            user_text += (
                "\n\n---\nПредыдущий ответ не подошёл:\n"
                f"{correction}\n"
                "Исправь ровно эти ошибки. Формат ответа тот же."
            )

        image_list = list(images or ())
        if image_list and not provider.supports_images:
            log.warning("провайдер %s не принимает картинки — вызов без них", provider.name)
            image_list = []
        # Ключ передаётся только со звуком: провайдеры без него (и подмены в
        # тестах) про этот аргумент не знают.
        extra: dict[str, Any] = {}
        if audio and provider.supports_audio:
            extra["audio"] = audio
        elif audio:
            log.info("провайдер %s не слушает звук — вызов по одному тексту", provider.name)

        last: ProviderError | None = None
        for attempt in range(1, self.cfg.llm_max_retries + 2):
            t0 = time.perf_counter()
            try:
                result = await provider.complete(
                    model=model,
                    system=prompt.system,
                    user=user_text,
                    schema=schema,
                    images=image_list,
                    max_tokens=max_tokens,
                    effort=effort,
                    **extra,
                )
            except ProviderError as e:
                last = e
                if not e.retryable or attempt > self.cfg.llm_max_retries:
                    raise LLMUnavailable(f"{prompt_name}: {e}") from e
                delay = 1.5 * attempt
                log.warning("LLM %s: %s, повтор через %.1fs", prompt_name, e, delay)
                await asyncio.sleep(delay)
                continue

            usage = Usage(
                prompt=prompt_name,
                tier=model_tier,
                model=model,
                provider=provider.name,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cache_read_tokens=result.cache_read_tokens,
                cache_creation_tokens=result.cache_creation_tokens,
                seconds=round(time.perf_counter() - t0, 2),
                attempt=attempt,
                billed_usd=result.billed_usd,
            )
            await self._record(usage)
            log.info(
                "LLM %s [%s/%s/%s] in=%d cached=%d out=%d $%.4f %.1fs",
                prompt_name, provider.name, model_tier, model,
                usage.input_tokens, usage.cache_read_tokens, usage.output_tokens,
                usage.cost_usd, usage.seconds,
            )

            try:
                data = parse_json(result.text)
            except json.JSONDecodeError as e:
                # Обрыв по лимиту токенов повторять НЕЛЬЗЯ: те же настройки дадут
                # тот же обрыв, а заплатим дважды. Проверено дорогой ценой —
                # два таких ретрая сожгли весь остаток баланса.
                if result.stop_reason in ("length", "max_tokens"):
                    raise LLMBudgetExceeded(
                        f"{prompt_name}: ответ оборван по лимиту токенов "
                        f"({result.output_tokens} шт.), повтор не поможет — "
                        f"уменьшите размышления модели или поднимите max_tokens"
                    ) from e
                raise LLMBadOutput(
                    f"{prompt_name}: ответ не разобрался как JSON "
                    f"(stop={result.stop_reason or '?'}): {e}"
                ) from e

            shape_errors = jsonshape.check(data, schema)
            if shape_errors:
                raise LLMBadOutput(f"{prompt_name}: форма ответа не по схеме: " + "; ".join(shape_errors[:6]))
            if not isinstance(data, dict):
                raise LLMBadOutput(f"{prompt_name}: ожидался объект, пришёл {type(data).__name__}")

            return LLMResult(data=data, usage=usage, raw=result.text)

        raise LLMUnavailable(f"{prompt_name}: {last}")

    async def _record(self, usage: Usage) -> None:
        self.usages.append(usage)
        if self._usage_sink is None:
            return
        res = self._usage_sink(usage)
        if asyncio.iscoroutine(res):
            await res

    async def aclose(self) -> None:
        if self._provider is not None:
            await self._provider.aclose()

    # ------------------------------------------------------------------ отчётность
    def total_cost(self) -> float:
        return sum(u.cost_usd for u in self.usages)

    def report(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "calls": [u.to_dict() for u in self.usages],
            "total_cost_usd": round(self.total_cost(), 6),
            "total_input_tokens": sum(u.input_tokens for u in self.usages),
            "total_output_tokens": sum(u.output_tokens for u in self.usages),
            "total_cache_read_tokens": sum(u.cache_read_tokens for u in self.usages),
        }
