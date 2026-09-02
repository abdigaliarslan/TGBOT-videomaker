"""Абстрактный LLM-клиент (§4).

Вызывающий код знает тир (`cheap`/`smart`/`vision`), а не имя модели. Промпты
лежат в `prompts/*.md`, а не в питоновских строках. Системная часть промпта
одинакова между роликами → идёт в prompt cache, иначе переплата вдвое.

Ретрай ровно один: второй провал → фолбэк, не третья попытка.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from src.core.config import settings

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

SYSTEM_MARK = "<!-- system -->"
USER_MARK = "<!-- user -->"

# $/1M токенов. Кеш-чтение ~0.1x входа, запись в кеш ~1.25x входа.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}
# Модели, которым передаём adaptive thinking. У остальных параметр вызовет 400.
THINKING_MODELS = ("claude-opus-5", "claude-fable-5", "claude-sonnet-5", "claude-opus-4-8")


class LLMUnavailable(RuntimeError):
    """Ключа нет или API не отвечает. Вызывающий обязан уйти в детерминированный фолбэк."""


class LLMBadOutput(ValueError):
    """Модель вернула JSON, который не прошёл валидацию вызывающего."""


@dataclass(slots=True)
class Usage:
    prompt: str
    tier: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    seconds: float = 0.0
    attempt: int = 1

    @property
    def cost_usd(self) -> float:
        inp, out = PRICES.get(self.model, (5.0, 25.0))
        return (
            self.input_tokens * inp
            + self.cache_read_tokens * inp * 0.1
            + self.cache_creation_tokens * inp * 1.25
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
    """Подстановка {{var}}. Не str.format: в промптах полно фигурных скобок из JSON."""
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


UsageSink = Callable[[Usage], Awaitable[None] | None]


class LLMClient:
    """Один вход для всех вызовов модели."""

    def __init__(self, *, api_key: str | None = None, usage_sink: UsageSink | None = None) -> None:
        self._api_key = api_key if api_key is not None else settings.anthropic_api_key
        self._usage_sink = usage_sink
        self._client: Any = None
        self.usages: list[Usage] = []

    # ------------------------------------------------------------------ доступность
    @property
    def ready(self) -> bool:
        return bool(self._api_key) and settings.llm_enabled

    def _ensure_client(self) -> Any:
        if not self.ready:
            raise LLMUnavailable(
                "ANTHROPIC_API_KEY не задан (или LLM_ENABLED=0) — работаем без LLM"
            )
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key,
                timeout=settings.llm_timeout_s,
                max_retries=0,  # ретраями управляем сами (§4: ровно один)
            )
        return self._client

    # ------------------------------------------------------------------ основной вызов
    async def complete(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        schema: dict[str, Any],
        model_tier: str,
        *,
        images: Iterable[bytes] | None = None,
        correction: str | None = None,
        max_tokens: int = 16_000,
        effort: str | None = None,
    ) -> LLMResult:
        """Вызов модели со structured output. Один ретрай при транспортной ошибке.

        `correction` — текст ошибки валидации для повторного вызова (§3): не новый
        промпт, а дописка к пользовательской части, чтобы кеш системной уцелел.
        """
        client = self._ensure_client()
        prompt = load_prompt(prompt_name)
        model = settings.model_for_tier(model_tier)

        user_text = render(prompt.user, variables)
        if correction:
            user_text += (
                "\n\n---\nПредыдущий ответ не прошёл валидацию:\n"
                f"{correction}\n"
                "Исправь ровно эти ошибки. Формат ответа тот же."
            )

        content: list[dict[str, Any]] = []
        for img in images or ():
            import base64

            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(img).decode("ascii"),
                    },
                }
            )
        content.append({"type": "text", "text": user_text})

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            # Системная часть идентична между роликами → кешируем её (§4).
            "system": [
                {
                    "type": "text",
                    "text": prompt.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": content}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if effort:
            kwargs["output_config"]["effort"] = effort
        if any(model.startswith(m) for m in THINKING_MODELS):
            kwargs["thinking"] = {"type": "adaptive"}

        last_error: Exception | None = None
        for attempt in range(1, settings.llm_max_retries + 2):
            t0 = time.perf_counter()
            try:
                message = await self._send(client, kwargs)
            except Exception as e:  # noqa: BLE001 — классифицируем ниже
                last_error = e
                if not _retryable(e) or attempt > settings.llm_max_retries:
                    raise LLMUnavailable(f"{prompt_name}: {type(e).__name__}: {e}") from e
                delay = 1.5 * attempt
                log.warning("LLM %s: %s, повтор через %.1fs", prompt_name, e, delay)
                await asyncio.sleep(delay)
                continue

            usage = _usage_from(message, prompt_name, model_tier, model, attempt)
            usage.seconds = round(time.perf_counter() - t0, 2)
            await self._record(usage)

            text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                # output_config гарантирует валидный JSON; сюда попадаем только при
                # обрыве по max_tokens.
                raise LLMBadOutput(
                    f"{prompt_name}: ответ не разобрался как JSON "
                    f"(stop_reason={getattr(message, 'stop_reason', '?')})"
                ) from e

            log.info(
                "LLM %s [%s/%s] in=%d cached=%d out=%d $%.4f %.1fs",
                prompt_name,
                model_tier,
                model,
                usage.input_tokens,
                usage.cache_read_tokens,
                usage.output_tokens,
                usage.cost_usd,
                usage.seconds,
            )
            return LLMResult(data=data, usage=usage, raw=text)

        raise LLMUnavailable(f"{prompt_name}: {last_error}")

    async def _send(self, client: Any, kwargs: dict[str, Any]) -> Any:
        """Стримим: у director большой max_tokens, нестримовый запрос упрётся в таймаут."""
        async with client.messages.stream(**kwargs) as stream:
            return await stream.get_final_message()

    async def _record(self, usage: Usage) -> None:
        self.usages.append(usage)
        if self._usage_sink is None:
            return
        res = self._usage_sink(usage)
        if asyncio.iscoroutine(res):
            await res

    # ------------------------------------------------------------------ отчётность
    def total_cost(self) -> float:
        return sum(u.cost_usd for u in self.usages)

    def report(self) -> dict[str, Any]:
        return {
            "calls": [u.to_dict() for u in self.usages],
            "total_cost_usd": round(self.total_cost(), 6),
            "total_input_tokens": sum(u.input_tokens for u in self.usages),
            "total_output_tokens": sum(u.output_tokens for u in self.usages),
            "total_cache_read_tokens": sum(u.cache_read_tokens for u in self.usages),
        }


def _usage_from(message: Any, prompt: str, tier: str, model: str, attempt: int) -> Usage:
    u = getattr(message, "usage", None)
    return Usage(
        prompt=prompt,
        tier=tier,
        model=model,
        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        cache_creation_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        attempt=attempt,
    )


def _retryable(e: Exception) -> bool:
    """Повторяем только транспорт и перегрузку. 400 повторять бессмысленно."""
    name = type(e).__name__
    if name in ("APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"):
        return True
    status = getattr(e, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)
