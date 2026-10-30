"""Провайдеры моделей за одним интерфейсом.

§4 требует один абстрактный клиент и тиры вместо имён моделей в вызывающем коде.
Провайдер — такая же деталь реализации, как имя модели: меняется в конфиге,
логика пайплайна о нём не знает.

Провайдеры различаются в двух местах, и оба важны для нас:

1. **Принуждение к схеме.** Anthropic и Gemini умеют жёстко ограничить ответ
   JSON-схемой. DeepSeek на сегодня умеет только «верни любой JSON», поэтому
   схему приходится класть в промпт, а форму проверять у себя (`jsonshape`).
2. **Кеширование промпта.** У Anthropic оно явное (`cache_control`), у DeepSeek
   и Gemini — автоматическое. Общее одно: системная часть должна быть
   неизменной и идти первой, иначе кеш не сработает и мы переплатим.
"""

from __future__ import annotations

import base64
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ProviderResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    stop_reason: str = ""
    #: сколько вызов стоил по счёту провайдера, если он это сообщает (fal) —
    #: точнее нашей таблицы цен: в ней нет наценки посредника.
    #: None — не сообщил; нуля здесь не бывает по смыслу, см. `_billed_usd`
    billed_usd: float | None = None


class ProviderError(RuntimeError):
    """Ошибка обращения к провайдеру. `retryable` решает судьбу ретрая (§4)."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


# --------------------------------------------------------------------------- цены

# $/1M токенов: (вход-промах, вход-кеш, выход).
#
# У DeepSeek указаны пиковые (дорогие) ставки: лучше переоценить себестоимость,
# чем недооценить. Кеш-попадание там в 30 раз дешевле промаха — ради этого
# системная часть промпта и вынесена в неизменяемый префикс.
PRICES: dict[str, tuple[float, float, float]] = {
    # anthropic
    "claude-opus-5": (5.00, 0.50, 25.00),
    "claude-fable-5": (10.00, 1.00, 50.00),
    "claude-sonnet-5": (2.00, 0.20, 10.00),
    "claude-haiku-4-5": (1.00, 0.10, 5.00),
    "claude-opus-4-8": (5.00, 0.50, 25.00),
    "claude-sonnet-4-6": (3.00, 0.30, 15.00),
    # deepseek (пиковые ставки)
    "deepseek-v4-flash": (0.44, 0.014, 1.32),
    "deepseek-v4-pro": (1.32, 0.044, 3.96),
    "deepseek-v4-flash-vision-exp": (0.44, 0.014, 1.32),
    # gemini
    "gemini-3.1-pro-preview": (1.25, 0.31, 10.00),
    "gemini-3.7-flash": (0.30, 0.075, 2.50),
    # Официальной цены на 3.5-flash под рукой нет — ставим по линейке flash.
    # Ошибка в эту сторону завышает себестоимость, а не занижает.
    "gemini-3.5-flash": (0.30, 0.075, 2.50),
    "gemini-3.1-flash-lite": (0.10, 0.025, 0.40),
}
FALLBACK_PRICE = (1.00, 0.10, 5.00)


def price_for(model: str) -> tuple[float, float, float]:
    # У посредников (fal → OpenRouter) модель приходит с вендором:
    # «google/gemini-3.5-flash».
    model = model.rsplit("/", 1)[-1]
    if model in PRICES:
        return PRICES[model]
    for known, price in PRICES.items():  # «claude-opus-5-20260101» → «claude-opus-5»
        if model.startswith(known):
            return price
    return FALLBACK_PRICE


# --------------------------------------------------------------------------- база


class Provider(ABC):
    name: str = "?"
    #: умеет ли провайдер принудить ответ к JSON-схеме
    supports_schema: bool = False
    #: умеет ли принимать картинки
    supports_images: bool = False
    #: умеет ли слушать звук (`audio=(байты, mime)` в `complete`)
    supports_audio: bool = False

    def __init__(self, api_key: str, base_url: str, timeout: float = 180.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @abstractmethod
    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        images: Sequence[bytes] = (),
        max_tokens: int = 16_000,
        effort: str | None = None,
    ) -> ProviderResult: ...

    async def aclose(self) -> None:  # pragma: no cover — переопределяют не все
        return None


def _http_error(e: httpx.HTTPStatusError) -> ProviderError:
    status = e.response.status_code
    body = e.response.text[:400]
    # 429 и 5xx — временные. 400/401/403 повторять бессмысленно.
    return ProviderError(f"HTTP {status}: {body}", status=status, retryable=status == 429 or status >= 500)


def _bad_body(resp: httpx.Response, why: Exception) -> ProviderError:
    """2xx, но тело не похоже на ответ модели (дефект №3).

    Так отвечает не модель, а посредник на пути к ней: fal → OpenRouter →
    Google — это две прокси, а прокси отдают HTML-страницу ошибки с кодом 200.
    `resp.json()` кидает на ней `json.JSONDecodeError`, то есть ValueError, а не
    ProviderError. Без перевода он проходит мимо ретрая (§4) И мимо фолбэков
    `director`/`broll` — те ловят LLMUnavailable и LLMBadOutput, — и роняет
    задачу целиком вместо того, чтобы деградировать до детерминированного EDL.

    Ретраить можно и нужно: сбой чужой и, как правило, разовый.
    """
    return ProviderError(
        f"HTTP {resp.status_code}, но тело не разбирается как ответ модели "
        f"({why}): {resp.text[:200]!r}",
        status=resp.status_code,
        retryable=True,
    )


def _json_object(resp: httpx.Response) -> dict[str, Any]:
    """Тело ответа объектом JSON или retryable ProviderError. См. `_bad_body`."""
    try:
        data = resp.json()
    except ValueError as e:  # json.JSONDecodeError — его подвид
        raise _bad_body(resp, e) from e
    if not isinstance(data, dict):
        # Валидный JSON, но не объект: `data.get(...)` ниже упал бы
        # AttributeError — тем же необработанным исключением, что и выше.
        raise _bad_body(resp, TypeError(f"ожидался объект, пришёл {type(data).__name__}"))
    return data


def _billed_usd(cost: Any) -> float | None:
    """Стоимость, названная шлюзом. None — «шлюз её не назвал» (дефект №4).

    Два капкана. Первый: `isinstance(True, int)` — истина, и `cost: true`
    посчитался бы как $1.00. Второй: ноль. Вызов модели не бывает бесплатным, и
    `cost: 0` значит «шлюз не посчитал», а не «даром»; записанный нуль молча
    обнулял бы себестоимость ролика в отчёте §7 навсегда и без сигнала. В обоих
    случаях честнее вернуться к таблице цен: она хотя бы завышает (в ней нет
    наценки посредника), а не обнуляет.
    """
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    value = float(cost)
    return value if value > 0 else None


# --------------------------------------------------------------------------- anthropic


class AnthropicProvider(Provider):
    name = "anthropic"
    supports_schema = True
    supports_images = True

    def __init__(self, api_key: str, base_url: str, timeout: float = 180.0) -> None:
        super().__init__(api_key, base_url, timeout)
        self._client: Any = None

    def _ensure(self) -> Any:
        if self._client is None:
            import anthropic

            kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout, "max_retries": 0}
            if self.base_url and self.base_url != "https://api.anthropic.com":
                kwargs["base_url"] = self.base_url
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def complete(
        self, *, model, system, user, schema, images=(), max_tokens=16_000, effort=None
    ) -> ProviderResult:
        client = self._ensure()
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(img).decode("ascii"),
                },
            }
            for img in images
        ]
        content.append({"type": "text", "text": user})

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            # Системная часть идентична между роликами → кешируем явно (§4).
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": content}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if effort:
            kwargs["output_config"]["effort"] = effort
        if model.startswith(("claude-opus-5", "claude-fable-5", "claude-sonnet-5", "claude-opus-4-8")):
            kwargs["thinking"] = {"type": "adaptive"}

        try:
            async with client.messages.stream(**kwargs) as stream:
                message = await stream.get_final_message()
        except Exception as e:  # noqa: BLE001 — классифицируем по имени класса SDK
            raise ProviderError(f"{type(e).__name__}: {e}", retryable=_sdk_retryable(e)) from e

        usage = getattr(message, "usage", None)
        return ProviderResult(
            text="".join(b.text for b in message.content if getattr(b, "type", "") == "text"),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_creation_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            stop_reason=str(getattr(message, "stop_reason", "") or ""),
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _sdk_retryable(e: Exception) -> bool:
    if type(e).__name__ in (
        "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"
    ):
        return True
    # 2xx с телом не-JSON у SDK выглядит так: сам JSONDecodeError или
    # APIResponseValidationError («ответ не той формы»). Через `except Exception`
    # в `complete` это УЖЕ становится ProviderError, но неретраябельной — а сбой
    # тут чужой и разовый, ровно как у httpx-провайдеров (дефект №3). Держим три
    # провайдера в одном поведении.
    if isinstance(e, json.JSONDecodeError) or type(e).__name__ == "APIResponseValidationError":
        return True
    status = getattr(e, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


# --------------------------------------------------------------------------- openai-совместимые


class OpenAICompatProvider(Provider):
    """DeepSeek и всё, что говорит на диалекте /chat/completions.

    `json_schema` в `response_format` у DeepSeek сейчас отключён («This
    response_format type is unavailable now»), поэтому схему кладём в промпт и
    проверяем форму сами. Кеширование контекста автоматическое: в ответе
    приходят `prompt_cache_hit_tokens`.
    """

    name = "openai-compat"
    supports_schema = False
    supports_images = True
    #: схема авторизации в заголовке: `Bearer <ключ>` у DeepSeek, `Key <ключ>` у fal
    auth_scheme = "Bearer"

    def __init__(
        self, api_key: str, base_url: str, timeout: float = 180.0, *, thinking: bool = False
    ) -> None:
        super().__init__(api_key, base_url, timeout)
        self._http: httpx.AsyncClient | None = None
        self.thinking = thinking

    def _ensure(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"{self.auth_scheme} {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._http

    def _tune(
        self, body: dict[str, Any], *, model: str, schema: dict[str, Any], effort: str | None
    ) -> None:
        """Диалект провайдера: формат ответа и размышления. Здесь — DeepSeek."""
        # Размышления считаются в max_tokens И оплачиваются по цене выхода.
        # Наши задачи структурные (разложить слова по блокам), глубокого
        # рассуждения не требуют. Замерено на реальном ролике: с включёнными
        # размышлениями `direct` съел все 16000 токенов, не выдал ни байта
        # ответа (stop=length) и стоил $0.022 впустую. Выключаем по умолчанию.
        if not self.thinking:
            body["thinking"] = {"type": "disabled"}
        elif effort in ("low", "medium", "high"):
            body["reasoning_effort"] = effort

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        http = self._ensure()
        try:
            resp = await http.post("/chat/completions", json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise _http_error(e) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"{type(e).__name__}: {e}", retryable=True) from e
        # Разбор тела — ОТДЕЛЬНО от запроса: на 2xx с HTML-страницей прокси он
        # кидает ValueError, который не ProviderError (дефект №3).
        return _json_object(resp)

    async def complete(
        self, *, model, system, user, schema, images=(), max_tokens=16_000, effort=None, audio=None
    ) -> ProviderResult:
        if images or audio:
            parts: list[dict[str, Any]] = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                        + base64.standard_b64encode(img).decode("ascii")
                    },
                }
                for img in images
            ]
            if audio:
                data, mime = audio
                parts.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.standard_b64encode(data).decode("ascii"),
                            "format": mime.rsplit("/", 1)[-1],
                        },
                    }
                )
            parts.append({"type": "text", "text": user})
            user_content: Any = parts
        else:
            user_content = user

        body = {
            "model": model,
            # Системное сообщение первым и неизменным — на этом стоит кеш.
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "temperature": 0.4,
        }
        self._tune(body, model=model, schema=schema, effort=effort)
        data = await self._send(body, model=model)

        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        cached = int(usage.get("prompt_cache_hit_tokens") or 0)
        if not cached:
            details = usage.get("prompt_tokens_details") or {}
            cached = int(details.get("cached_tokens") or 0)
        total_in = int(usage.get("prompt_tokens") or 0)

        return ProviderResult(
            text=(choice.get("message") or {}).get("content") or "",
            # prompt_tokens включает кешированные — вычитаем, чтобы не считать дважды.
            input_tokens=max(0, total_in - cached),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cache_read_tokens=cached,
            stop_reason=str(choice.get("finish_reason") or ""),
            billed_usd=_billed_usd(usage.get("cost")),
        )

    async def _send(self, body: dict[str, Any], *, model: str) -> dict[str, Any]:
        return await self._post(body)

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


class FalProvider(OpenAICompatProvider):
    """Модели через fal: его OpenRouter-шлюз, тот же диалект /chat/completions.

    Ключ один на всё — картинки бот рисует через fal же. Модели — те же Gemini,
    что и напрямую, но без бесплатного тарифа: прямой ключ Gemini на сервере
    упирался в 20 запросов в сутки на flash, а ролик тратит 3-5 таких вызовов.

    Отличия от DeepSeek, замеренные на живом ключе:

    - схема принуждается (`json_schema`), так что в промпт её не кладём;
    - звук принимается (`input_audio`), корректор транскрипта его слушает;
    - стоимость приходит в ответе (`usage.cost`) — с наценкой шлюза, которой
      нет в нашей таблице цен;
    - размышления у flash-lite отключаются, а у flash — нет: «Reasoning is
      mandatory for this endpoint and cannot be disabled». Таким моделям шлём
      минимальное усилие и запоминаем их на время жизни провайдера.
    """

    supports_schema = True
    supports_audio = True
    auth_scheme = "Key"

    def __init__(
        self, api_key: str, base_url: str, timeout: float = 180.0, *, thinking: bool = False
    ) -> None:
        super().__init__(api_key, base_url, timeout, thinking=thinking)
        #: модели, которым нельзя отключить размышления
        self._must_reason: set[str] = set()

    def _tune(
        self, body: dict[str, Any], *, model: str, schema: dict[str, Any], effort: str | None
    ) -> None:
        # `strict` выключен: строгий режим требует от схемы того, чего в наших
        # схемах нет (все поля обязательные, `additionalProperties: false`
        # повсюду). Форму всё равно проверяет `jsonshape.check`.
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": False, "schema": schema},
        }
        if self.thinking and effort in ("low", "medium", "high"):
            body["reasoning"] = {"effort": effort}
        elif model in self._must_reason:
            body["reasoning"] = {"effort": "minimal"}
        else:
            body["reasoning"] = {"enabled": False}

    async def _send(self, body: dict[str, Any], *, model: str) -> dict[str, Any]:
        try:
            return await self._post(body)
        except ProviderError as e:
            if e.status != 400 or "reasoning is mandatory" not in str(e).lower():
                raise
        log.info("%s: размышления не отключаются — шлю минимальное усилие", model)
        self._must_reason.add(model)
        body["reasoning"] = {"effort": "minimal"}
        return await self._post(body)


# --------------------------------------------------------------------------- gemini


class GeminiProvider(Provider):
    name = "gemini"
    supports_schema = True
    supports_images = True
    supports_audio = True

    def __init__(
        self, api_key: str, base_url: str, timeout: float = 180.0, *, thinking: bool = False
    ) -> None:
        super().__init__(api_key, base_url, timeout)
        self._http: httpx.AsyncClient | None = None
        self.thinking = thinking
        #: модели, отвергшие thinkingConfig — второй раз им его не шлём
        self._no_thinking_config: set[str] = set()

    def _ensure(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            )
        return self._http

    def _thinking_config(self, effort: str | None) -> dict[str, Any] | None:
        """Чем ограничиваем размышления. None — не трогаем настройку вовсе."""
        if not self.thinking:
            return {"thinkingBudget": 0}
        if effort in ("low", "medium", "high"):
            return {"thinkingLevel": effort}
        return None

    async def _post(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        http = self._ensure()
        try:
            resp = await http.post(f"/models/{model}:generateContent", json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise _http_error(e) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"{type(e).__name__}: {e}", retryable=True) from e
        # Та же дыра, что у openai-совместимых: 2xx с телом не-JSON (дефект №3).
        return _json_object(resp)

    async def complete(
        self, *, model, system, user, schema, images=(), max_tokens=16_000, effort=None, audio=None
    ) -> ProviderResult:
        parts: list[dict[str, Any]] = [
            {
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": base64.standard_b64encode(img).decode("ascii"),
                }
            }
            for img in images
        ]
        if audio:
            data, mime = audio
            parts.append(
                {
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.standard_b64encode(data).decode("ascii"),
                    }
                }
            )
        parts.append({"text": user})

        body = {
            # systemInstruction неизменна между роликами — на неё и ложится
            # неявный кеш Gemini.
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
                "maxOutputTokens": max_tokens,
                "temperature": 0.4,
            },
        }
        # Тот же капкан, что у DeepSeek: размышления считаются в maxOutputTokens
        # и оплачиваются как выход. Замерено на живом ключе, gemini-3.7-flash,
        # один и тот же запрос: без настройки 72 токена мыслей, с бюджетом 0 —
        # 29. Полного нуля модели третьего поколения не дают, но думают по
        # минимуму, а это всё, что нам нужно.
        #
        # ⚠ Настройку принимают не все: gemini-3.6-flash отвечает на неё
        # «400 Request contains an invalid argument» и не работает совсем
        # (замерено: 0 успехов из 6). Список моделей зашивать нельзя — он
        # протухнет к следующему релизу Gemini, поэтому узнаём по факту отказа
        # и запоминаем на время жизни провайдера.
        cfg_thinking = self._thinking_config(effort)
        if cfg_thinking is not None and model not in self._no_thinking_config:
            body["generationConfig"]["thinkingConfig"] = cfg_thinking

        try:
            data = await self._post(model, body)
        except ProviderError as e:
            retry_without = (
                e.status == 400 and "thinkingConfig" in body["generationConfig"]
            )
            if not retry_without:
                raise
            log.warning("модель %s не приняла thinkingConfig — повтор без него", model)
            self._no_thinking_config.add(model)
            body["generationConfig"].pop("thinkingConfig")
            data = await self._post(model, body)

        candidates = data.get("candidates") or []
        text = ""
        finish = ""
        if candidates:
            finish = str(candidates[0].get("finishReason") or "")
            text = "".join(
                p.get("text", "") for p in (candidates[0].get("content") or {}).get("parts") or []
            )
        usage = data.get("usageMetadata") or {}
        cached = int(usage.get("cachedContentTokenCount") or 0)
        return ProviderResult(
            text=text,
            input_tokens=max(0, int(usage.get("promptTokenCount") or 0) - cached),
            # `candidatesTokenCount` не включает размышления, а платим мы и за
            # них по цене выхода. Без этого слагаемого себестоимость ролика в
            # отчёте §7 занижена — тем сильнее, чем больше модель думала.
            output_tokens=int(usage.get("candidatesTokenCount") or 0)
            + int(usage.get("thoughtsTokenCount") or 0),
            cache_read_tokens=cached,
            stop_reason=finish,
        )

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


# --------------------------------------------------------------------------- фабрика


PROVIDERS: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider,
    "deepseek": OpenAICompatProvider,
    "openai": OpenAICompatProvider,
    "gemini": GeminiProvider,
    "fal": FalProvider,
}


def make_provider(
    name: str, api_key: str, base_url: str, timeout: float = 180.0, *, thinking: bool = False
) -> Provider:
    try:
        cls = PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"неизвестный провайдер {name!r}; доступны: {', '.join(sorted(PROVIDERS))}"
        ) from None
    if cls in (OpenAICompatProvider, FalProvider, GeminiProvider):
        provider: Provider = cls(api_key, base_url, timeout, thinking=thinking)  # type: ignore[call-arg]
    else:
        provider = cls(api_key, base_url, timeout)
    provider.name = name
    return provider


def schema_hint(schema: dict[str, Any]) -> str:
    """Схема текстом — для провайдеров, которые не умеют её принуждать.

    Форму ответа всё равно проверяет `jsonshape.check`: промпт это просьба, а не
    гарантия.
    """
    return (
        "\n\n---\n"
        "Ответ — ОДИН объект JSON строго по этой схеме, без пояснений и без "
        "markdown-обёртки:\n\n```json\n"
        + json.dumps(schema, ensure_ascii=False, indent=1)
        + "\n```"
    )
