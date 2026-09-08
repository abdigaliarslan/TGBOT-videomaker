"""Общие двойники для тестов LLM."""

from __future__ import annotations

from typing import Any, Sequence

from src.llm.providers import Provider, ProviderResult


class FakeProvider(Provider):
    """Провайдер-двойник: отдаёт заранее заготовленные ответы или кидает ошибки.

    Сеть не трогаем — проверяется наша обвязка, а не чужой API.
    """

    def __init__(
        self,
        script: Sequence[Any],
        *,
        supports_schema: bool = True,
        supports_images: bool = True,
        name: str = "fake",
    ) -> None:
        super().__init__(api_key="test", base_url="http://fake", timeout=5.0)
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.name = name
        self.supports_schema = supports_schema
        self.supports_images = supports_images
        self.closed = False

    async def complete(self, **kwargs: Any) -> ProviderResult:  # type: ignore[override]
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("двойник исчерпан: вызовов больше, чем заготовлено ответов")
        nxt = self.script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    async def aclose(self) -> None:
        self.closed = True


def result_of(payload: Any, **usage: Any) -> ProviderResult:
    import json

    return ProviderResult(
        text=json.dumps(payload, ensure_ascii=False) if not isinstance(payload, str) else payload,
        input_tokens=usage.get("input_tokens", 100),
        output_tokens=usage.get("output_tokens", 50),
        cache_read_tokens=usage.get("cache_read_tokens", 0),
        cache_creation_tokens=usage.get("cache_creation_tokens", 0),
        stop_reason=usage.get("stop_reason", "stop"),
    )
