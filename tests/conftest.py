"""Общие двойники для тестов LLM."""

from __future__ import annotations

import os
from typing import Any, Sequence

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.db import repo
from src.db.models import Base
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


# --------------------------------------------------- настоящий Postgres
#
# Фикстура общая, а не в одном файле: база нужна всему, что держится на
# транзакции и условии в WHERE, — деньгам, переходам статуса, памяти выбора
# пользователя. Такое двойником не изобразить.

SCHEMA = f"reelbot_test_{os.getpid()}"
DSN = os.environ.get(
    "TEST_POSTGRES_DSN", "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/postgres"
)
WHY = (
    f"нужен Postgres по {DSN} — локально поднимается docker-compose, в CI это служба "
    "postgres; адрес переопределяется переменной TEST_POSTGRES_DSN"
)
# В CI пропуск этих тестов недопустим: гейт выкладки из пропусков не сторожит
# ничего, а здесь на кону деньги. Переменную ставит сам workflow, и если базы
# там не окажется, выкладка упадёт, а не позеленеет.
REQUIRED = bool(os.environ.get("TEST_POSTGRES_REQUIRED"))


def _no_database(e: BaseException) -> None:
    (pytest.fail if REQUIRED else pytest.skip)(f"{WHY} ({type(e).__name__}: {e})")


@pytest.fixture
async def db():
    """Своя схема на каждый тест: ни чужих данных, ни следов между тестами.

    Движок создаётся ВНУТРИ теста, а не в session-фикстуре: асинхронный движок
    привязан к тому loop, в котором открыл соединение, и переезд между loop'ами
    даёт «another operation is in progress» вместо понятной ошибки.
    """
    admin = create_async_engine(DSN, poolclass=NullPool)
    try:
        async with admin.begin() as c:
            await c.execute(text(f"drop schema if exists {SCHEMA} cascade"))
            await c.execute(text(f"create schema {SCHEMA}"))
    except Exception as e:  # noqa: BLE001 — базы нет: пропуск локально, провал в CI
        await admin.dispose()
        _no_database(e)

    engine = create_async_engine(
        DSN, connect_args={"server_settings": {"search_path": SCHEMA}}
    )
    async with engine.begin() as c:
        await c.run_sync(Base.metadata.create_all)
    repo._engine = engine
    repo._sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    # Пробный ролик новичку здесь выключен: денежные тесты считают баланс от
    # нуля, и подарок при создании пользователя сдвинул бы каждое число. Сама
    # проба проверяется отдельно и включает её явно.
    trial_was = repo.default_settings.trial_videos
    repo.default_settings.trial_videos = 0
    try:
        yield
    finally:
        repo.default_settings.trial_videos = trial_was
        await engine.dispose()
        async with admin.begin() as c:
            await c.execute(text(f"drop schema if exists {SCHEMA} cascade"))
        await admin.dispose()
        repo._engine = None
        repo._sessionmaker = None


