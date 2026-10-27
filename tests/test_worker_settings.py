"""Настройки пулов arq: всё важное объявлено в самом классе."""

from __future__ import annotations

import inspect

import pytest
from arq.worker import Worker, get_kwargs

from src.worker import tasks

POOLS = [tasks.WorkerSettings, tasks.LightWorkerSettings, tasks.GpuWorkerSettings]


@pytest.mark.parametrize("cls", POOLS, ids=lambda c: c.__name__)
def test_arq_sees_the_redis_of_this_project(cls):
    """arq берёт настройки из `__dict__`, наследование ему невидимо.

    Уборщик, унаследовавший `redis_settings` от воркера, на сервере подключился
    к Redis по умолчанию (localhost:6379) — а там был Redis соседнего проекта.
    """
    kwargs = get_kwargs(cls)
    assert kwargs.get("redis_settings") is tasks.WorkerSettings.redis_settings
    assert "on_startup" in kwargs and "on_shutdown" in kwargs
    assert set(kwargs) <= set(inspect.signature(Worker).parameters)


def test_the_janitor_does_not_warm_up_speech_and_browser():
    assert get_kwargs(tasks.LightWorkerSettings)["on_startup"] is tasks.light_startup
