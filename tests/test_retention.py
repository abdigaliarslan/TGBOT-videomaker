"""Удаление файлов по сроку хранения: что удаляется, что остаётся."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from src.worker import tasks


@dataclass
class FakeJob:
    id: uuid.UUID
    source_key: str | None
    result_key: str | None


class FakeStorage:
    def __init__(self, broken: set[str] = frozenset()):
        self.deleted: list[str] = []
        self.swept_before: datetime | None = None
        self.broken = broken

    async def delete(self, key):
        if key in self.broken:
            raise RuntimeError("диск")
        self.deleted.append(key)

    async def delete_older_than(self, cutoff):
        self.swept_before = cutoff
        return 3


@pytest.fixture
def env(monkeypatch):
    jobs = [FakeJob(uuid.uuid4(), "sources/a.mp4", "results/a.mp4"), FakeJob(uuid.uuid4(), None, "results/b.mp4")]
    seen = {"cutoff": None, "purged": []}

    async def expired(cutoff, limit=500):
        seen["cutoff"] = cutoff
        return jobs

    async def purged(job_id):
        seen["purged"].append(job_id)

    monkeypatch.setattr(tasks.repo, "jobs_with_expired_files", expired)
    monkeypatch.setattr(tasks.repo, "mark_files_purged", purged)
    monkeypatch.setattr(tasks.settings, "storage_retention_days", 7)
    return jobs, seen


async def test_expired_jobs_lose_their_files_and_get_marked(env, monkeypatch):
    jobs, seen = env
    fake = FakeStorage()
    monkeypatch.setattr(tasks, "storage", fake)
    out = await tasks.purge_expired_files({})

    assert sorted(fake.deleted) == ["results/a.mp4", "results/b.mp4", "sources/a.mp4"]
    assert seen["purged"] == [j.id for j in jobs]
    assert out == {"jobs": 2, "files": 3, "swept": 3}
    age = datetime.now(timezone.utc) - seen["cutoff"]
    assert timedelta(days=7) - timedelta(minutes=1) < age < timedelta(days=7) + timedelta(minutes=1)


async def test_the_storage_sweep_trails_the_database_by_a_day(env, monkeypatch):
    """Проход по хранилищу не имеет права обогнать учёт в базе и стереть файл
    задачи, которую ещё не пометили удалённой."""
    _, seen = env
    fake = FakeStorage()
    monkeypatch.setattr(tasks, "storage", fake)
    await tasks.purge_expired_files({})
    assert seen["cutoff"] - fake.swept_before == timedelta(seconds=tasks.STORAGE_SWEEP_GRACE)


async def test_one_broken_file_does_not_stop_the_cleanup(env, monkeypatch):
    jobs, seen = env
    monkeypatch.setattr(tasks, "storage", FakeStorage(broken={"sources/a.mp4"}))
    await tasks.purge_expired_files({})
    assert seen["purged"] == [j.id for j in jobs]


async def test_zero_days_means_keep_forever(env, monkeypatch):
    _, seen = env
    monkeypatch.setattr(tasks.settings, "storage_retention_days", 0)
    fake = FakeStorage()
    monkeypatch.setattr(tasks, "storage", fake)
    assert await tasks.purge_expired_files({}) == {"jobs": 0, "files": 0, "swept": 0}
    assert fake.deleted == [] and seen["cutoff"] is None


def test_the_janitor_runs_the_purge():
    names = {c.name for c in tasks.LightWorkerSettings.cron_jobs}
    assert "cron:purge_expired_files" in names
    assert tasks.purge_expired_files in tasks.LightWorkerSettings.functions
