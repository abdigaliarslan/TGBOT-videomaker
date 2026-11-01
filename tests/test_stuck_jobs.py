"""Сторож зависших задач: воркера убили — задача не висит вечно.

Воркер умирает не по своей воле: выкладка перезапускает службу, OOM-killer
снимает процесс на тяжёлом ролике. Запись в БД остаётся в статусе RUNNING
навсегда, и пользователь попадает в двойную ловушку: `active_jobs_count(user)
>= 1` больше никогда не пустит его прислать видео, а списанный с баланса ролик
не вернётся. Наблюдатель в чате при этом молча доживёт свои три часа на
последнем проценте полосы.

Живую задачу от мёртвой отличает пульс в Redis: воркер подновляет ключ, пока
работает. Нет пульса — нет воркера.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.db.models import JobStatus
from src.worker import tasks


@dataclass
class FakeRedis:
    """Та часть Redis, которой пользуется сторож: ключи со сроком жизни."""

    keys: dict[str, tuple[str, float | None]] = field(default_factory=dict)

    async def set(self, key, value, ex=None):
        self.keys[key] = (value, ex)

    async def get(self, key):
        got = self.keys.get(key)
        return None if got is None else got[0]

    async def exists(self, key):
        return 1 if key in self.keys else 0

    async def delete(self, *keys):
        for key in keys:
            self.keys.pop(key, None)


def job_of(status=JobStatus.RUNNING, *, minutes_ago=30):
    ago = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=42,
        status=status,
        started_at=ago if status == JobStatus.RUNNING else None,
        created_at=ago,
        charged=True,
    )


@pytest.fixture
def env(monkeypatch):
    """Подменяет БД вокруг сторожа: смысл теста — его решения, а не SQL."""
    seen = {"failed": [], "refunded": [], "cutoffs": None, "jobs": []}

    async def to_reap(running_before, queued_before):
        seen["cutoffs"] = (running_before, queued_before)
        return seen["jobs"]

    async def mark_failed(job_id, error, *, code="failed"):
        seen["failed"].append((job_id, code))

    async def refund(job_id):
        seen["refunded"].append(job_id)
        return True

    monkeypatch.setattr(tasks.repo, "jobs_to_reap", to_reap, raising=False)
    monkeypatch.setattr(tasks.repo, "mark_failed", mark_failed)
    monkeypatch.setattr(tasks.repo, "refund_job", refund)
    return seen


async def test_a_running_job_without_a_pulse_is_failed_and_refunded(env):
    """Главный случай: воркера убили на середине ролика."""
    job = job_of()
    env["jobs"] = [job]

    out = await tasks.reap_stuck_jobs({"redis": FakeRedis()})

    assert env["failed"] == [(job.id, "worker_lost")]
    assert env["refunded"] == [job.id]
    assert out == {"running": 1, "queued": 0}


async def test_a_running_job_with_a_live_pulse_is_left_alone(env):
    """Долгий ролик — не мёртвый ролик. Пульс есть — сторож проходит мимо."""
    job = job_of()
    env["jobs"] = [job]
    redis = FakeRedis()
    await redis.set(tasks.heartbeat_key(str(job.id)), "1", ex=tasks.JOB_HEARTBEAT_TTL)

    out = await tasks.reap_stuck_jobs({"redis": redis})

    assert env["failed"] == []
    assert env["refunded"] == []
    assert out == {"running": 0, "queued": 0}


async def test_a_job_younger_than_the_grace_is_never_even_asked_about(env):
    """Запас по времени: задача, помеченной RUNNING только что, ещё не успела
    поставить первый пульс. Отсечка уходит в запрос, а не проверяется после."""
    env["jobs"] = []
    await tasks.reap_stuck_jobs({"redis": FakeRedis()})

    running_before, _ = env["cutoffs"]
    age = datetime.now(timezone.utc) - running_before
    assert timedelta(seconds=tasks.STUCK_GRACE) - timedelta(minutes=1) < age
    assert age < timedelta(seconds=tasks.STUCK_GRACE) + timedelta(minutes=1)
    assert tasks.STUCK_GRACE > tasks.JOB_HEARTBEAT_TTL, (
        "запас меньше срока жизни пульса — сторож снимет живую задачу"
    )


async def test_the_user_is_told_why_the_reel_never_came(env):
    """Наблюдатель в чате читает прогресс из Redis, а не из БД. Без этой записи
    пользователь остался бы смотреть на полосу до конца своих трёх часов."""
    job = job_of()
    env["jobs"] = [job]
    redis = FakeRedis()

    await tasks.reap_stuck_jobs({"redis": redis})

    import json

    raw = await redis.get(tasks.progress_key(str(job.id)))
    assert raw is not None, "сторож не сказал наблюдателю, что задачи больше нет"
    data = json.loads(raw)
    assert data["stage"] == "failed"
    assert data["error_code"] == "worker_lost"


async def test_a_queued_job_that_nobody_ever_picked_up_is_released(env):
    """Redis потеряли (flush, переезд) — очередь пуста, а запись в БД ждёт.
    Отсечка для ожидания в очереди своя и длинная: за ней не авария, а простой."""
    job = job_of(JobStatus.QUEUED, minutes_ago=60 * 24)
    env["jobs"] = [job]

    out = await tasks.reap_stuck_jobs({"redis": FakeRedis()})

    assert env["failed"] == [(job.id, "worker_lost")]
    assert env["refunded"] == [job.id]
    assert out == {"running": 0, "queued": 1}
    _, queued_before = env["cutoffs"]
    age = datetime.now(timezone.utc) - queued_before
    assert age > timedelta(seconds=tasks.STUCK_GRACE), (
        "ожидание в очереди отмеряется тем же коротким запасом, что и смерть воркера"
    )


def test_the_reason_has_its_own_phrase_in_both_languages():
    """Код без фразы молча превращается в «Не получилось смонтировать» — а это
    другая история: здесь ролик не пробовали монтировать, его вернули."""
    from src.core.i18n import ERROR_CODES, error_text

    assert "worker_lost" in ERROR_CODES
    ru = error_text("worker_lost", "ru", min_s=5, max_m=10)
    en = error_text("worker_lost", "en", min_s=5, max_m=10)
    assert ru != error_text("failed", "ru", min_s=5, max_m=10)
    assert ru != en


async def test_one_unreapable_job_does_not_stop_the_round(env, monkeypatch):
    """Падение на одной задаче не имеет права оставить остальные висеть."""
    bad, good = job_of(), job_of()
    env["jobs"] = [bad, good]

    async def mark_failed(job_id, error, *, code="failed"):
        if job_id == bad.id:
            raise RuntimeError("база отвалилась")
        env["failed"].append((job_id, code))

    monkeypatch.setattr(tasks.repo, "mark_failed", mark_failed)

    out = await tasks.reap_stuck_jobs({"redis": FakeRedis()})

    assert env["failed"] == [(good.id, "worker_lost")]
    assert out == {"running": 1, "queued": 0}


async def test_the_watchdog_runs_on_a_schedule():
    """Сторож без расписания — мёртвый код. Живёт в пуле уборки: пул `cpu` в
    момент аварии занят или мёртв, а уборщику для этого ничего не нужно."""
    assert tasks.reap_stuck_jobs in tasks.LightWorkerSettings.functions
    scheduled = [c for c in tasks.LightWorkerSettings.cron_jobs if c.coroutine is tasks.reap_stuck_jobs]
    assert scheduled, "сторож не поставлен в расписание"


def test_the_query_asks_about_the_two_statuses_by_their_stored_spelling():
    """Единственная проверка SQL: в тестах базы нет, запрос едет мимо них.

    Статус хранится ИМЕНЕМ (`native_enum=False` у SQLAlchemy), то есть
    'RUNNING', а не 'running'. Ручной фильтр по строчным буквам уже давал пустой
    ответ на живой базе — молча, без ошибки. ORM эту разницу знает сама, и тест
    стоит на страже того, что фильтр остаётся по `JobStatus`, а не по строке.
    """
    import inspect

    from src.db import repo

    src = inspect.getsource(repo.jobs_to_reap)
    assert "JobStatus.RUNNING" in src and "JobStatus.QUEUED" in src
    assert '"running"' not in src and "'running'" not in src
    assert "started_at" in src and "created_at" in src


# --------------------------------------------------------------------- пульс


async def test_the_pulse_beats_while_the_reel_is_being_made(monkeypatch, tmp_path):
    """Пульс ставит сам прогон, и ставит ДО первого тяжёлого этапа: иначе
    сторож успеет снять задачу, пока распознаётся речь."""
    redis = FakeRedis()
    job = SimpleNamespace(
        id=uuid.uuid4(), user_id=42, preset="pulse", options=None,
        source_local_path=None, source_key=None,
    )
    beats: list[str | None] = []

    async def fake_process_video(*a, **k):
        beats.append(await redis.get(tasks.heartbeat_key(str(job.id))))
        raise tasks.JobError("хватит", "too_short")

    async def noop(*a, **k):
        return None

    async def fake_get_job(job_id):
        return job

    async def fake_fetch(j, dst):
        return dst

    monkeypatch.setattr(tasks, "settings", SimpleNamespace(
        work_dir=tmp_path, debug=False, llm_provider="fal",
    ))
    monkeypatch.setattr(tasks.repo, "get_job", fake_get_job)
    monkeypatch.setattr(tasks.repo, "mark_running", noop)
    monkeypatch.setattr(tasks.repo, "update_job", noop)
    monkeypatch.setattr(tasks.repo, "mark_failed", noop)
    monkeypatch.setattr(tasks.repo, "refund_job", noop)
    monkeypatch.setattr(tasks, "fetch_source", fake_fetch)
    monkeypatch.setattr(tasks, "process_video", fake_process_video)
    monkeypatch.setattr(tasks, "LLMClient", lambda **k: SimpleNamespace(ready=False))

    with pytest.raises(tasks.JobError):
        await tasks.process_job({"redis": redis}, str(job.id))

    assert beats == ["1"], "пульса не было, когда прогон уже шёл"
    assert tasks.heartbeat_key(str(job.id)) not in redis.keys, (
        "пульс остался после конца задачи — сторож будет считать её живой"
    )


async def test_the_pulse_is_renewed_before_it_expires(monkeypatch):
    """Ключ живёт дольше, чем пауза между удрами, — иначе задача будет мертветь
    и оживать между двумя подновлениями."""
    assert tasks.JOB_HEARTBEAT_EVERY < tasks.JOB_HEARTBEAT_TTL / 2

    redis = FakeRedis()
    monkeypatch.setattr(tasks, "JOB_HEARTBEAT_EVERY", 0.01)
    import asyncio

    async with tasks.heartbeat(redis, "job-1"):
        key = tasks.heartbeat_key("job-1")
        assert await redis.exists(key)
        await redis.delete(key)
        await asyncio.sleep(0.05)
        assert await redis.exists(key), "пульс поставили один раз и забыли"
    assert not await redis.exists(tasks.heartbeat_key("job-1"))
