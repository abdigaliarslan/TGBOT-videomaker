"""Деньги: списание, возврат и зачисление оплаты — на настоящем Postgres.

Это единственные функции в проекте, ошибка в которых стоит денег, и до сих
пор их не проверял никто: тесты базы не поднимают, а подменённое репо
проверяет только вызывающего. Между тем все три держатся ровно на том, чего
двойником не изобразить, — на транзакции и на условии в WHERE:

* `create_job` списывает ролик и создаёт задачу одной транзакцией, а проверку
  баланса делает одним UPDATE с условием. Двойника тут нет: смысл в том, что
  две одновременные загрузки не уведут баланс в минус.
* `refund_job` возвращает ровно один раз, снимая флаг в той же транзакции.
  Повторный вызов — не теория: возврат зовут и падение прогона, и сторож
  зависших задач.
* `credit_payment` опирается на `ON CONFLICT DO NOTHING` по id платежа в
  Telegram: повтор вебхука не имеет права зачислить пакет дважды.

База своя: отдельная СХЕМА в локальном Postgres, чтобы ничего чужого не
задеть. Нет базы — тесты пропускаются с объяснением.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from sqlalchemy import text

from src.db import repo
from src.db.models import Base, JobStatus

@pytest.fixture
async def user(db):
    """Пользователь с балансом на три ролика."""
    u = await repo.upsert_user(1001, username="t", first_name="Т", language_code="ru")
    async with repo.session() as s:
        await s.execute(text("update users set balance = 3 where id = 1001"))
    return u


def job_args(**over):
    args = dict(
        source_file_id=None, source_key=None, source_size=10,
        preset="pulse", chat_id=1001, options=None, charge=True,
    )
    args.update(over)
    return args


async def balance_of(user_id: int = 1001) -> int:
    got = await repo.get_user(user_id)
    assert got is not None
    return got.balance


# ------------------------------------------------------------------- списание


async def test_a_job_costs_exactly_one_reel(user):
    job = await repo.create_job(1001, **job_args())
    assert await balance_of() == 2
    assert job.charged is True
    assert job.status == JobStatus.QUEUED


async def test_an_owner_pays_nothing(user):
    job = await repo.create_job(1001, **job_args(charge=False))
    assert await balance_of() == 3
    assert job.charged is False


async def test_an_empty_balance_leaves_no_job_behind(user):
    """Отказ по балансу не оставляет ни задачи, ни минуса."""
    for _ in range(3):
        await repo.create_job(1001, **job_args())
    assert await balance_of() == 0

    with pytest.raises(repo.InsufficientBalance):
        await repo.create_job(1001, **job_args())

    assert await balance_of() == 0
    assert len(await repo.user_jobs(1001)) == 3


async def test_a_reel_is_not_charged_for_a_job_that_was_never_created(user):
    """Списание и создание задачи — одна транзакция, и это проверяется поломкой
    ПОСЛЕ списания.

    Без общей транзакции обрыв здесь оставлял бы списанный ролик без задачи:
    денег нет, монтажа нет, и вернуть нечего — возврат привязан к задаче.
    Поломку даёт сама база: имя темы длиннее своей колонки.
    """
    from sqlalchemy.exc import DBAPIError

    with pytest.raises(DBAPIError):
        await repo.create_job(1001, **job_args(preset="т" * 100))

    assert await balance_of() == 3, "ролик списан за задачу, которой нет"
    assert await repo.user_jobs(1001) == []


async def test_two_uploads_at_once_cannot_overdraw_the_balance(user):
    """Один ролик на балансе, две загрузки одновременно: проходит ровно одна.

    Проверка баланса и списание — один UPDATE с условием; разнеси их на SELECT и
    UPDATE, и этот тест уйдёт в минус.
    """
    async with repo.session() as s:
        await s.execute(text("update users set balance = 1 where id = 1001"))

    got = await asyncio.gather(
        repo.create_job(1001, **job_args()),
        repo.create_job(1001, **job_args()),
        return_exceptions=True,
    )
    ok = [g for g in got if not isinstance(g, BaseException)]
    refused = [g for g in got if isinstance(g, repo.InsufficientBalance)]
    assert len(ok) == 1, f"списали больше, чем было: {got}"
    assert len(refused) == 1, f"ожидал один отказ по балансу, получил {got}"
    assert await balance_of() == 0


# -------------------------------------------------------------------- возврат


async def test_a_refund_happens_exactly_once(user):
    """Возврат зовут и падение прогона, и сторож зависших задач — иногда оба."""
    job = await repo.create_job(1001, **job_args())
    assert await balance_of() == 2

    assert await repo.refund_job(job.id) is True
    assert await balance_of() == 3
    assert await repo.refund_job(job.id) is False, "второй возврат прошёл"
    assert await balance_of() == 3


async def test_an_uncharged_job_gives_nothing_back(user):
    job = await repo.create_job(1001, **job_args(charge=False))
    assert await repo.refund_job(job.id) is False
    assert await balance_of() == 3


async def test_two_refunds_at_once_return_one_reel(user):
    """Прогон упал и сторож проснулся в одну секунду."""
    job = await repo.create_job(1001, **job_args())
    got = await asyncio.gather(repo.refund_job(job.id), repo.refund_job(job.id))
    assert sorted(got) == [False, True], f"два возврата за один ролик: {got}"
    assert await balance_of() == 3


# -------------------------------------------------------------------- оплата


async def test_a_payment_is_credited_once_per_charge_id(user):
    args = dict(charge_id="tg-charge-1", stars=250, videos=5, payload="pack:5")
    assert await repo.credit_payment(1001, **args) is True
    assert await balance_of() == 8
    assert await repo.credit_payment(1001, **args) is False, "повтор вебхука зачислил второй раз"
    assert await balance_of() == 8


async def test_two_packages_add_up(user):
    await repo.credit_payment(1001, charge_id="a", stars=250, videos=5, payload="p")
    await repo.credit_payment(1001, charge_id="b", stars=100, videos=2, payload="p")
    assert await balance_of() == 10


async def test_the_same_charge_arriving_twice_at_once_is_credited_once(user):
    args = dict(charge_id="race", stars=250, videos=5, payload="p")
    got = await asyncio.gather(
        repo.credit_payment(1001, **args), repo.credit_payment(1001, **args)
    )
    assert sorted(got) == [False, True], f"одна оплата зачислена дважды: {got}"
    assert await balance_of() == 8


# ----------------------------------------------- переходы статуса и сторож


async def test_a_job_is_taken_into_work_only_from_the_queue(user):
    job = await repo.create_job(1001, **job_args())
    assert await repo.mark_running(job.id) is True
    assert await repo.mark_running(job.id) is False, "взяли в работу задачу, уже бывшую в работе"


async def test_a_job_cut_short_by_a_restart_is_resumed_but_not_a_reaped_one(user):
    """Повтор после выкладки подхватывает задачу, оставшуюся RUNNING, но не ту,
    которую сторож уже снял и вернул: та ушла бы пользователю бесплатно."""
    job = await repo.create_job(1001, **job_args())
    await repo.mark_running(job.id)
    assert await repo.mark_running(job.id, resume=True) is True

    await repo.fail_if_still(job.id, "воркер умер", code="worker_lost", expect=JobStatus.RUNNING)
    assert await repo.mark_running(job.id, resume=True) is False


async def test_the_verdict_only_lands_on_the_status_it_expected(user):
    job = await repo.create_job(1001, **job_args())
    await repo.mark_running(job.id)

    assert await repo.fail_if_still(job.id, "пробник", expect=JobStatus.QUEUED) is False
    assert (await repo.get_job(job.id)).status == JobStatus.RUNNING

    assert await repo.fail_if_still(
        job.id, "воркер умер", code="worker_lost", expect=JobStatus.RUNNING
    ) is True
    fresh = await repo.get_job(job.id)
    assert fresh.status == JobStatus.FAILED
    assert fresh.error_code == "worker_lost"


async def test_the_watchdog_picks_the_stuck_and_leaves_the_fresh(user):
    """Та самая выборка, которую подменяют тесты сторожа."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    stuck = await repo.create_job(1001, **job_args())
    fresh = await repo.create_job(1001, **job_args())
    waiting = await repo.create_job(1001, **job_args(charge=False))
    await repo.mark_running(stuck.id)
    await repo.mark_running(fresh.id)
    async with repo.session() as s:
        await s.execute(text("update jobs set started_at = :t where id = :i"),
                        {"t": now - timedelta(hours=2), "i": stuck.id})
        await s.execute(text("update jobs set created_at = :t where id = :i"),
                        {"t": now - timedelta(hours=9), "i": waiting.id})

    got = await repo.jobs_to_reap(
        running_before=now - timedelta(minutes=5), queued_before=now - timedelta(hours=6)
    )
    ids = {j.id for j in got}
    assert stuck.id in ids, "задача без признаков жизни не попала в выборку"
    assert waiting.id in ids, "задача, простоявшая в очереди девять часов, не попала в выборку"
    assert fresh.id not in ids, "в выборку попала задача, которая только началась"
