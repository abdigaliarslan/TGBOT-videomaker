"""Доступ к БД. Сессия на операцию, никаких долгоживущих сессий в воркере."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from sqlalchemy import and_, cast, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import Settings, settings as default_settings
from src.db.models import Base, Job, JobStatus, LLMCall, Payment, User
from src.llm.client import Usage

log = logging.getLogger(__name__)

_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_engine(cfg: Settings | None = None):
    global _engine, _sessionmaker
    cfg = cfg or default_settings
    if _engine is None:
        _engine = create_async_engine(
            cfg.postgres_dsn,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None


@asynccontextmanager
async def session() -> AsyncIterator[AsyncSession]:
    if _sessionmaker is None:
        init_engine()
    assert _sessionmaker is not None
    async with _sessionmaker() as s:
        try:
            yield s
            await s.commit()
        except Exception:
            await s.rollback()
            raise


async def create_all() -> None:
    """Только для тестов и первого локального запуска. В проде — alembic."""
    engine = init_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ------------------------------------------------------------------ пользователи


def upsert_user_stmt(
    user_id: int,
    *,
    username: str | None,
    first_name: str | None,
    language_code: str | None,
    is_admin: bool,
    now: datetime,
    preset: str | None = None,
):
    """INSERT … ON CONFLICT для пользователя. Вынесен, чтобы проверяться без БД.

    `preset` стоит в values и НЕ стоит в `set_`, и это два разных требования.

    В values — потому что иначе новый пользователь получал тему из умолчания
    колонки, а не из `DEFAULT_PRESET` (дефект №5): настройку меняли, а первый
    ролик человек всё равно видел в glass.

    Не в `set_` — потому что `set_` выполняется на КАЖДОМ сообщении существующего
    пользователя, и тема из конфига затирала бы тему, выбранную руками.
    """
    preset = preset or default_settings.default_preset
    return (
        pg_insert(User)
        .values(
            id=user_id,
            username=username,
            first_name=first_name,
            language_code=language_code,
            is_admin=is_admin,
            preset=preset,
            created_at=now,
            last_seen_at=now,
        )
        .on_conflict_do_update(
            index_elements=[User.id],
            set_={
                "username": username,
                "first_name": first_name,
                "language_code": language_code,
                "is_admin": is_admin,
                "last_seen_at": now,
            },
        )
        .returning(User)
    )


async def upsert_user(
    user_id: int,
    *,
    username: str | None = None,
    first_name: str | None = None,
    language_code: str | None = None,
    is_admin: bool = False,
) -> User:
    async with session() as s:
        stmt = upsert_user_stmt(
            user_id,
            username=username,
            first_name=first_name,
            language_code=language_code,
            is_admin=is_admin,
            now=datetime.now(timezone.utc),
        )
        return (await s.execute(stmt)).scalar_one()


async def get_user(user_id: int) -> User | None:
    async with session() as s:
        return await s.get(User, user_id)


async def set_user_lang(user_id: int, lang: str) -> None:
    async with session() as s:
        await s.execute(update(User).where(User.id == user_id).values(lang=lang))


async def user_ids_with_chosen_lang() -> list[tuple[int, str]]:
    """Пользователи, выбиравшие язык руками, — только у них в чате бывает своя кнопка меню."""
    async with session() as s:
        q = select(User.id, User.lang).where(User.lang.isnot(None))
        return [(row.id, row.lang) for row in (await s.execute(q)).all()]


async def set_user_preset(user_id: int, preset: str) -> None:
    async with session() as s:
        await s.execute(update(User).where(User.id == user_id).values(preset=preset))


async def set_user_settings(user_id: int, **values: Any) -> None:
    """Запоминает выбор пользователя между роликами: цвет выделения, шрифт.

    Слияние делает БАЗА (`settings || :patch`), а не Python. Прочитать-слить-
    записать значило бы потерять запись, если два ролика ставятся одновременно:
    из приложения и из чата. Один UPDATE такой гонки не имеет.

    Тема сюда не кладётся — у неё своя колонка `users.preset` с историей.
    """
    if not values:
        return
    async with session() as s:
        await s.execute(
            update(User)
            .where(User.id == user_id)
            .values(settings=func.coalesce(User.settings, text("'{}'::jsonb")).op("||")(
                cast(values, JSONB)
            ))
        )


async def bump_videos_done(user_id: int) -> None:
    async with session() as s:
        await s.execute(
            update(User).where(User.id == user_id).values(videos_done=User.videos_done + 1)
        )


# ------------------------------------------------------------------ задачи


class InsufficientBalance(Exception):
    """На балансе нет ни одного ролика — задачу не создаём."""


async def create_job(
    user_id: int,
    *,
    source_file_id: str | None,
    source_key: str | None,
    source_size: int | None,
    preset: str,
    chat_id: int | None = None,
    options: dict[str, Any] | None = None,
    charge: bool = False,
) -> Job:
    """Создаёт задачу. С `charge=True` — списывает ролик в ТОЙ ЖЕ транзакции.

    Списание и задача — одна транзакция намеренно: по отдельности обрыв между
    ними оставлял бы либо списанный ролик без задачи, либо задачу без оплаты.
    Проверка баланса и списание — один UPDATE с условием, поэтому две
    одновременные загрузки не уведут баланс в минус.
    """
    async with session() as s:
        if charge:
            left = (
                await s.execute(
                    update(User)
                    .where(User.id == user_id, User.balance >= 1)
                    .values(balance=User.balance - 1)
                    .returning(User.balance)
                )
            ).scalar_one_or_none()
            if left is None:
                raise InsufficientBalance(user_id)
        job = Job(
            user_id=user_id,
            source_file_id=source_file_id,
            source_key=source_key,
            source_size=source_size,
            preset=preset,
            chat_id=chat_id,
            status=JobStatus.QUEUED,
            options=options,
            charged=charge,
        )
        s.add(job)
        await s.flush()
        await s.refresh(job)
        return job


async def get_job(job_id: uuid.UUID | str) -> Job | None:
    async with session() as s:
        return await s.get(Job, uuid.UUID(str(job_id)))


async def update_job(job_id: uuid.UUID | str, **values: Any) -> None:
    async with session() as s:
        await s.execute(update(Job).where(Job.id == uuid.UUID(str(job_id))).values(**values))


async def mark_running(job_id: uuid.UUID | str) -> bool:
    """Берёт задачу в работу. False — её там быть не должно.

    Переход разрешён только из очереди. arq возвращает задачу в очередь, если
    воркер умер, не доработав: ключ `in-progress` живёт `job_timeout`, а после
    него задача снова на виду. Если сторож её к тому времени освободил и вернул
    ролик на баланс, второй прогон отдал бы видео бесплатно. Условие в WHERE —
    единственное место, где это можно отсечь без гонки.
    """
    async with session() as s:
        got = (
            await s.execute(
                update(Job)
                .where(Job.id == uuid.UUID(str(job_id)), Job.status == JobStatus.QUEUED)
                .values(
                    status=JobStatus.RUNNING,
                    started_at=datetime.now(timezone.utc),
                    progress=0.0,
                )
                .returning(Job.id)
            )
        ).scalar_one_or_none()
        return got is not None


async def mark_failed(job_id: uuid.UUID | str, error: str, *, code: str = "failed") -> None:
    await update_job(
        job_id,
        status=JobStatus.FAILED,
        error=error[:4000],
        error_code=code[:32],
        finished_at=datetime.now(timezone.utc),
    )


async def fail_if_still(
    job_id: uuid.UUID | str, error: str, *, code: str = "failed", expect: JobStatus
) -> bool:
    """Помечает задачу упавшей, но только если её статус всё ещё `expect`.

    Нужно сторожу (`worker.tasks.reap_stuck_jobs`): между выборкой кандидатов и
    приговором задача успевает доделаться — пульс как раз и снимается в конце
    прогона. Без условия в WHERE сторож переписал бы DONE на FAILED и вернул
    ролик на баланс за видео, которое пользователь уже получил.

    Обратную сторону гонки (сторож успел раньше, чем `mark_done`) условием не
    закрыть, и закрывать её намеренно не стали: там DONE ложится поверх FAILED,
    то есть пользователь получает и ролик, и возврат. Запрет писать DONE поверх
    приговора стоил бы дороже — готовый ролик не дошёл бы до него вовсе.
    """
    async with session() as s:
        got = (
            await s.execute(
                update(Job)
                .where(Job.id == uuid.UUID(str(job_id)), Job.status == expect)
                .values(
                    status=JobStatus.FAILED,
                    error=error[:4000],
                    error_code=code[:32],
                    finished_at=datetime.now(timezone.utc),
                )
                .returning(Job.id)
            )
        ).scalar_one_or_none()
        return got is not None


async def mark_done(job_id: uuid.UUID | str, **values: Any) -> None:
    await update_job(
        job_id,
        status=JobStatus.DONE,
        progress=1.0,
        finished_at=datetime.now(timezone.utc),
        **values,
    )


async def active_jobs_count(user_id: int | None = None) -> int:
    async with session() as s:
        q = select(func.count(Job.id)).where(
            Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])
        )
        if user_id is not None:
            q = q.where(Job.user_id == user_id)
        return int((await s.execute(q)).scalar_one())


async def jobs_in_flight() -> list[Job]:
    """Задачи, за которыми кто-то ждёт ответа.

    Нужны при старте бота: наблюдатели живут в памяти процесса, и после
    рестарта пользователь иначе никогда не дождётся своего ролика.

    Сюда же попадают УЖЕ ГОТОВЫЕ, но не отданные. Если монтаж доделался, пока
    бот лежал, задача больше не «в работе» — и без этой ветки файл так и
    остался бы лежать в хранилище. Проверено на живом ролике.
    """
    async with session() as s:
        q = (
            select(Job)
            .where(
                Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.DONE]),
                Job.delivered.is_(False),
                Job.chat_id.isnot(None),
                Job.status_message_id.isnot(None),
            )
            .order_by(Job.created_at)
        )
        return list((await s.execute(q)).scalars())


async def jobs_to_reap(running_before: datetime, queued_before: datetime) -> list[Job]:
    """Задачи, за которыми, судя по времени, уже некому следить.

    Две разные беды одним запросом, потому что и лечатся они одинаково:

    * RUNNING с давним `started_at` — воркер взял задачу и не отчитался. Жив он
      или нет, по базе не видно: это решает пульс в Redis, и отсечка здесь
      только отбирает кандидатов, которым пульс вообще пора было поставить.
    * QUEUED с давним `created_at` — задачу никто не взял. Пульса у неё нет и
      быть не может, поэтому отсечка здесь — единственная проверка, и она
      длинная: за ней не авария, а честный простой в очереди.
    """
    async with session() as s:
        q = (
            select(Job)
            .where(
                or_(
                    and_(
                        Job.status == JobStatus.RUNNING,
                        func.coalesce(Job.started_at, Job.created_at) < running_before,
                    ),
                    and_(Job.status == JobStatus.QUEUED, Job.created_at < queued_before),
                )
            )
            .order_by(Job.created_at)
        )
        return list((await s.execute(q)).scalars())


async def mark_delivered(job_id: uuid.UUID | str) -> None:
    """Ролик ушёл пользователю — второй раз слать не нужно."""
    await update_job(job_id, delivered=True)


async def refund_job(job_id: uuid.UUID | str) -> bool:
    """Возвращает списанный за задачу ролик. Ровно один раз; без списания — ничего.

    Флаг снимается и баланс растёт в одной транзакции: повторный вызов (воркер
    упал и задачу разбирают снова) находит `charged = false` и ничего не делает.
    """
    async with session() as s:
        owner = (
            await s.execute(
                update(Job)
                .where(Job.id == uuid.UUID(str(job_id)), Job.charged.is_(True))
                .values(charged=False)
                .returning(Job.user_id)
            )
        ).scalar_one_or_none()
        if owner is None:
            return False
        await s.execute(update(User).where(User.id == owner).values(balance=User.balance + 1))
        return True


async def credit_payment(
    user_id: int,
    *,
    charge_id: str,
    stars: int,
    videos: int,
    payload: str,
    referral_percent: int | None = None,
) -> bool:
    """Зачисляет оплату. False — этот платёж уже зачислен (повторный апдейт).

    Бонус пригласившему — в той же транзакции и под той же защитой от повтора:
    повторный апдейт об оплате не зачисляет ни пакет, ни бонус. Бонус копится в
    сотых ролика (`users.ref_credit`), а на баланс переходят целые ролики —
    оба поля считаются одним UPDATE из старого значения копилки, так что
    одновременные оплаты двух приглашённых не теряют долю друг друга.
    """
    from src.core.referral import share_of

    percent = default_settings.referral_percent if referral_percent is None else referral_percent
    async with session() as s:
        referrer = (
            await s.execute(select(User.referrer_id).where(User.id == user_id))
        ).scalar_one_or_none()
        bonus = share_of(videos, percent) if referrer is not None else 0
        inserted = (
            await s.execute(
                pg_insert(Payment)
                .values(
                    user_id=user_id,
                    telegram_charge_id=charge_id,
                    stars=stars,
                    videos=videos,
                    payload=payload[:128],
                    ref_bonus=bonus,
                    created_at=datetime.now(timezone.utc),
                )
                .on_conflict_do_nothing(index_elements=[Payment.telegram_charge_id])
                .returning(Payment.id)
            )
        ).scalar_one_or_none()
        if inserted is None:
            return False
        await s.execute(update(User).where(User.id == user_id).values(balance=User.balance + videos))
        if bonus:
            pot = User.ref_credit + bonus
            await s.execute(
                update(User)
                .where(User.id == referrer)
                .values(balance=User.balance + pot // 100, ref_credit=pot % 100)
            )
        return True


# ------------------------------------------------------------------ приглашения


async def attach_referrer(user_id: int, referrer_id: int) -> bool:
    """Записывает, кто пригласил. True — записано сейчас.

    Пригласившим становятся один раз и только до первой покупки: иначе
    человек, давно платящий сам, мог бы отдать бонусы со своих покупок любой
    ссылке, по которой случайно нажал /start. Себя пригласить нельзя, и
    пригласить того, кто пригласил тебя, — тоже: кольцо из двух аккаунтов
    гоняло бы бонусы по кругу. Всё это — условия одного UPDATE, без окна между
    проверкой и записью.
    """
    if user_id == referrer_id:
        return False
    async with session() as s:
        paid = select(Payment.id).where(Payment.user_id == user_id).exists()
        referrer_ok = (
            select(User.id)
            .where(
                User.id == referrer_id,
                or_(User.referrer_id.is_(None), User.referrer_id != user_id),
            )
            .exists()
        )
        done = (
            await s.execute(
                update(User)
                .where(User.id == user_id, User.referrer_id.is_(None), ~paid, referrer_ok)
                .values(referrer_id=referrer_id, referred_at=datetime.now(timezone.utc))
                .returning(User.id)
            )
        ).scalar_one_or_none()
        return done is not None


async def referral_stats(user_id: int, *, since: datetime) -> dict[str, int]:
    """Сводка для экрана «Приглашения»: всего и с `since` (начало суток).

    `paid` — сколько приглашённых купили хоть раз; `earned` — сколько сотых
    ролика пришло пригласившему. «Сегодня» у оплат — по дате оплаты, а не по
    дате прихода друга.
    """
    async with session() as s:
        friends = select(User.id).where(User.referrer_id == user_id).scalar_subquery()
        joined = (
            await s.execute(
                select(
                    func.count(User.id),
                    func.count(User.id).filter(User.referred_at >= since),
                ).where(User.referrer_id == user_id)
            )
        ).one()
        paid = (
            await s.execute(
                select(
                    func.count(func.distinct(Payment.user_id)),
                    func.count(func.distinct(Payment.user_id)).filter(Payment.created_at >= since),
                    func.coalesce(func.sum(Payment.ref_bonus), 0),
                    func.coalesce(func.sum(Payment.ref_bonus).filter(Payment.created_at >= since), 0),
                ).where(Payment.user_id.in_(friends))
            )
        ).one()
    return {
        "invited": int(joined[0]),
        "invited_today": int(joined[1]),
        "paid": int(paid[0]),
        "paid_today": int(paid[1]),
        "earned": int(paid[2]),
        "earned_today": int(paid[3]),
    }


async def referral_friends(user_id: int, *, since: datetime | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Приглашённые, свежие сверху: имя, когда пришёл, сколько принёс."""
    async with session() as s:
        bonus = (
            select(Payment.user_id, func.sum(Payment.ref_bonus).label("earned"), func.count(Payment.id).label("payments"))
            .group_by(Payment.user_id)
            .subquery()
        )
        q = (
            select(User.first_name, User.referred_at, bonus.c.earned, bonus.c.payments)
            .outerjoin(bonus, bonus.c.user_id == User.id)
            .where(User.referrer_id == user_id)
            .order_by(User.referred_at.desc().nulls_last())
            .limit(limit)
        )
        if since is not None:
            q = q.where(User.referred_at >= since)
        rows = (await s.execute(q)).all()
    return [
        {
            "name": r.first_name,
            "joined_at": r.referred_at,
            "earned": int(r.earned or 0),
            "paid": bool(r.payments),
        }
        for r in rows
    ]


async def user_payments(user_id: int, limit: int = 50) -> list[Payment]:
    """Оплаты пользователя, свежие сверху — для истории в «Тарифах»."""
    async with session() as s:
        q = (
            select(Payment)
            .where(Payment.user_id == user_id)
            .order_by(Payment.created_at.desc())
            .limit(limit)
        )
        return list((await s.execute(q)).scalars())


async def referral_bonuses(user_id: int, limit: int = 50) -> list[tuple[datetime, int]]:
    """Бонусы, пришедшие пользователю с оплат приглашённых: когда и сколько сотых."""
    async with session() as s:
        friends = select(User.id).where(User.referrer_id == user_id).scalar_subquery()
        q = (
            select(Payment.created_at, Payment.ref_bonus)
            .where(Payment.user_id.in_(friends), Payment.ref_bonus > 0)
            .order_by(Payment.created_at.desc())
            .limit(limit)
        )
        return [(r[0], int(r[1])) for r in (await s.execute(q)).all()]


async def payment_bonus(charge_id: str) -> tuple[int, int] | None:
    """(пригласивший, сотые ролика) по оплате — чтобы бот сказал ему спасибо."""
    async with session() as s:
        row = (
            await s.execute(
                select(User.referrer_id, Payment.ref_bonus)
                .join(User, User.id == Payment.user_id)
                .where(Payment.telegram_charge_id == charge_id)
            )
        ).one_or_none()
    if row is None or row[0] is None or not row[1]:
        return None
    return int(row[0]), int(row[1])


async def jobs_with_expired_files(cutoff: datetime, limit: int = 500) -> list[Job]:
    """Законченные задачи старше `cutoff`, у которых файлы ещё не удалены.

    Только законченные: у задачи в работе исходник нужен воркеру прямо сейчас.
    Возраст — от окончания, а у задач, упавших до старта, — от создания.
    """
    async with session() as s:
        q = (
            select(Job)
            .where(
                Job.status.in_([JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED]),
                Job.files_purged_at.is_(None),
                func.coalesce(Job.finished_at, Job.created_at) < cutoff,
            )
            .order_by(Job.created_at)
            .limit(limit)
        )
        return list((await s.execute(q)).scalars())


async def mark_files_purged(job_id: uuid.UUID | str) -> None:
    await update_job(
        job_id,
        source_key=None,
        result_key=None,
        source_local_path=None,
        files_purged_at=datetime.now(timezone.utc),
    )


async def user_jobs(user_id: int, limit: int = 30) -> list[Job]:
    """Все задачи пользователя, свежие сверху — для экрана «Ролики»."""
    async with session() as s:
        q = (
            select(Job)
            .where(Job.user_id == user_id)
            .order_by(Job.created_at.desc())
            .limit(limit)
        )
        return list((await s.execute(q)).scalars())


async def recent_jobs(user_id: int, limit: int = 5) -> list[Job]:
    async with session() as s:
        q = (
            select(Job)
            .where(Job.user_id == user_id, Job.status == JobStatus.DONE)
            .order_by(Job.created_at.desc())
            .limit(limit)
        )
        return list((await s.execute(q)).scalars())


# ------------------------------------------------------------------ учёт токенов


async def record_llm_call(usage: Usage, job_id: uuid.UUID | str | None = None) -> None:
    async with session() as s:
        s.add(
            LLMCall(
                job_id=uuid.UUID(str(job_id)) if job_id else None,
                prompt=usage.prompt,
                tier=usage.tier,
                model=usage.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_creation_tokens=usage.cache_creation_tokens,
                cost_usd=usage.cost_usd,
                seconds=usage.seconds,
                attempt=usage.attempt,
            )
        )


async def cost_report(days: int = 7) -> list[dict[str, Any]]:
    """Себестоимость по типам вызова — то, что надо мерить с первого дня (§7)."""
    async with session() as s:
        q = (
            select(
                LLMCall.prompt,
                LLMCall.model,
                func.count(LLMCall.id).label("calls"),
                func.sum(LLMCall.input_tokens).label("input_tokens"),
                func.sum(LLMCall.cache_read_tokens).label("cache_read_tokens"),
                func.sum(LLMCall.output_tokens).label("output_tokens"),
                func.sum(LLMCall.cost_usd).label("cost_usd"),
            )
            .where(LLMCall.created_at > func.now() - func.make_interval(0, 0, 0, days))
            .group_by(LLMCall.prompt, LLMCall.model)
            .order_by(func.sum(LLMCall.cost_usd).desc())
        )
        return [dict(row._mapping) for row in (await s.execute(q)).all()]
