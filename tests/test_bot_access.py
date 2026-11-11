"""Доступ в хендлерах бота: чужой джоб и заблокированный пользователь."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.bot.handlers import video

pytestmark = pytest.mark.anyio


class FakeCallback:
    def __init__(self, data: str, user_id: int) -> None:
        self.data = data
        self.from_user = SimpleNamespace(id=user_id)
        self.message = SimpleNamespace(chat=SimpleNamespace(id=user_id))
        self.answers: list[tuple[str, bool]] = []

    async def answer(self, text: str, show_alert: bool = False) -> None:
        self.answers.append((text, show_alert))


class Boom:
    """Бот, до которого проверка доступа не должна пускать."""

    def __getattr__(self, name):
        raise AssertionError(f"дошли до bot.{name}")


@pytest.fixture
def owned_job(monkeypatch):
    job = SimpleNamespace(user_id=1, edl={"x": 1}, files_purged_at=None, source_key="k",
                          source_local_path=None)

    async def get_job(job_id):
        return job

    async def lang_for_job(job_id):
        return "ru"

    async def get_user(user_id):
        return SimpleNamespace(is_blocked=user_id == 9)

    monkeypatch.setattr(video.repo, "get_job", get_job)
    monkeypatch.setattr(video.repo, "get_user", get_user)
    monkeypatch.setattr(video, "_lang_for_job", lang_for_job)
    monkeypatch.setattr(video.settings, "admin_ids", [])
    return job


async def test_foreign_job_is_not_rerendered(owned_job):
    cb = FakeCallback("rr:job-1:glass", user_id=2)
    await video.on_rerender(cb, Boom())
    assert cb.answers and cb.answers[0][1] is True


async def test_blocked_owner_cannot_rerender(owned_job):
    owned_job.user_id = 9
    cb = FakeCallback("rr:job-1:glass", user_id=9)
    await video.on_rerender(cb, Boom())
    assert cb.answers and cb.answers[0][1] is True


async def test_unknown_preset_is_not_queued(owned_job):
    cb = FakeCallback("rr:job-1:../../etc", user_id=1)
    await video.on_rerender(cb, Boom())
    assert cb.answers and cb.answers[0][1] is True


class FakeQueue:
    """Redis-двойник бота: замок, ключ прогресса и постановка в очередь."""

    def __init__(self, *, locked: bool = False, progress: dict | None = None) -> None:
        self.locked = locked
        self.progress = progress
        self.enqueued: list[tuple] = []
        self.deleted: list[str] = []

    async def set(self, key, value, **kw):
        return None if self.locked else True

    async def get(self, key):
        import json

        return json.dumps(self.progress) if self.progress is not None else None

    async def delete(self, key):
        self.deleted.append(key)

    async def enqueue_job(self, *a, **kw):
        self.enqueued.append(a)

    async def close(self):
        return None


def queue_of(monkeypatch, queue: FakeQueue) -> FakeQueue:
    async def get_queue(*a, **kw):
        return queue

    monkeypatch.setattr(video, "get_queue", get_queue)
    return queue


class ChatBot:
    """Бот, который помнит, что сказал пользователю."""

    def __init__(self) -> None:
        self.said: list[str] = []

    async def send_message(self, chat_id, text, **kw):
        self.said.append(text)
        return SimpleNamespace(message_id=1)

    async def edit_message_text(self, text, **kw):
        self.said.append(text)


async def test_a_second_rerender_waits_for_the_first(owned_job, monkeypatch):
    """Перерендер бесплатный, а очередь у него общая с оплаченными роликами.

    Без замка десяток нажатий на кнопки тем вставал впереди чужих платных задач:
    воркер берёт по одному.
    """
    async def no_jobs(*a, **kw):
        return 0

    monkeypatch.setattr(video.repo, "active_jobs_count", no_jobs)
    queue = queue_of(monkeypatch, FakeQueue(locked=True))

    cb = FakeCallback("rr:job-1:glass", user_id=1)
    await video.on_rerender(cb, Boom())

    assert cb.answers and cb.answers[-1][1] is True
    assert not queue.enqueued


async def test_a_full_queue_takes_no_rerenders(owned_job, monkeypatch):
    async def full(*a, **kw):
        return video.settings.queue_max_size

    async def never(*a, **kw):
        raise AssertionError("перерендер дошёл до очереди, хотя она забита")

    monkeypatch.setattr(video.repo, "active_jobs_count", full)
    monkeypatch.setattr(video, "get_queue", never)

    cb = FakeCallback("rr:job-1:glass", user_id=1)
    await video.on_rerender(cb, Boom())

    assert cb.answers and cb.answers[-1][1] is True


async def test_a_rerender_is_queued_with_a_clean_slate(owned_job, monkeypatch):
    """Исход прошлого перерендера в ту же тему живёт в Redis три часа. Не стереть
    его — наблюдатель примет старое «готово» или старый отказ за новый ответ."""
    async def no_jobs(*a, **kw):
        return 0

    async def no_watch(*a, **kw):
        return None

    monkeypatch.setattr(video.repo, "active_jobs_count", no_jobs)
    monkeypatch.setattr(video, "_watch_rerender", no_watch)
    queue = queue_of(monkeypatch, FakeQueue())

    await video.on_rerender(FakeCallback("rr:job-1:glass", user_id=1), ChatBot())

    assert queue.deleted == [video.progress_key("job-1:glass")]
    assert queue.enqueued == [("rerender_job", "job-1", "glass")]


async def test_a_failed_rerender_reaches_the_chat(monkeypatch):
    """Наблюдатель перерендера ждал только успеха: отказ воркера оставлял
    пользователя смотреть на «Пересобираю…» до конца срока."""
    async def lang_for_job(job_id):
        return "ru"

    async def no_sleep(_sec):
        return None

    monkeypatch.setattr(video, "_lang_for_job", lang_for_job)
    monkeypatch.setattr(video.asyncio, "sleep", no_sleep)
    queue_of(monkeypatch, FakeQueue(progress={"stage": "failed", "error_code": "source_lost"}))
    bot = ChatBot()

    await video._watch_rerender(bot, 1, 2, "job-1", "glass")

    from src.core.i18n import error_text

    assert bot.said == [
        error_text(
            "source_lost", "ru",
            min_s=video.settings.min_video_seconds, max_m=video.settings.max_video_minutes,
        )
    ]


async def test_the_watcher_learns_the_outcome_from_the_database(monkeypatch):
    """Ключ прогресса в Redis истекает и может не дойти; исход задачи наверняка
    записан только в базе. Наблюдатель сверяется с ней, а не ждёт ключа вечно."""
    from src.db.models import JobStatus

    async def lang_for_job(job_id):
        return "ru"

    async def no_sleep(_sec):
        return None

    async def done_job(job_id):
        return SimpleNamespace(status=JobStatus.DONE, error_code=None)

    delivered: list[str] = []

    async def deliver(bot, chat_id, message_id, job_id, preset):
        delivered.append(job_id)

    monkeypatch.setattr(video, "_lang_for_job", lang_for_job)
    monkeypatch.setattr(video.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(video.repo, "get_job", done_job)
    monkeypatch.setattr(video, "_deliver", deliver)
    queue_of(monkeypatch, FakeQueue(progress=None))

    await video._watch(Boom(), 1, 2, "job-1", "glass")

    assert delivered == ["job-1"]


def test_the_watcher_outlives_the_longest_honest_job():
    """Задача вправе шесть часов стоять в очереди и три часа работать. Наблюдатель
    с трёхчасовым сроком сдавался раньше, и готовый ролик не доезжал."""
    from src.worker import tasks

    assert video.WATCH_SECONDS >= tasks.QUEUED_MAX_WAIT + tasks.JOB_TIMEOUT


async def test_a_dead_queue_does_not_keep_the_money(monkeypatch, tmp_path):
    """Redis недоступен в момент постановки: ролик уже списан, а задачу никто не
    возьмёт. Без отката она стояла бы в очереди до отсечки сторожа, и всё это
    время пользователь не мог прислать следующее видео."""
    async def upsert_user(*a, **kw):
        return SimpleNamespace(
            is_blocked=False, is_admin=False, lang="ru", language_code="ru",
            preset="glass", settings=None, balance=1,
        )

    async def no_jobs(*a, **kw):
        return 0

    async def create_job(*a, **kw):
        return SimpleNamespace(id="job-1")

    async def noop(*a, **kw):
        return None

    async def resolve_source(bot, file_id, dst):
        return tmp_path / "source.mp4", None

    async def dead_queue(*a, **kw):
        raise ConnectionError("redis недоступен")

    undone: list[str] = []

    async def mark_failed(job_id, error, **kw):
        undone.append("failed")

    async def refund_job(job_id):
        undone.append("refund")
        return True

    def never_watch(*a, **kw):
        raise AssertionError("наблюдатель повешен на задачу, которой нет в очереди")

    monkeypatch.setattr(video.repo, "upsert_user", upsert_user)
    monkeypatch.setattr(video.repo, "active_jobs_count", no_jobs)
    monkeypatch.setattr(video.repo, "create_job", create_job)
    monkeypatch.setattr(video.repo, "update_job", noop)
    monkeypatch.setattr(video.repo, "mark_failed", mark_failed)
    monkeypatch.setattr(video.repo, "refund_job", refund_job)
    monkeypatch.setattr(video, "_resolve_source", resolve_source)
    monkeypatch.setattr(video.storage, "upload", noop)
    monkeypatch.setattr(video, "get_queue", dead_queue)
    monkeypatch.setattr(video, "_watch", never_watch)
    monkeypatch.setattr(video.settings, "bot_api_url", "")
    bot = ChatBot()

    async def answer(text, **kw):
        return SimpleNamespace(message_id=1)

    msg = SimpleNamespace(
        from_user=SimpleNamespace(id=5, username=None, first_name=None, language_code="ru"),
        chat=SimpleNamespace(id=5),
        video=SimpleNamespace(mime_type="video/mp4", duration=30, file_size=1024, file_id="f"),
        video_note=None, document=None, answer=answer,
    )
    await video.on_video(msg, bot)

    assert undone == ["failed", "refund"]
    assert bot.said, "пользователь не узнал, что ролик не принят"


async def test_blocked_user_cannot_send_video(monkeypatch):
    async def upsert_user(*a, **kw):
        return SimpleNamespace(is_blocked=True, is_admin=False, lang="ru", language_code="ru")

    async def never(*a, **kw):
        raise AssertionError("заблокированный дошёл до очереди")

    monkeypatch.setattr(video.repo, "upsert_user", upsert_user)
    monkeypatch.setattr(video.repo, "active_jobs_count", never)
    replies: list[str] = []

    async def answer(text, **kw):
        replies.append(text)

    msg = SimpleNamespace(
        from_user=SimpleNamespace(id=5, username=None, first_name=None, language_code="ru"),
        video=SimpleNamespace(mime_type="video/mp4"), video_note=None, document=None,
        answer=answer,
    )
    await video.on_video(msg, Boom())
    assert replies


async def test_file_over_cloud_limit_goes_to_the_app_without_charging(monkeypatch):
    """Облачный Bot API не отдаёт файл больше 20 МБ: повтор не поможет, а баланс
    списывать не за что."""
    async def upsert_user(*a, **kw):
        return SimpleNamespace(is_blocked=False, is_admin=False, lang="ru", language_code="ru")

    async def no_jobs(*a, **kw):
        return 0

    async def never(*a, **kw):
        raise AssertionError("джоб создан для файла, который не скачать")

    monkeypatch.setattr(video.repo, "upsert_user", upsert_user)
    monkeypatch.setattr(video.repo, "active_jobs_count", no_jobs)
    monkeypatch.setattr(video.repo, "create_job", never)
    monkeypatch.setattr(video.settings, "bot_api_url", "")
    replies: list[str] = []

    async def answer(text, **kw):
        replies.append(text)

    msg = SimpleNamespace(
        from_user=SimpleNamespace(id=5, username=None, first_name=None, language_code="ru"),
        video=SimpleNamespace(mime_type="video/mp4", duration=61, file_size=48 * 1024 * 1024),
        video_note=None, document=None, answer=answer,
    )
    await video.on_video(msg, Boom())
    assert replies and "20 МБ" in replies[0]


# ------------------------------------------------------------------ старт бота


async def test_get_me_survives_a_short_telegram_outage(monkeypatch):
    """Короткий сбой Telegram не должен ронять процесс.

    3 ноября 2026 сеть сервера перестала доходить до api.telegram.org: `get_me`
    падал, процесс выходил с кодом 1, systemd поднимал заново — раз в две
    минуты. Mini App живёт в этом же процессе и стартует ПОСЛЕ `get_me`, а
    адрес туннеля на каждом запуске новый, поэтому кнопка в чате у всех
    пользователей протухала с той же частотой.
    """
    import asyncio

    from src.bot import main as bot_main

    slept: list[float] = []

    async def no_sleep(sec):
        slept.append(sec)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    tries = {"n": 0}

    class Flaky:
        async def get_me(self):
            tries["n"] += 1
            if tries["n"] < 3:
                raise RuntimeError("Gateway Timeout")
            return SimpleNamespace(username="bot")

    me = await bot_main._get_me_persistently(Flaky())
    assert me.username == "bot"
    assert tries["n"] == 3
    assert slept == [3, 6], "повторы идут с нарастающей паузой"


async def test_a_long_telegram_outage_ends_the_process(monkeypatch):
    """Если Telegram не ответил и через минуту — выходим, а не висим молча.

    Процесс, который ничего не может, держать смысла нет: дальше дело systemd.
    """
    import asyncio

    from src.bot import main as bot_main

    async def no_sleep(_sec):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    class Dead:
        async def get_me(self):
            raise RuntimeError("Gateway Timeout")

    with pytest.raises(SystemExit) as e:
        await bot_main._get_me_persistently(Dead())
    assert "Telegram недоступен" in str(e.value)


# ------------------------------------------------------------------ бесплатный режим


def test_free_mode_charges_nobody_and_paid_mode_charges_all_but_owners():
    """Решение «списывать ли ролик» одно на оба входа — чат и приложение."""
    from src.core.config import Settings

    paid = Settings(_env_file=None, free_videos=False)
    free = Settings(_env_file=None, free_videos=True)
    assert paid.charges(is_admin=False) and not paid.charges(is_admin=True)
    assert not free.charges(is_admin=False) and not free.charges(is_admin=True)


async def test_free_mode_lets_an_empty_balance_send_a_video_from_the_chat(monkeypatch):
    """Ролик из чата в бесплатном режиме ставится без списания — иначе
    пустой баланс упирался бы в пейволл, которого сейчас нет."""
    async def upsert_user(*a, **kw):
        return SimpleNamespace(
            is_blocked=False, is_admin=False, lang="ru", language_code="ru",
            preset="glass", settings=None, balance=0,
        )

    async def no_jobs(*a, **kw):
        return 0

    seen = {}

    class Stop(Exception):
        pass

    async def create_job(*a, **kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(video.repo, "upsert_user", upsert_user)
    monkeypatch.setattr(video.repo, "active_jobs_count", no_jobs)
    monkeypatch.setattr(video.repo, "create_job", create_job)
    monkeypatch.setattr(video.settings, "free_videos", True)

    async def answer(text, **kw):
        return SimpleNamespace(message_id=1)

    msg = SimpleNamespace(
        from_user=SimpleNamespace(id=5, username=None, first_name=None, language_code="ru"),
        chat=SimpleNamespace(id=5),
        video=SimpleNamespace(mime_type="video/mp4", duration=30, file_size=1024, file_id="f"),
        video_note=None, document=None, answer=answer,
    )
    with pytest.raises(Stop):
        await video.on_video(msg, Boom())
    assert seen["charge"] is False
