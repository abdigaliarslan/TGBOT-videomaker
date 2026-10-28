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
