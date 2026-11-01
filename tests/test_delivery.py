"""Отдача готового ролика: обрыв связи не должен стоить пользователю монтажа."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError, TelegramRetryAfter

from src.bot.handlers import video as mod

pytestmark = pytest.mark.anyio


@dataclass
class FakeJob:
    id: uuid.UUID
    preset: str = "glass"
    source_duration: float | None = 30.0
    result_duration: float | None = 20.0
    result_key: str | None = "results/r.mp4"


class FakeBot:
    """Бот, который падает первые `fail` раз, а потом отправляет."""

    def __init__(self, fail: int = 0, error: Exception | None = None) -> None:
        self.fail = fail
        self.error = error or TelegramNetworkError(method=None, message="reset by peer")
        self.calls = 0

    async def send_video(self, **kw):
        self.calls += 1
        if self.calls <= self.fail:
            raise self.error
        return "отправлено"


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """Тесты не должны спать по-настоящему — но паузы всё равно замеряем."""
    slept: list[float] = []

    async def fake_sleep(sec: float) -> None:
        slept.append(sec)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    return slept


def job() -> FakeJob:
    return FakeJob(id=uuid.uuid4())


async def test_sends_on_first_try(tmp_path):
    bot = FakeBot(fail=0)
    await mod._send_video(bot, 1, job(), tmp_path / "r.mp4", "glass")
    assert bot.calls == 1


async def test_survives_connection_reset(tmp_path, no_waiting):
    """Ровно тот случай, что был на живом ролике."""
    bot = FakeBot(fail=2)
    await mod._send_video(bot, 1, job(), tmp_path / "r.mp4", "glass")
    assert bot.calls == 3
    assert no_waiting == [mod.DELIVERY_BACKOFF, mod.DELIVERY_BACKOFF * 2]  # пауза растёт


async def test_gives_up_after_limit_and_raises(tmp_path, no_waiting):
    """Сдались — значит наверх летит ошибка, и задача остаётся неотданной."""
    bot = FakeBot(fail=99)
    with pytest.raises(TelegramNetworkError):
        await mod._send_video(bot, 1, job(), tmp_path / "r.mp4", "glass")
    assert bot.calls == mod.DELIVERY_ATTEMPTS
    assert len(no_waiting) == mod.DELIVERY_ATTEMPTS - 1  # после последней попытки не ждём


async def test_honours_retry_after(tmp_path, no_waiting):
    """Флуд-контроль сам называет паузу — слушаемся его, а не своей формулы."""
    bot = FakeBot(fail=1, error=TelegramRetryAfter(method=None, message="flood", retry_after=17))
    await mod._send_video(bot, 1, job(), tmp_path / "r.mp4", "glass")
    assert no_waiting == [17.0]


async def test_logic_errors_are_not_retried(tmp_path):
    """Не тот чат или слишком большой файл от повтора не починятся."""
    bot = FakeBot(fail=99, error=TelegramBadRequest(method=None, message="chat not found"))
    with pytest.raises(TelegramBadRequest):
        await mod._send_video(bot, 1, job(), tmp_path / "r.mp4", "glass")
    assert bot.calls == 1


async def test_the_video_is_sent_with_its_real_frame(tmp_path, monkeypatch):
    """Без ширины и высоты iPhone показывал вертикальный ролик растянутым почти в квадрат."""
    from src.pipeline import probe as probe_mod

    class Info:
        width, height, duration = 1080, 1920, 31.77

    monkeypatch.setattr(probe_mod, "probe", lambda _p: Info())
    sent = {}

    class Bot:
        async def send_video(self, **kw):
            sent.update(kw)

    await mod._send_video(Bot(), 1, job(), tmp_path / "r.mp4", "glass")
    assert (sent["width"], sent["height"], sent["duration"]) == (1080, 1920, 32)


async def test_an_unreadable_file_is_still_sent(tmp_path):
    """Не прочитали размеры — отправляем как раньше, а не теряем ролик."""
    bot = FakeBot(fail=0)
    await mod._send_video(bot, 1, job(), tmp_path / "missing.mp4", "glass")
    assert bot.calls == 1


async def test_a_failure_report_survives_the_html_parser(tmp_path, monkeypatch):
    """Бот говорит с Telegram в режиме HTML, а текст исключения HTML не знает.

    Угловая скобка из `<class 'aiohttp...'>` делала сообщение невалидной
    разметкой, Telegram его отклонял, а `_safe_edit` ошибку глотал — и
    пользователь оставался смотреть на полосу прогресса, так и не узнав, что
    ролик готов и его можно забрать кнопкой.
    """
    edits: list[str] = []

    class Bot:
        async def edit_message_text(self, text, **kw):
            # Как сам Telegram: незакрытый тег — отказ.
            import re

            tags = re.findall(r"<[^>]*>?", text)
            bad = [tag for tag in tags if not re.fullmatch(r"</?(b|i|u|s|code|pre|a)[^<]*>", tag)]
            if bad:
                raise TelegramBadRequest(method=None, message=f"can't parse entities: {bad}")
            edits.append(text)

    async def one_job(_id):
        return FakeJob(id=uuid.uuid4())

    monkeypatch.setattr(mod.repo, "get_job", one_job)
    monkeypatch.setattr(mod.repo, "get_user", lambda _id: _none())
    monkeypatch.setattr(
        mod.storage, "download", lambda *a, **k: _boom("<class 'aiohttp.Timeout'> & co")
    )

    await mod._deliver(Bot(), 1, 2, "job-1", "glass")

    assert edits, "сообщение пользователю не прошло разметку и потерялось"
    assert "aiohttp.Timeout" in edits[-1], "причина потерялась вместе с разметкой"


async def _none():
    return None


async def _boom(message: str):
    raise RuntimeError(message)
