"""Точка входа бота.

Локальный Bot API server подключается через `TelegramAPIServer.from_base` —
это снимает лимит 20 МБ и поднимает его до 2 ГБ (§1).
"""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.enums import ParseMode

from src.bot.handlers import common, video
from src.core.config import settings
from src.core.logging import setup_logging
from src.db import repo
from src.storage.s3 import storage

log = logging.getLogger(__name__)


def make_bot() -> Bot:
    session = None
    if settings.bot_api_url:
        # local=True — aiogram знает, что file_path это путь на диске, а не URL.
        api = TelegramAPIServer.from_base(settings.bot_api_url.rstrip("/"), is_local=True)
        session = AiohttpSession(api=api)
        log.info("использую локальный Bot API: %s", settings.bot_api_url)
    else:
        log.warning(
            "BOT_API_URL пуст — работает облачный Bot API, лимит файла 20 МБ. "
            "Поднимите telegram-bot-api из docker-compose, чтобы принимать до 2 ГБ."
        )
    return Bot(
        token=settings.bot_token,
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


async def main() -> None:
    setup_logging(settings.log_level)
    if not settings.bot_token:
        raise SystemExit("BOT_TOKEN не задан в .env")

    repo.init_engine(settings)
    await storage.ensure_bucket_async()

    bot = make_bot()
    me = await bot.get_me()
    log.info("бот @%s запущен", me.username)

    dp = Dispatcher()
    dp.include_router(common.router)
    dp.include_router(video.router)

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        # Задачи, шедшие во время рестарта, иначе доделались бы в тишину.
        await video.resume_watchers(bot)
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        await repo.dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
