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

from src.bot.handlers import common, fallback, payments, video
from src.bot.keyboards import set_webapp_url
from src.core.config import settings
from src.core.logging import setup_logging
from src.db import repo
from src.storage.s3 import storage

log = logging.getLogger(__name__)


def make_bot() -> Bot:
    # Отдача готового ролика — это заливка десятков мегабайт. Дефолтные 60 с
    # на неё не хватает: проверено, 24 МБ отваливались по таймауту, и юзер
    # оставался без файла при полностью успешном монтаже.
    if settings.bot_api_url:
        # local=True — aiogram знает, что file_path это путь на диске, а не URL.
        api = TelegramAPIServer.from_base(settings.bot_api_url.rstrip("/"), is_local=True)
        session = AiohttpSession(api=api, timeout=settings.bot_upload_timeout_s)
        log.info("использую локальный Bot API: %s", settings.bot_api_url)
    else:
        session = AiohttpSession(timeout=settings.bot_upload_timeout_s)
        log.warning(
            "BOT_API_URL пуст — работает облачный Bot API, лимит файла 20 МБ. "
            "Поднимите telegram-bot-api из docker-compose, чтобы принимать до 2 ГБ."
        )
    return Bot(
        token=settings.bot_token,
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


async def start_webapp(bot: Bot):
    """Сервер Mini App, туннель и кнопка меню. Любой сбой здесь не роняет бота.

    Чат работает и без приложения, поэтому упавший туннель — это предупреждение
    в логе и бот без кнопки, а не бот, который не запустился.
    """
    from aiogram.types import MenuButtonWebApp, WebAppInfo

    from src.webapp.server import start_server
    from src.webapp.tunnel import Tunnel, ensure_binary

    runner = tunnel = None
    try:
        runner = await start_server(bot, settings)
    except Exception as e:  # noqa: BLE001
        log.error("Mini App не поднялся (порт %s занят?): %s", settings.webapp_port, e)
        return None, None

    url = settings.webapp_url.strip()
    if not url and settings.webapp_tunnel:
        try:
            binary = await asyncio.to_thread(ensure_binary, settings.cloudflared_bin)
            tunnel = Tunnel(binary, f"http://{settings.webapp_host}:{settings.webapp_port}")
            url = await tunnel.start()
        except Exception as e:  # noqa: BLE001
            log.error("туннель для Mini App не поднялся: %s", e)
            tunnel = None
            url = ""
    if not url:
        log.warning("у Mini App нет публичного адреса — задайте WEBAPP_URL или WEBAPP_TUNNEL=1")
        return runner, tunnel

    set_webapp_url(url)
    try:
        # Кнопка меню общая для всех чатов: адрес туннеля меняется на каждом
        # запуске, и старые чаты подхватывают свежий без единого сообщения.
        await bot.set_chat_menu_button(
            menu_button=MenuButtonWebApp(text="Монтаж", web_app=WebAppInfo(url=url))
        )
        log.info("Mini App: %s", url)
        # Кнопки в чатах, где у пользователя своя подпись, хранят старый адрес —
        # перевешиваем фоном, чтобы не задерживать старт.
        from src.bot.handlers.common import refresh_menu_buttons

        asyncio.create_task(refresh_menu_buttons(bot))
    except Exception as e:  # noqa: BLE001
        log.error("не смог поставить кнопку меню: %s", e)
    return runner, tunnel


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
    dp.include_router(payments.router)
    dp.include_router(common.router)
    dp.include_router(video.router)
    # Последним: подбирает текст, который не взял никто, включая старые кнопки.
    dp.include_router(fallback.router)

    runner, tunnel = await start_webapp(bot)
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        # Задачи, шедшие во время рестарта, иначе доделались бы в тишину.
        await video.resume_watchers(bot)
        await dp.start_polling(bot)
    finally:
        if tunnel is not None:
            await tunnel.stop()
        if runner is not None:
            await runner.cleanup()
        await bot.session.close()
        await repo.dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
