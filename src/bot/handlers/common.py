"""Команды: /start, /help, /preset, /stats."""

from __future__ import annotations

import logging

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message

from src.bot.keyboards import presets_keyboard
from src.core.config import settings
from src.core.preset import available, load_preset
from src.db import repo

log = logging.getLogger(__name__)
router = Router(name="common")

WELCOME = (
    "Пришлите вертикальное или горизонтальное видео — верну готовый ролик:\n\n"
    "• пословные субтитры с акцентами\n"
    "• вырезанные паузы и заминки\n"
    "• кадр 9:16 с лицом по центру\n\n"
    "Длительность: от {min_s:.0f} с до {max_m:.0f} мин.\n"
    "Стиль субтитров — /preset."
)


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    user = message.from_user
    if user is None:
        return
    await repo.upsert_user(
        user.id,
        username=user.username,
        first_name=user.first_name,
        language_code=user.language_code,
        is_admin=user.id in settings.admin_ids,
    )
    await message.answer(
        WELCOME.format(min_s=settings.min_video_seconds, max_m=settings.max_video_minutes)
    )


@router.message(Command("help"))
async def on_help(message: Message) -> None:
    await message.answer(
        "Как это работает:\n\n"
        "1. Распознаю речь пословно.\n"
        "2. Решаю, что вырезать и что выделить.\n"
        "3. Режу паузы, кадрирую 9:16, рисую субтитры.\n\n"
        "Команды:\n"
        "/preset — стиль субтитров\n"
        "/stats — что уже сделано\n\n"
        "Если ролик не получился — пришлите его ещё раз, "
        "а лучше напишите, что именно не так."
    )


@router.message(Command("preset"))
async def on_preset(message: Message) -> None:
    user = message.from_user
    if user is None:
        return
    db_user = await repo.get_user(user.id)
    current = db_user.preset if db_user else settings.default_preset
    names = ", ".join(available())
    await message.answer(
        f"Стиль субтитров. Сейчас: «{current}».\nДоступны: {names}",
        reply_markup=presets_keyboard(current),
    )


@router.callback_query(lambda c: c.data and c.data.startswith("preset:"))
async def on_preset_pick(callback: CallbackQuery) -> None:
    if callback.data is None or callback.from_user is None:
        return
    name = callback.data.split(":", 1)[1]
    if name not in available():
        await callback.answer("Такого стиля нет", show_alert=True)
        return
    await repo.set_user_preset(callback.from_user.id, name)
    title = load_preset(name).title or name
    await callback.answer(f"Стиль: {title}")
    if isinstance(callback.message, Message):
        await callback.message.edit_reply_markup(reply_markup=presets_keyboard(name))


@router.message(Command("stats"))
async def on_stats(message: Message) -> None:
    user = message.from_user
    if user is None:
        return
    db_user = await repo.get_user(user.id)
    if db_user is None:
        await message.answer("Пока ничего не смонтировано. Пришлите видео.")
        return

    lines = [f"Готовых роликов: {db_user.videos_done}", f"Стиль: {db_user.preset}"]
    if db_user.is_admin:
        report = await repo.cost_report(days=7)
        if report:
            lines.append("\nРасход на модель за 7 дней:")
            total = 0.0
            for row in report:
                total += float(row["cost_usd"] or 0)
                lines.append(
                    f"  {row['prompt']:<18} {row['calls']:>4} выз.  "
                    f"${float(row['cost_usd'] or 0):.3f}  "
                    f"(кеш {int(row['cache_read_tokens'] or 0)} т.)"
                )
            lines.append(f"  Итого: ${total:.2f}")
        else:
            lines.append("\nВызовов модели пока не было.")
    await message.answer("\n".join(lines))
