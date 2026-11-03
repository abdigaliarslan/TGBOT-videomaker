"""Всё, что не подобрали остальные роутеры. Подключается ПОСЛЕДНИМ.

Главный случай — старая клавиатура «🎬 Начать монтаж / 💎 Баланс / 📊 Моя
статистика / 💡 Как это работает». Её оставила в чатах прежняя версия бота на
этом же токене, а Telegram держит такую клавиатуру, пока бот сам её не уберёт.
Кнопки шлют обычный текст, и раньше бот на него молчал: пользователь жал
«Начать монтаж», и ничего не происходило. Теперь кнопки делают то, что обещают,
а клавиатура убирается первым же ответом.
"""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.types import Message, ReplyKeyboardRemove

from src.bot.handlers import common
from src.bot.keyboards import app_keyboard, welcome_keyboard
from src.core.i18n import lang_of, t
from src.db import repo

log = logging.getLogger(__name__)
router = Router(name="fallback")

# Текст кнопки без эмодзи → что показать. Сравнение по вхождению: эмодзи и
# регистр у старых версий клавиатуры могли отличаться.
_LEGACY = {
    "начать монтаж": "start",
    "баланс": "balance",
    "моя статистика": "stats",
    "как это работает": "help",
}


def legacy_action(text: str | None) -> str | None:
    # Кнопки НЫНЕШНЕЙ клавиатуры сюда не попадают, даже если их слова совпадают
    # со старыми («Начать монтаж», «Баланс»). Это не перестраховка: иначе всё
    # держалось бы на порядке роутеров, и перестановка молча заменяла бы ответ
    # на «кнопки обновились» вместе со снятием самой клавиатуры.
    from src.bot.handlers import menu

    if menu.action_for(text) is not None:
        return None
    low = (text or "").lower()
    for label, action in _LEGACY.items():
        if label in low:
            return action
    return None


@router.message(F.text)
async def on_text(message: Message) -> None:
    user = message.from_user
    if user is None:
        return
    db_user = await repo.upsert_user(
        user.id, username=user.username, first_name=user.first_name,
        language_code=user.language_code, is_admin=user.id in common.settings.admin_ids,
    )
    lang = lang_of(db_user)
    action = legacy_action(message.text)
    if action is None:
        await message.answer(t("fallback.hint", lang), reply_markup=app_keyboard(None, lang))
        return

    log.info("старая кнопка %r от %s", message.text, user.id)
    # Сначала убираем клавиатуру: у сообщения одна разметка, и инлайн-кнопки
    # приложения едут следующим сообщением.
    await message.answer(t("fallback.keyboard_removed", lang), reply_markup=ReplyKeyboardRemove())
    if action == "start":
        await message.answer(common.welcome_text(lang), reply_markup=welcome_keyboard(lang))
    elif action == "balance":
        await message.answer(
            t("stats.balance", lang, n=db_user.balance), reply_markup=app_keyboard("balance", lang)
        )
    elif action == "stats":
        await common.on_stats(message)
    else:
        await common.on_help(message)
