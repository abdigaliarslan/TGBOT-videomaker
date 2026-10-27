"""Команды: /start, /help, /language, /preset, /stats."""

from __future__ import annotations

import contextlib
import logging

from aiogram import Bot, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message, User

from src.bot.keyboards import language_keyboard, presets_keyboard, webapp_url, welcome_keyboard
from src.core.config import settings
from src.core.i18n import DEFAULT_LANG, LANGS, lang_of, t, videos_word
from src.core.preset import available, load_preset
from src.db import repo

log = logging.getLogger(__name__)
router = Router(name="common")


async def _upsert(user: User):
    return await repo.upsert_user(
        user.id,
        username=user.username,
        first_name=user.first_name,
        language_code=user.language_code,
        is_admin=user.id in settings.admin_ids,
    )


def welcome_text(lang: str) -> str:
    return t(
        "start.welcome",
        lang,
        min_s=f"{settings.min_video_seconds:.0f}",
        max_m=f"{settings.max_video_minutes:.0f}",
    )


async def apply_menu_button(bot: Bot, chat_id: int, lang: str) -> None:
    """Кнопка меню в чате пользователя под его язык.

    Русский совпадает с общей кнопкой бота, поэтому своей копии в чате НЕ
    заводим — сбрасываем чат на общую. Копия в чате хранит адрес целиком, а
    адрес туннеля новый на каждом запуске: такая копия после рестарта вела в
    мёртвый туннель, и «Монтаж» просто ничего не открывал. Общую кнопку бот
    перевешивает сам при старте, а английские копии обновляет
    `refresh_menu_buttons`.
    """
    from aiogram.types import MenuButtonDefault, MenuButtonWebApp, WebAppInfo

    url = webapp_url()
    with contextlib.suppress(Exception):  # кнопка меню — удобство, не повод ронять ответ
        if lang == DEFAULT_LANG or not url:
            await bot.set_chat_menu_button(chat_id=chat_id, menu_button=MenuButtonDefault())
            return
        await bot.set_chat_menu_button(
            chat_id=chat_id,
            menu_button=MenuButtonWebApp(text=t("btn.menu", lang), web_app=WebAppInfo(url=url)),
        )


async def refresh_menu_buttons(bot: Bot) -> int:
    """После смены адреса приложения — привести кнопки в чатах к свежему адресу.

    Касается всех, кто выбирал язык: английским перевешивается своя кнопка,
    русским своя копия сбрасывается на общую. Второе чинит чаты, где копию с
    русской подписью успела завести прошлая версия, — у владельца бота кнопка
    вела именно в такой мёртвый туннель.

    Идёт фоном и медленно: у Telegram лимит на частоту запросов, а старт бота
    из-за сотни чатов задерживаться не должен.
    """
    import asyncio

    ids = await repo.user_ids_with_chosen_lang()
    for user_id, lang in ids:
        await apply_menu_button(bot, user_id, lang)
        await asyncio.sleep(0.05)
    if ids:
        log.info("кнопка меню обновлена в чатах: %d", len(ids))
    return len(ids)


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    user = message.from_user
    if user is None:
        return
    db_user = await _upsert(user)
    if db_user.lang not in LANGS:
        # Первый запуск: сначала язык, приветствие — уже на нём.
        await message.answer(t("lang.pick", lang_of(db_user)), reply_markup=language_keyboard())
        return
    await message.answer(welcome_text(db_user.lang), reply_markup=welcome_keyboard(db_user.lang))


@router.message(Command("language"))
async def on_language(message: Message) -> None:
    user = message.from_user
    if user is None:
        return
    db_user = await _upsert(user)
    await message.answer(t("lang.pick", lang_of(db_user)), reply_markup=language_keyboard(db_user.lang))


@router.callback_query(lambda c: c.data and c.data.startswith("lang:"))
async def on_language_pick(callback: CallbackQuery, bot: Bot) -> None:
    if callback.data is None or callback.from_user is None:
        return
    lang = callback.data.split(":", 1)[1]
    if lang not in LANGS:
        await callback.answer()
        return
    await _upsert(callback.from_user)
    await repo.set_user_lang(callback.from_user.id, lang)
    await callback.answer(t("lang.set", lang))
    if isinstance(callback.message, Message):
        # Сообщение с выбором превращается в приветствие на выбранном языке:
        # никаких лишних сообщений в чате и сразу видно, что выбор применился.
        with contextlib.suppress(Exception):
            await callback.message.edit_text(welcome_text(lang), reply_markup=welcome_keyboard(lang))
        await apply_menu_button(bot, callback.message.chat.id, lang)


@router.message(Command("help"))
async def on_help(message: Message) -> None:
    user = message.from_user
    db_user = await repo.get_user(user.id) if user else None
    lang = lang_of(db_user) if db_user else lang_of(user)
    await message.answer(t("help.text", lang))


@router.message(Command("preset"))
async def on_preset(message: Message) -> None:
    user = message.from_user
    if user is None:
        return
    db_user = await repo.get_user(user.id)
    lang = lang_of(db_user) if db_user else lang_of(user)
    current = db_user.preset if db_user else settings.default_preset
    await message.answer(
        t("preset.current", lang, current=current, names=", ".join(available())),
        reply_markup=presets_keyboard(current),
    )


@router.callback_query(lambda c: c.data and c.data.startswith("preset:"))
async def on_preset_pick(callback: CallbackQuery) -> None:
    if callback.data is None or callback.from_user is None:
        return
    db_user = await repo.get_user(callback.from_user.id)
    lang = lang_of(db_user) if db_user else lang_of(callback.from_user)
    name = callback.data.split(":", 1)[1]
    if name not in available():
        await callback.answer(t("preset.unknown", lang), show_alert=True)
        return
    await repo.set_user_preset(callback.from_user.id, name)
    title = load_preset(name).title or name
    await callback.answer(t("preset.picked", lang, title=title))
    if isinstance(callback.message, Message):
        await callback.message.edit_reply_markup(reply_markup=presets_keyboard(name))


@router.message(Command("stats"))
async def on_stats(message: Message) -> None:
    user = message.from_user
    if user is None:
        return
    db_user = await repo.get_user(user.id)
    lang = lang_of(db_user) if db_user else lang_of(user)
    if db_user is None:
        await message.answer(t("stats.empty", lang))
        return

    lines = [t("stats.done", lang, n=db_user.videos_done), t("stats.style", lang, preset=db_user.preset)]
    if not db_user.is_admin:
        lines.append(t("stats.balance", lang, n=f"{db_user.balance} {videos_word(db_user.balance, lang)}"))
    if db_user.is_admin:
        report = await repo.cost_report(days=7)
        if report:
            lines.append(t("stats.cost_head", lang))
            total = 0.0
            for row in report:
                total += float(row["cost_usd"] or 0)
                lines.append(
                    f"  {row['prompt']:<18} {row['calls']:>4} × "
                    f"${float(row['cost_usd'] or 0):.3f}  "
                    f"(cache {int(row['cache_read_tokens'] or 0)})"
                )
            lines.append(t("stats.cost_total", lang, total=total))
        else:
            lines.append(t("stats.cost_none", lang))
    await message.answer("\n".join(lines))
