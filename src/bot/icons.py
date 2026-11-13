"""Значки кнопок — контурные кастомные эмодзи вместо цветных в подписи.

Набор `cliply_icons_by_ClipLy_app` создал сам бот (Bot API
`createNewStickerSet`, `needs_repainting`): значок перекрашивается в цвет
текста кнопки и на любой теме выглядит частью подписи, а не наклейкой.
Рисунки — белый контур 100×100 на прозрачном; исходники не нужны, набор
живёт в Telegram.

⚠ Telegram пускает кастомные эмодзи в кнопки, только пока у владельца бота
есть Premium (или бот купил имя на Fragment). Кончится подписка — запросы с
такими кнопками начнут получать отказ. `IconFallback` ловит такой отказ,
повторяет запрос без значков и выключает их до перезапуска: кнопки остаются
рабочими, просто без значков. Цвет кнопки (`style`) от Premium не зависит.
"""

from __future__ import annotations

import logging
from typing import Any

from aiogram.client.session.middlewares.base import BaseRequestMiddleware
from aiogram.exceptions import TelegramBadRequest

log = logging.getLogger(__name__)

# Действие → custom_emoji_id из набора `cliply_icons_by_ClipLy_app`.
ICONS: dict[str, str] = {
    "start": "5456219684246434360",     # ▷
    "balance": "5456403152364415051",   # кошелёк
    "videos": "5456407232583348741",    # кадр с ▷
    "referral": "5456638203039624565",  # подарок
    "support": "5456321681129777582",   # гарнитура
    "app": "5456426637245587796",       # искра
}

# Цвет кнопки: главное действие синее, деньги зелёные. Остальные — как в теме.
STYLES: dict[str, str] = {"start": "primary", "balance": "success"}

_enabled = True


def enabled() -> bool:
    return _enabled


def disable(reason: str) -> None:
    global _enabled
    if _enabled:
        log.warning("значки кнопок выключены до перезапуска: %s", reason)
    _enabled = False


def icon(action: str) -> str | None:
    return ICONS.get(action) if _enabled else None


def _strip(markup: Any) -> bool:
    """Снимает значки с кнопок разметки на месте. Правда, если что-то снял."""
    rows = getattr(markup, "inline_keyboard", None) or getattr(markup, "keyboard", None) or []
    changed = False
    for row in rows:
        for button in row:
            if getattr(button, "icon_custom_emoji_id", None):
                button.icon_custom_emoji_id = None
                changed = True
    return changed


def _about_icons(message: str) -> bool:
    low = message.lower()
    return any(k in low for k in ("icon", "custom_emoji", "custom emoji", "premium"))


class IconFallback(BaseRequestMiddleware):
    """Отказ Telegram из-за значка → тот же запрос без значков."""

    async def __call__(self, make_request, bot, method):
        try:
            return await make_request(bot, method)
        except TelegramBadRequest as e:
            markup = getattr(method, "reply_markup", None)
            if markup is None or not _about_icons(e.message) or not _strip(markup):
                raise
            disable(e.message)
            return await make_request(bot, method)
