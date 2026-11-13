"""Значки и цвет кнопок: контурные эмодзи вместо цветных в подписи.

Значок — кастомный эмодзи из набора бота `cliply_icons_by_ClipLy_app`
(перекрашивается в цвет текста кнопки). Telegram пускает такие значки, только
пока у владельца бота есть Premium, поэтому отказ из-за значка не имеет права
сломать сообщение: оно уходит ещё раз, без значков.
"""

from __future__ import annotations

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMessage
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup

from src.bot import icons, keyboards
from src.bot.handlers import menu
from src.core.i18n import LANGS, t

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _icons_on(monkeypatch):
    monkeypatch.setattr(icons, "_enabled", True)
    was = keyboards.webapp_url()
    keyboards.set_webapp_url("https://example.trycloudflare.com")
    yield
    keyboards.set_webapp_url(was)


def test_main_buttons_carry_icons_and_no_colour_emoji_in_the_label():
    kb = keyboards.main_keyboard("ru")
    buttons = {b.text: b for row in kb.keyboard for b in row}
    for action in menu.ACTIONS:
        b = buttons[t(f"kb.{action}", "ru")]
        assert b.icon_custom_emoji_id == icons.ICONS[action], action
        assert b.text[0].isalpha(), f"в подписи осталось эмодзи: {b.text!r}"
    assert buttons[t("kb.start", "ru")].style == "primary"
    assert buttons[t("kb.balance", "ru")].style == "success"


def test_the_open_app_button_has_its_icon():
    kb = keyboards.app_keyboard(None, "ru")
    assert kb.inline_keyboard[0][0].icon_custom_emoji_id == icons.ICONS["app"]


def test_old_emoji_labels_still_work_in_every_language():
    """Клавиатура с «▶️ Начать монтаж» остаётся у людей, пока бот не пришлёт
    новую. Нажатие на неё обязано делать то же, что и раньше."""
    for lang in LANGS:
        for action in menu.ACTIONS:
            assert menu.action_for(menu.PREVIOUS_LABELS[action][lang]) == action


def test_the_new_keyboard_is_delivered_once_more():
    """Кто уже получил клавиатуру с цветными эмодзи, получит новую — отметка другая."""
    assert menu.SEEN_FLAG != "kb"


def test_icons_can_be_switched_off():
    icons.disable("тест")
    kb = keyboards.main_keyboard("ru")
    assert all(b.icon_custom_emoji_id is None for row in kb.keyboard for b in row)


async def test_a_rejected_icon_is_retried_without_icons(monkeypatch):
    calls: list = []

    async def make_request(bot, method):
        calls.append(method.model_copy(deep=True))
        if len(calls) == 1:
            raise TelegramBadRequest(method=method, message="Bad Request: BUTTON_ICON_INVALID")
        return "ок"

    markup = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="Баланс", icon_custom_emoji_id="1",
                                                           style="success")]])
    method = SendMessage(chat_id=1, text="x", reply_markup=markup)
    got = await icons.IconFallback()(make_request, None, method)
    assert got == "ок" and len(calls) == 2
    retried = calls[1].reply_markup.keyboard[0][0]
    assert retried.icon_custom_emoji_id is None
    assert retried.style == "success", "цвет кнопки не зависит от Premium — его не снимаем"
    assert icons.enabled() is False, "дальше клавиатуры собираются уже без значков"


async def test_other_errors_are_not_swallowed():
    async def make_request(bot, method):
        raise TelegramBadRequest(method=method, message="Bad Request: chat not found")

    markup = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="a", callback_data="a",
                                                                         icon_custom_emoji_id="1")]])
    with pytest.raises(TelegramBadRequest):
        await icons.IconFallback()(make_request, None, SendMessage(chat_id=1, text="x", reply_markup=markup))
    assert icons.enabled() is True
