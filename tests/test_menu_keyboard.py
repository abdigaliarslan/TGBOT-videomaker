"""Нижняя клавиатура бота: раскладка, узнавание подписей, адрес приложения."""

from __future__ import annotations

import pytest

from src.bot.handlers import fallback, menu
from src.bot.keyboards import main_keyboard, set_webapp_url
from src.core.i18n import LANGS, t

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _tunnel_url():
    """Адрес приложения живёт в модуле — возвращаем его после каждого теста."""
    from src.bot import keyboards

    was = keyboards.webapp_url()
    set_webapp_url("https://example.trycloudflare.com")
    yield
    set_webapp_url(was)


def test_the_keyboard_has_five_buttons_in_one_two_two():
    """Раскладка — часть задачи: первая кнопка во всю ширину, дальше парами."""
    kb = main_keyboard("ru")
    assert [len(row) for row in kb.keyboard] == [1, 2, 2]
    assert [b.text for row in kb.keyboard for b in row] == [
        t("kb.start", "ru"), t("kb.balance", "ru"),
        t("kb.videos", "ru"), t("kb.referral", "ru"), t("kb.support", "ru"),
    ]
    # Без этих двух клавиатура либо занимает пол-экрана, либо исчезает после
    # первого нажатия — и пользователь остаётся без навигации.
    assert kb.resize_keyboard and kb.is_persistent


def test_every_label_is_recognised_in_every_language():
    """Клавиатура остаётся в чате после смены языка: Telegram хранит её, пока
    бот не пришлёт другую. Значит подписи надо узнавать на всех языках сразу."""
    for lang in LANGS:
        for action in menu.ACTIONS:
            assert menu.action_for(t(f"kb.{action}", lang)) == action, (lang, action)
    assert menu.action_for("привет") is None
    assert menu.action_for(None) is None


def test_labels_do_not_collide_between_languages():
    """Одна подпись — одно действие. Совпадение означало бы, что кнопка делает
    разное в зависимости от языка, на котором её завели."""
    seen: dict[str, tuple[str, str]] = {}
    for lang in LANGS:
        for action in menu.ACTIONS:
            key = menu._norm(t(f"kb.{action}", lang))
            assert key not in seen or seen[key][1] == action, (key, seen.get(key), lang, action)
            seen[key] = (lang, action)


def test_the_tunnel_address_is_not_baked_into_the_button(monkeypatch):
    """С туннелем адрес новый на каждом запуске, а клавиатуру Telegram хранит у
    пользователя. Вшитый туда адрес назавтра ведёт в никуда, и кнопка молча
    перестаёт открывать приложение — поэтому при туннеле она обычная."""
    from src.bot import keyboards

    monkeypatch.setattr(keyboards.settings, "webapp_url", "", raising=False)
    assert main_keyboard("ru").keyboard[0][0].web_app is None

    monkeypatch.setattr(keyboards.settings, "webapp_url", "https://reels.example", raising=False)
    button = main_keyboard("ru").keyboard[0][0]
    assert button.web_app is not None
    assert button.web_app.url == "https://example.trycloudflare.com", (
        "адрес берётся тот, что поднят сейчас, а не переписанный из настройки"
    )


def test_the_old_keyboard_handler_keeps_its_hands_off_the_new_buttons():
    """«Начать монтаж» есть и на старой клавиатуре, и на новой. Если бы его
    забирал fallback, ответом было бы «кнопки обновились» вместе со снятием
    самой клавиатуры — и так на каждое нажатие."""
    for action in menu.ACTIONS:
        for lang in LANGS:
            assert fallback.legacy_action(t(f"kb.{action}", lang)) is None
    # Настоящие старые кнопки он при этом по-прежнему узнаёт.
    assert fallback.legacy_action("📊 Моя статистика") == "stats"
    assert fallback.legacy_action("💎 Баланс") == "balance"


# ------------------------------------------------- доставка клавиатуры в старые чаты


def _message(user_id: int = 5):
    from datetime import datetime, timezone

    from aiogram.types import Chat, Message, User

    return Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=Chat(id=user_id, type="private"),
        from_user=User(id=user_id, is_bot=False, first_name="Т"),
    )


class _User:
    def __init__(self, settings):
        self.settings = settings
        self.lang = "ru"
        self.language_code = "ru"


async def test_the_keyboard_reaches_an_old_chat_exactly_once(monkeypatch):
    """Прежняя версия бота клавиатуру сняла, а новую Telegram сам не покажет:
    реплай-клавиатура приходит только с сообщением. Без доставки новые кнопки
    увидели бы лишь те, кто сам наберёт /start."""
    from src.db import repo

    shown: list[int] = []
    saved: list[dict] = []
    stored = {"settings": {}}

    async def get_user(_uid):
        return _User(stored["settings"])

    async def set_settings(_uid, **values):
        saved.append(values)
        stored["settings"] = {**stored["settings"], **values}

    async def show(message, lang):
        shown.append(message.chat.id)

    monkeypatch.setattr(repo, "get_user", get_user)
    monkeypatch.setattr(repo, "set_user_settings", set_settings)
    monkeypatch.setattr(menu, "show_keyboard", show)

    async def handler(_event, _data):
        return "ответ"

    once = menu.KeyboardOnce()
    assert await once(handler, _message(), {}) == "ответ"
    assert shown == [5] and saved == [{menu.SEEN_FLAG: 1}]

    # Второе сообщение — уже без клавиатуры: отметка легла в `users.settings`.
    assert await once(handler, _message(), {}) == "ответ"
    assert shown == [5], "клавиатура приходит один раз, а не на каждое сообщение"


async def test_a_broken_delivery_never_eats_the_answer(monkeypatch):
    """Клавиатура — удобство. Сбой базы не должен отменять ответ на сообщение."""
    from src.db import repo

    async def boom(_uid):
        raise RuntimeError("база недоступна")

    monkeypatch.setattr(repo, "get_user", boom)

    async def handler(_event, _data):
        return "ответ"

    assert await menu.KeyboardOnce()(handler, _message(), {}) == "ответ"
