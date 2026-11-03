"""Клавиатуры. Ничего, кроме разметки."""

from __future__ import annotations

from aiogram.types import (
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    WebAppInfo,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.core.config import settings
from src.core.i18n import LANG_NAMES, LANGS, t
from src.core.preset import available, load_preset


def _titles() -> list[tuple[str, str]]:
    """Темы в порядке ленты приложения: `order`, а не алфавит.

    Порядок один на оба входа — чат и Mini App, — иначе «третья слева» значит
    в них разное, а выбирают люди глазами, а не по имени файла.
    """
    items = []
    for name in available():
        try:
            preset = load_preset(name)
        except Exception:  # noqa: BLE001 — битый пресет не должен ронять меню
            items.append((100, name, name))
            continue
        items.append((preset.order, name, preset.title or name))
    items.sort()
    return [(name, title) for _, name, title in items]


def presets_keyboard(current: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for name, title in _titles():
        mark = "● " if name == current else "○ "
        kb.button(text=f"{mark}{title}", callback_data=f"preset:{name}")
    kb.adjust(2)
    return kb.as_markup()


def result_keyboard(job_id: str, current: str, lang: str | None = None) -> InlineKeyboardMarkup:
    """После готового ролика: перерендер тем же EDL, но другим стилем (§0.4).

    Кнопка — одно название темы, а не «Пересобрать в „X“»: тем семнадцать, и
    развёрнутая формулировка в столбик давала под каждым роликом стену из
    шестнадцати строк. Что делает нажатие, говорит подпись к видео
    (`video.caption_rebuild`), и говорит один раз.
    """
    del lang  # подпись переехала в caption; у кнопки остаётся только название
    kb = InlineKeyboardBuilder()
    for name, title in _titles():
        if name == current:
            continue
        kb.button(text=title, callback_data=f"rerender:{job_id}:{name}")
    kb.adjust(3)
    return kb.as_markup()


def language_keyboard(current: str | None = None) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for code in LANGS:
        mark = "● " if code == current else ""
        kb.button(text=f"{mark}{LANG_NAMES[code]}", callback_data=f"lang:{code}")
    kb.adjust(2)
    return kb.as_markup()


# Адрес Mini App известен только после старта: туннель выдаёт его на запуске и
# на каждом запуске новый. Хендлеры берут его отсюда, а не из настроек.
_webapp_url: str | None = None


def set_webapp_url(url: str | None) -> None:
    global _webapp_url
    _webapp_url = url.rstrip("/") if url else None


def webapp_url() -> str | None:
    return _webapp_url


def _app_url(section: str | None) -> str | None:
    if not _webapp_url:
        return None
    # Вкладка — параметром запроса: фрагмент адреса Telegram занимает своими
    # параметрами запуска, и `#balance` там не доживёт до приложения.
    return f"{_webapp_url}/?screen={section}" if section else _webapp_url


def app_keyboard(section: str | None = None, lang: str | None = None) -> InlineKeyboardMarkup | None:
    """Кнопка «открыть приложение». Без адреса приложения — без кнопки, а не с битой."""
    url = _app_url(section)
    if not url:
        return None
    kb = InlineKeyboardBuilder()
    kb.button(text=t("btn.open_app", lang), web_app=WebAppInfo(url=url))
    return kb.as_markup()


# Подписи кнопок нижней клавиатуры: действие → ключ перевода. Порядок задаёт
# раскладку — первая кнопка во всю ширину, остальные парами.
MAIN_BUTTONS: tuple[tuple[str, ...], ...] = (
    ("start",),
    ("balance", "videos"),
    ("referral", "support"),
)


def main_keyboard(lang: str | None = None) -> ReplyKeyboardMarkup:
    """Постоянная клавиатура под полем ввода: пять кнопок, 1 / 2 / 2.

    Первая кнопка открывает Mini App НАПРЯМУЮ только при постоянном адресе
    (`WEBAPP_URL`). С туннелем адрес новый на каждом запуске, а эту клавиатуру
    Telegram хранит у пользователя, пока её не заменят, — вшитый адрес туннеля
    назавтра вёл бы в никуда и кнопка молча не открывала бы ничего. Поэтому с
    туннелем кнопка обычная: бот отвечает на неё инлайн-кнопкой со СВЕЖИМ
    адресом, и это один лишний тап против неработающей кнопки.
    """
    url = _webapp_url
    direct = url and settings.webapp_url.strip()
    rows = [
        [
            KeyboardButton(text=t(f"kb.{name}", lang), web_app=WebAppInfo(url=url))
            if name == "start" and direct
            else KeyboardButton(text=t(f"kb.{name}", lang))
            for name in row
        ]
        for row in MAIN_BUTTONS
    ]
    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True, is_persistent=True)


def welcome_keyboard(lang: str) -> InlineKeyboardMarkup:
    """Приветствие: открыть приложение и тут же сменить язык, не вспоминая команду."""
    kb = InlineKeyboardBuilder()
    url = _app_url(None)
    if url:
        kb.button(text=t("btn.open_app", lang), web_app=WebAppInfo(url=url))
    other = [code for code in LANGS if code != lang]
    for code in other:
        kb.button(text=LANG_NAMES[code], callback_data=f"lang:{code}")
    kb.adjust(1, len(other))
    return kb.as_markup()
