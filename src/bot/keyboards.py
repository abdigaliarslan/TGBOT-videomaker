"""Клавиатуры. Ничего, кроме разметки."""

from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup, WebAppInfo
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.core.i18n import LANG_NAMES, LANGS, t
from src.core.preset import available, load_preset


def presets_keyboard(current: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for name in available():
        try:
            title = load_preset(name).title or name
        except Exception:  # noqa: BLE001 — битый пресет не должен ронять меню
            title = name
        mark = "● " if name == current else "○ "
        kb.button(text=f"{mark}{title}", callback_data=f"preset:{name}")
    kb.adjust(2)
    return kb.as_markup()


def result_keyboard(job_id: str, current: str, lang: str | None = None) -> InlineKeyboardMarkup:
    """После готового ролика: перерендер тем же EDL, но другим стилем (§0.4)."""
    kb = InlineKeyboardBuilder()
    for name in available():
        if name == current:
            continue
        try:
            title = load_preset(name).title or name
        except Exception:  # noqa: BLE001
            title = name
        kb.button(
            text=t("video.rebuild_button", lang, title=title),
            callback_data=f"rerender:{job_id}:{name}",
        )
    kb.adjust(1)
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
