"""Клавиатуры. Ничего, кроме разметки."""

from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

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


def result_keyboard(job_id: str, current: str) -> InlineKeyboardMarkup:
    """После готового ролика: перерендер тем же EDL, но другим стилем (§0.4)."""
    kb = InlineKeyboardBuilder()
    for name in available():
        if name == current:
            continue
        try:
            title = load_preset(name).title or name
        except Exception:  # noqa: BLE001
            title = name
        kb.button(text=f"Пересобрать в «{title}»", callback_data=f"rerender:{job_id}:{name}")
    kb.adjust(1)
    return kb.as_markup()
