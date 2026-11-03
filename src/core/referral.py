"""Код приглашения в ссылке `t.me/<бот>?start=<код>`.

Код — id пользователя в base36 с префиксом `r`: короткий (`r1kq7zd` вместо
десяти цифр), без таблицы кодов и без коллизий. Префикс отделяет приглашение
от любых других параметров `/start`, которые появятся потом.

Тайны в коде нет и быть не должно: id в Telegram и так виден всем, с кем
человек переписывается, а подделать приглашение «от чужого имени» бессмысленно —
бонус получает тот, чей код, а не тот, кто его вставил.
"""

from __future__ import annotations

import re

PREFIX = "r"
_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"
_CODE = re.compile(rf"^{PREFIX}([0-9a-z]{{1,13}})$")


def code_of(user_id: int) -> str:
    if user_id <= 0:
        raise ValueError("id пользователя положительный")
    out = ""
    n = user_id
    while n:
        n, d = divmod(n, 36)
        out = _DIGITS[d] + out
    return PREFIX + out


def parse_code(payload: str | None) -> int | None:
    """Код → id пригласившего. Всё, что не похоже на код, — None, без исключений."""
    m = _CODE.match((payload or "").strip().lower())
    if not m:
        return None
    value = int(m.group(1), 36)
    return value if 0 < value < 2**63 else None


def share_of(videos: int, percent: int) -> int:
    """Бонус пригласившему в СОТЫХ ролика: 5 роликов × 10 % = 50 сотых."""
    return max(0, videos) * max(0, percent)
