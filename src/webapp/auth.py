"""Кто пришёл в Mini App — по подписи `initData`, а не по словам клиента.

Приложение открыто в браузере Telegram, и любой запрос к API может прийти не
оттуда. Единственное, чему можно верить, — строка `initData`, которую Telegram
подписал ключом, выведенным из токена бота. Проверка ровно по документации
(core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app):

    secret = HMAC_SHA256(key="WebAppData", msg=bot_token)
    hash   = hex(HMAC_SHA256(key=secret, msg=data_check_string))

где `data_check_string` — все поля КРОМЕ `hash`, отсортированные по имени и
склеенные через перевод строки. Поле `signature` (подпись для третьих сторон)
в строку входит: исключается только `hash`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from urllib.parse import parse_qsl


class AuthError(Exception):
    """Подпись не сошлась, устарела или строки нет вовсе."""


@dataclass(frozen=True, slots=True)
class WebAppUser:
    id: int
    first_name: str | None = None
    username: str | None = None
    language_code: str | None = None
    photo_url: str | None = None


def _secret(bot_token: str) -> bytes:
    return hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()


def sign_init_data(fields: dict[str, str], bot_token: str) -> str:
    """Подписывает поля так же, как Telegram. Нужна тестам и ручной проверке API."""
    from urllib.parse import urlencode

    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    digest = hmac.new(_secret(bot_token), check.encode(), hashlib.sha256).hexdigest()
    return urlencode({**fields, "hash": digest})


def validate_init_data(
    init_data: str,
    bot_token: str,
    *,
    max_age_s: int,
    now: float | None = None,
) -> WebAppUser:
    if not init_data:
        raise AuthError("нет initData — приложение открыто не из Telegram")
    if not bot_token:
        raise AuthError("у сервера нет токена бота — проверить подпись нечем")

    fields = dict(parse_qsl(init_data, keep_blank_values=True, strict_parsing=False))
    received = fields.pop("hash", "")
    if not received:
        raise AuthError("в initData нет подписи")

    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    expected = hmac.new(_secret(bot_token), check.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, received):
        raise AuthError("подпись initData не сошлась")

    try:
        auth_date = int(fields.get("auth_date", "0"))
    except ValueError as e:
        raise AuthError("auth_date не число") from e
    age = (time.time() if now is None else now) - auth_date
    if auth_date <= 0 or age > max_age_s:
        raise AuthError("initData устарела — откройте приложение заново")

    try:
        raw = json.loads(fields.get("user", ""))
        return WebAppUser(
            id=int(raw["id"]),
            first_name=raw.get("first_name"),
            username=raw.get("username"),
            language_code=raw.get("language_code"),
            photo_url=raw.get("photo_url"),
        )
    except (ValueError, KeyError, TypeError) as e:
        raise AuthError("в initData нет пользователя") from e
