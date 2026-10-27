"""Подпись initData: кто пришёл в Mini App."""

from __future__ import annotations

import hashlib
import hmac
import json
from urllib.parse import quote

import pytest

from src.webapp.auth import AuthError, sign_init_data, validate_init_data

TOKEN = "123456:TEST-token"
NOW = 1_800_000_000
USER = json.dumps({"id": 42, "first_name": "Арслан", "username": "arslan"}, ensure_ascii=False)


def fields(**over: str) -> dict[str, str]:
    base = {"auth_date": str(NOW - 60), "query_id": "AAE", "user": USER, "signature": "sig"}
    base.update(over)
    return base


def test_a_string_signed_by_telegram_is_accepted():
    user = validate_init_data(sign_init_data(fields(), TOKEN), TOKEN, max_age_s=3600, now=NOW)
    assert (user.id, user.first_name, user.username) == (42, "Арслан", "arslan")


def test_the_hash_follows_the_documented_recipe_not_our_own_signer():
    """Подпись собрана руками по документации, мимо `sign_init_data`.

    Иначе тест сверял бы подписчик сам с собой, и общая ошибка в понимании
    формулы проходила бы зелёной — а Telegram её не простит.
    """
    f = fields()
    check = "\n".join(f"{k}={f[k]}" for k in sorted(f))
    secret = hmac.new(b"WebAppData", TOKEN.encode(), hashlib.sha256).digest()
    digest = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    raw = "&".join(f"{k}={quote(v, safe='')}" for k, v in f.items()) + f"&hash={digest}"
    assert validate_init_data(raw, TOKEN, max_age_s=3600, now=NOW).id == 42


def test_a_forged_user_is_rejected():
    from urllib.parse import parse_qsl, urlencode

    signed = dict(parse_qsl(sign_init_data(fields(), TOKEN)))
    signed["user"] = signed["user"].replace('"id": 42', '"id": 43')
    assert '"id": 43' in signed["user"]
    forged = urlencode(signed)  # подпись старая, пользователь чужой
    with pytest.raises(AuthError, match="подпись"):
        validate_init_data(forged, TOKEN, max_age_s=3600, now=NOW)


def test_another_bots_signature_is_rejected():
    with pytest.raises(AuthError, match="подпись"):
        validate_init_data(sign_init_data(fields(), "999:other"), TOKEN, max_age_s=3600, now=NOW)


def test_a_stale_string_is_rejected():
    stale = sign_init_data(fields(auth_date=str(NOW - 7200)), TOKEN)
    with pytest.raises(AuthError, match="устарела"):
        validate_init_data(stale, TOKEN, max_age_s=3600, now=NOW)


@pytest.mark.parametrize("raw", ["", "auth_date=1&user=%7B%7D"])
def test_missing_data_or_hash_is_rejected(raw: str):
    with pytest.raises(AuthError):
        validate_init_data(raw, TOKEN, max_age_s=3600, now=NOW)


def test_a_server_without_token_refuses_instead_of_trusting():
    with pytest.raises(AuthError, match="токена"):
        validate_init_data(sign_init_data(fields(), ""), "", max_age_s=3600, now=NOW)
