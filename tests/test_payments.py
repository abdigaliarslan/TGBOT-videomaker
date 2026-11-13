"""Проверка счёта перед оплатой: что можно оплачивать, а что нет."""

from __future__ import annotations

import pytest

from src.bot.handlers.payments import check_invoice, parse_payload

# Токены → звёзды, как в `Settings.token_packages`.
PACKAGES = {240: 749, 590: 1899, 1170: 3599}


@pytest.mark.parametrize(
    "payload,expected",
    [("tokens:240", 240), ("tokens:0", None), ("tokens:-1", None), ("tokens:x", None),
     ("stars:5", None), ("", None),
     # Счёт старого формата «videos:N» выставлен до перехода на токены: оплачивать
     # его нельзя — N роликов за звёзды по старой цене.
     ("videos:5", None)],
)
def test_payload_parsing(payload: str, expected: int | None):
    assert parse_payload(payload) == expected


def test_a_current_package_at_its_price_is_payable():
    assert check_invoice("tokens:590", "XTR", 1899, PACKAGES) is None


def test_an_invoice_issued_before_a_price_change_is_refused():
    """Счёт с прошлой ценой не должен продать пакет дешевле или дороже нынешнего."""
    assert check_invoice("tokens:590", "XTR", 1500, PACKAGES) == "pay.price_changed"


def test_a_package_that_no_longer_exists_is_refused():
    assert check_invoice("tokens:300", "XTR", 749, PACKAGES) == "pay.price_changed"


def test_an_old_videos_invoice_is_refused():
    assert check_invoice("videos:5", "XTR", 200, PACKAGES) == "pay.bad_payload"


def test_only_stars_are_accepted():
    assert check_invoice("tokens:240", "USD", 749, PACKAGES) == "pay.stars_only"


def test_every_refusal_has_both_translations():
    """Отказ показывается человеку — у каждого ключа обязан быть и русский, и английский."""
    from src.core.i18n import TEXTS

    for key in ("pay.bad_payload", "pay.stars_only", "pay.price_changed"):
        assert set(TEXTS[key]) == {"ru", "en"}


def test_the_default_packages_are_the_owners():
    """Как у Brandly: 240 / 590 / 1170 токенов; цены владельца в звёздах."""
    from src.core.config import Settings

    cfg = Settings(_env_file=None)
    assert [(p.tokens, p.stars, p.rub) for p in cfg.token_packages] == [
        (240, 749, 1099), (590, 1899, 2799), (1170, 3599, 5299),
    ]
    assert [p.popular for p in cfg.token_packages] == [False, True, False]
    # 23 токена за ролик — при нём сходятся подписи «до 10 / 25 / 50 роликов».
    assert cfg.tokens_per_video == 23
    assert [p.tokens // cfg.tokens_per_video for p in cfg.token_packages] == [10, 25, 50]
    assert cfg.trial_tokens == cfg.trial_videos * 23


def test_packages_come_from_a_json_list():
    from src.core.config import Settings

    cfg = Settings(_env_file=None, token_packages='[{"tokens": 590, "stars": 10}, {"tokens": 240, "stars": 5}]')
    assert [p.tokens for p in cfg.token_packages] == [240, 590]
    assert cfg.package(590).stars == 10 and cfg.package(1) is None
