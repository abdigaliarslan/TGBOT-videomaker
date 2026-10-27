"""Проверка счёта перед оплатой: что можно оплачивать, а что нет."""

from __future__ import annotations

import pytest

from src.bot.handlers.payments import check_invoice, parse_payload

PACKAGES = {1: 50, 5: 200, 20: 600}


@pytest.mark.parametrize(
    "payload,expected",
    [("videos:5", 5), ("videos:0", None), ("videos:-1", None), ("videos:x", None), ("stars:5", None), ("", None)],
)
def test_payload_parsing(payload: str, expected: int | None):
    assert parse_payload(payload) == expected


def test_a_current_package_at_its_price_is_payable():
    assert check_invoice("videos:5", "XTR", 200, PACKAGES) is None


def test_an_invoice_issued_before_a_price_change_is_refused():
    """Счёт с прошлой ценой не должен продать пакет дешевле или дороже нынешнего."""
    assert check_invoice("videos:5", "XTR", 150, PACKAGES) == "pay.price_changed"


def test_a_package_that_no_longer_exists_is_refused():
    assert check_invoice("videos:3", "XTR", 150, PACKAGES) == "pay.price_changed"


def test_only_stars_are_accepted():
    assert check_invoice("videos:1", "USD", 50, PACKAGES) == "pay.stars_only"


def test_every_refusal_has_both_translations():
    """Отказ показывается человеку — у каждого ключа обязан быть и русский, и английский."""
    from src.core.i18n import TEXTS

    for key in ("pay.bad_payload", "pay.stars_only", "pay.price_changed"):
        assert set(TEXTS[key]) == {"ru", "en"}
