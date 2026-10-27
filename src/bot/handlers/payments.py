"""Оплата звёздами: подтверждение перед списанием и зачисление после.

Счёт выставляет Mini App (`/api/invoice`), а деньги приходят сюда, в бота: так
устроены платежи Telegram. Зачисление идёт ТОЛЬКО по `successful_payment` —
ответ клиента «оплачено» ничего не зачисляет, его можно подделать.
"""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.types import Message, PreCheckoutQuery

from src.core.config import settings
from src.core.i18n import lang_of, t, videos_word
from src.db import repo

log = logging.getLogger(__name__)
router = Router(name="payments")


def parse_payload(payload: str) -> int | None:
    """`videos:N` → N. Любой другой формат — не наш счёт."""
    kind, _, raw = payload.partition(":")
    if kind != "videos":
        return None
    try:
        videos = int(raw)
    except ValueError:
        return None
    return videos if videos > 0 else None


def check_invoice(payload: str, currency: str, amount: int, packages: dict[int, int]) -> str | None:
    """Ключ причины отказа (`core.i18n`) или None, если счёт можно оплачивать.

    Цена сверяется с ТЕКУЩИМИ пакетами: счёт, выставленный до смены цен, иначе
    продал бы пакет по старой цене. Звёзды — единственная валюта.
    """
    videos = parse_payload(payload)
    if videos is None:
        return "pay.bad_payload"
    if currency != "XTR":
        return "pay.stars_only"
    price = packages.get(videos)
    if price is None or price != amount:
        return "pay.price_changed"
    return None


@router.pre_checkout_query()
async def on_pre_checkout(query: PreCheckoutQuery) -> None:
    reason = check_invoice(
        query.invoice_payload, query.currency, query.total_amount, settings.star_packages
    )
    if reason:
        log.warning("оплата отклонена (%s): %s", query.from_user.id, reason)
        db_user = await repo.get_user(query.from_user.id)
        lang = lang_of(db_user) if db_user else lang_of(query.from_user)
        await query.answer(ok=False, error_message=t(reason, lang))
        return
    await query.answer(ok=True)


@router.message(F.successful_payment)
async def on_paid(message: Message) -> None:
    paid = message.successful_payment
    user = message.from_user
    if paid is None or user is None:
        return
    videos = parse_payload(paid.invoice_payload)
    if videos is None:
        # Деньги уже списаны, а за что — не понять. Молча это оставлять нельзя.
        log.error("оплата без распознанного пакета: user=%s payload=%r charge=%s",
                  user.id, paid.invoice_payload, paid.telegram_payment_charge_id)
        await message.answer(t("pay.unrecognized", lang_of(user)))
        return

    db_user = await repo.upsert_user(
        user.id, username=user.username, first_name=user.first_name,
        language_code=user.language_code, is_admin=user.id in settings.admin_ids,
    )
    credited = await repo.credit_payment(
        user.id,
        charge_id=paid.telegram_payment_charge_id,
        stars=paid.total_amount,
        videos=videos,
        payload=paid.invoice_payload,
    )
    if not credited:
        log.info("повтор оплаты %s — уже зачислена", paid.telegram_payment_charge_id)
        return
    fresh = await repo.get_user(user.id)
    log.info("оплата: user=%s +%d роликов за %d XTR", user.id, videos, paid.total_amount)
    lang = lang_of(db_user)
    await message.answer(
        t("pay.credited", lang, videos=videos, word=videos_word(videos, lang),
          balance=fresh.balance if fresh else videos)
    )
