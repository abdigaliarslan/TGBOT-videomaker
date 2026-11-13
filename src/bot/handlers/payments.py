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
from src.core.i18n import lang_of, t, tokens_text
from src.db import repo

log = logging.getLogger(__name__)
router = Router(name="payments")


def parse_payload(payload: str) -> int | None:
    """`tokens:N` → N токенов. Любой другой формат — не наш счёт.

    Прежний `videos:N` (до перехода на токены) тоже «не наш»: такой счёт продал
    бы N роликов по старой цене.
    """
    kind, _, raw = payload.partition(":")
    if kind != "tokens":
        return None
    try:
        tokens = int(raw)
    except ValueError:
        return None
    return tokens if tokens > 0 else None


def current_packages() -> dict[int, int]:
    """Токены → звёзды по нынешним пакетам."""
    return {p.tokens: p.stars for p in settings.token_packages}


def check_invoice(payload: str, currency: str, amount: int, packages: dict[int, int]) -> str | None:
    """Ключ причины отказа (`core.i18n`) или None, если счёт можно оплачивать.

    Цена сверяется с ТЕКУЩИМИ пакетами: счёт, выставленный до смены цен, иначе
    продал бы пакет по старой цене. Звёзды — единственная валюта.
    """
    tokens = parse_payload(payload)
    if tokens is None:
        return "pay.bad_payload"
    if currency != "XTR":
        return "pay.stars_only"
    price = packages.get(tokens)
    if price is None or price != amount:
        return "pay.price_changed"
    return None


@router.pre_checkout_query()
async def on_pre_checkout(query: PreCheckoutQuery) -> None:
    reason = check_invoice(
        query.invoice_payload, query.currency, query.total_amount, current_packages()
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
    tokens = parse_payload(paid.invoice_payload)
    if tokens is None:
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
        tokens=tokens,
        payload=paid.invoice_payload,
    )
    if not credited:
        log.info("повтор оплаты %s — уже зачислена", paid.telegram_payment_charge_id)
        return
    fresh = await repo.get_user(user.id)
    log.info("оплата: user=%s +%d токенов за %d XTR", user.id, tokens, paid.total_amount)
    lang = lang_of(db_user)
    await message.answer(
        t("pay.credited", lang, tokens=tokens_text(tokens, lang),
          balance=tokens_text(fresh.balance if fresh else tokens, lang))
    )
    await _thank_referrer(message.bot, paid.telegram_payment_charge_id)


def share_text(hundredths: int, lang: str) -> str:
    """Сотые токена словами: 2400 → «24 токена», 50 → «0,5 токена», 10 → «0.1 of a token».

    У дроби в русском родительный единственного («0,5 токена») при любом
    числе, а у целых — обычное согласование.
    """
    whole, part = divmod(hundredths, 100)
    if not part:
        return tokens_text(whole, lang)
    number = f"{whole}.{part:02d}".rstrip("0")
    if lang == "ru":
        return f"{number.replace('.', ',')} токена"
    return f"{number} {'of a token' if whole == 0 else 'tokens'}"


async def _thank_referrer(bot, charge_id: str) -> None:
    """Сказать пригласившему, что пришёл бонус. Зачислен он уже — это только весть."""
    try:
        got = await repo.payment_bonus(charge_id)
        if got is None:
            return
        referrer_id, share = got
        ref_user = await repo.get_user(referrer_id)
        if ref_user is None:
            return
        lang = lang_of(ref_user)
        await bot.send_message(
            referrer_id,
            t("ref.bonus", lang, share=share_text(share, lang),
              balance=tokens_text(ref_user.balance, lang)),
        )
    except Exception as e:  # noqa: BLE001 — человек мог заблокировать бота; деньги уже на месте
        log.info("весть о бонусе по оплате %s не ушла: %s", charge_id, e)
