"""Приглашения: код в ссылке, правила привязки и бонус с оплаты.

Привязка и бонус — на настоящем Postgres, как и остальные денежные пути
(`test_money.py`): бонус — это ролики на балансе, а держится он на условиях
UPDATE и на транзакции оплаты, которых двойником не изобразить.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from src.bot.handlers.payments import share_text
from src.core.referral import code_of, parse_code
from src.db import repo

# ------------------------------------------------------------------ код


@pytest.mark.parametrize("user_id", [1, 35, 36, 777000, 5_123_456_789, 2**62])
def test_a_code_reads_back_as_the_same_user(user_id: int):
    code = code_of(user_id)
    assert code.startswith("r") and len(code) <= 14
    assert parse_code(code) == user_id
    assert parse_code(code.upper()) == user_id, "Telegram не меняет регистр, но человек может"


@pytest.mark.parametrize("junk", [None, "", "r", "abc", "r-1", "r" + "z" * 14, "ref_12", " r ", "r0"])
def test_anything_that_is_not_a_code_is_ignored(junk):
    assert parse_code(junk) is None


def test_a_bonus_reads_like_a_person_wrote_it():
    assert share_text(50, "ru") == "0,5 токена"
    assert share_text(2400, "ru") == "24 токена"
    assert share_text(100, "ru") == "1 токен"
    assert share_text(500, "ru") == "5 токенов"
    assert share_text(10, "en") == "0.1 of a token"
    assert share_text(150, "en") == "1.5 tokens"


# ------------------------------------------------------------------ база


@pytest.fixture
async def people(db):
    """Пригласивший 1, друг 2, посторонний 3."""
    for uid in (1, 2, 3):
        await repo.upsert_user(uid, first_name=f"U{uid}", language_code="ru")


async def balance(uid: int) -> tuple[int, int]:
    u = await repo.get_user(uid)
    assert u is not None
    return u.balance, u.ref_credit


async def pay(uid: int, tokens: int, charge: str, percent: int = 10) -> bool:
    """Оплата пакета; суммы в тестах ниже — в токенах, бонус — в сотых токена."""
    return await repo.credit_payment(
        uid, charge_id=charge, stars=tokens * 3, tokens=tokens, payload=f"tokens:{tokens}",
        referral_percent=percent,
    )


async def test_a_friend_is_attached_once(people):
    assert await repo.attach_referrer(2, 1) is True
    assert await repo.attach_referrer(2, 3) is False, "приглашение переписали на другого"
    assert (await repo.get_user(2)).referrer_id == 1


async def test_nobody_invites_themselves(people):
    assert await repo.attach_referrer(1, 1) is False


async def test_two_accounts_cannot_invite_each_other(people):
    assert await repo.attach_referrer(2, 1) is True
    assert await repo.attach_referrer(1, 2) is False, "кольцо из двух аккаунтов"


async def test_a_code_of_a_stranger_who_never_came_attaches_nothing(people):
    assert await repo.attach_referrer(2, 999) is False


async def test_someone_who_already_paid_cannot_be_invited(people):
    await pay(2, 1, "early")
    assert await repo.attach_referrer(2, 1) is False


async def test_a_bonus_builds_up_in_hundredths_and_lands_as_whole_videos(people):
    await repo.attach_referrer(2, 1)
    await pay(2, 5, "a")  # 50 сотых
    assert await balance(1) == (0, 50)
    await pay(2, 5, "b")  # ещё 50 — целый ролик
    assert await balance(1) == (1, 0)
    await pay(2, 20, "c")  # 200 сотых — два ролика
    assert await balance(1) == (3, 0)
    assert await balance(2) == (30, 0), "сам покупатель получил не свой пакет"


async def test_a_repeated_payment_update_pays_no_second_bonus(people):
    await repo.attach_referrer(2, 1)
    assert await pay(2, 20, "same") is True
    assert await pay(2, 20, "same") is False
    assert await balance(1) == (2, 0)


async def test_payments_without_an_inviter_give_nobody_anything(people):
    await pay(2, 20, "solo")
    assert await balance(1) == (0, 0)
    assert await balance(3) == (0, 0)


async def test_two_friends_paying_at_once_lose_no_share(people):
    """Копилка считается одним UPDATE из старого значения: доли не теряются."""
    await repo.attach_referrer(2, 1)
    await repo.attach_referrer(3, 1)
    await asyncio.gather(*(pay(2 + i % 2, 5, f"x{i}") for i in range(6)))
    # 6 × 50 сотых = 3 ролика ровно
    assert await balance(1) == (3, 0)


async def test_stats_split_today_from_before(people):
    await repo.attach_referrer(2, 1)
    await repo.attach_referrer(3, 1)
    async with repo.session() as s:
        await s.execute(text("update users set referred_at = now() - interval '3 days' where id = 3"))
    await pay(2, 5, "today")
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    stats = await repo.referral_stats(1, since=since)
    assert stats == {
        "invited": 2, "invited_today": 1, "paid": 1, "paid_today": 1, "earned": 50, "earned_today": 50,
    }
    friends = await repo.referral_friends(1)
    assert [f["name"] for f in friends] == ["U2", "U3"]
    assert friends[0]["paid"] and friends[0]["earned"] == 50
    assert not friends[1]["paid"]
    assert [f["name"] for f in await repo.referral_friends(1, since=since)] == ["U2"]


async def test_the_inviter_sees_the_bonus_in_their_history(people):
    await repo.attach_referrer(2, 1)
    await pay(2, 5, "h")
    assert [b[1] for b in await repo.referral_bonuses(1)] == [50]
    assert await repo.payment_bonus("h") == (1, 50)
    assert await repo.user_payments(1) == []
