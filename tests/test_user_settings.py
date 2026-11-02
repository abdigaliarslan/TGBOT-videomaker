"""Память выбора между роликами: тема, цвет выделения и шрифт.

Тема запоминалась всегда (`users.preset`), а цвет — нет: приложение каждый раз
брало умолчание. С появлением шрифта это стало заметнее — выбрать антикву и
получить обратно гротеск на следующем ролике неприятнее, чем потерять оттенок.

Колонка `users.settings` (JSONB) была объявлена в модели и не использовалась
ни одной строкой кода. Теперь в ней живут цвет и шрифт: миграция не нужна,
колонка уже есть.

Тесты идут по настоящему Postgres (фикстура `db` в conftest): слияние JSONB —
это поведение базы, а не питона, и двойником его не изобразить.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from src.db import repo


@pytest.fixture
async def user(db):
    return await repo.upsert_user(2002, username="u", first_name="У", language_code="ru")


async def test_a_fresh_user_has_no_remembered_choices(user):
    assert (await repo.get_user(2002)).settings == {}


async def test_the_choice_survives_until_the_next_reel(user):
    await repo.set_user_settings(2002, accent="red", font="serif")
    got = await repo.get_user(2002)
    assert got.settings == {"accent": "red", "font": "serif"}


async def test_a_later_choice_only_changes_what_it_names(user):
    """Слияние, а не замена: шрифт выбрали вчера, цвет — сегодня, и вчерашнее
    не должно пропасть."""
    await repo.set_user_settings(2002, accent="red", font="serif")
    await repo.set_user_settings(2002, accent="mint")
    assert (await repo.get_user(2002)).settings == {"accent": "mint", "font": "serif"}


async def test_nothing_to_save_touches_nothing(user):
    await repo.set_user_settings(2002, accent="red")
    await repo.set_user_settings(2002)
    assert (await repo.get_user(2002)).settings == {"accent": "red"}


async def test_the_choice_is_remembered_per_user(user):
    await repo.upsert_user(2003, username="v", first_name="В", language_code="ru")
    await repo.set_user_settings(2002, font="serif")
    await repo.set_user_settings(2003, font="narrow")
    assert (await repo.get_user(2002)).settings == {"font": "serif"}
    assert (await repo.get_user(2003)).settings == {"font": "narrow"}


async def test_two_saves_at_once_do_not_lose_one_another(user):
    """Два ролика подряд из приложения и из чата — запись идёт одним UPDATE."""
    import asyncio

    await asyncio.gather(
        repo.set_user_settings(2002, accent="red"),
        repo.set_user_settings(2002, font="serif"),
    )
    got = (await repo.get_user(2002)).settings
    assert set(got) == {"accent", "font"}, f"одна из записей потерялась: {got}"


async def test_an_unknown_user_is_not_created_by_a_save(db):
    """Запись настроек не имеет права создать пользователя из воздуха."""
    await repo.set_user_settings(999999, font="serif")
    assert await repo.get_user(999999) is None
    async with repo.session() as s:
        assert (await s.execute(text("select count(*) from users"))).scalar_one() == 0
