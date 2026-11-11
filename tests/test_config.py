"""Тесты конфигурации, включая перевод путей локального Bot API (M0 ⚠)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import Settings
from src.core.preset import (
    Preset,
    available,
    hex_to_rgb,
    load_preset,
    to_ass_alpha,
    to_ass_colour,
    to_css_rgba,
)


# ------------------------------------------------------------------ Bot API пути


def cfg(tmp: Path) -> Settings:
    return Settings(
        bot_api_data_dir_host=tmp / "host",
        bot_api_data_dir_container="/var/lib/telegram-bot-api",
    )


def test_container_path_becomes_host_path(tmp_path):
    """Локальный Bot API отдаёт путь внутри своего контейнера, а не URL."""
    s = cfg(tmp_path)
    got = s.bot_api_path_to_local("/var/lib/telegram-bot-api/123456/videos/file_1.mp4")
    assert got == (tmp_path / "host" / "123456" / "videos" / "file_1.mp4").resolve()


def test_foreign_path_is_returned_unchanged(tmp_path):
    s = cfg(tmp_path)
    assert s.bot_api_path_to_local("videos/file_1.mp4") == Path("videos/file_1.mp4")


def test_cloud_relative_path_survives(tmp_path):
    """У облачного Bot API file_path относительный — его трогать нельзя."""
    s = cfg(tmp_path)
    assert str(s.bot_api_path_to_local("documents/file_5.mp4")) == "documents/file_5.mp4"


# ------------------------------------------------------------------ прочая конфигурация


def test_admin_ids_parsed_from_json_string():
    assert Settings(admin_ids="[1, 2, 3]").admin_ids == [1, 2, 3]


def test_admin_ids_parsed_from_bare_list():
    assert Settings(admin_ids="1,2 3").admin_ids == [1, 2, 3]


def test_admin_ids_empty_string_is_empty_list():
    assert Settings(admin_ids="").admin_ids == []


def test_blank_language_means_autodetect():
    assert Settings(whisper_language="  ").whisper_language == ""


def test_max_video_seconds_derived():
    assert Settings(max_video_minutes=2.5).max_video_seconds == 150.0


def test_the_shipped_provider_is_the_one_that_answers():
    """Дефект №1: LLM_PROVIDER=fal жил только в .env.

    .env не в git и исключён из rsync выкладки (deploy/update.sh) — провайдер,
    прописанный там, в прод не попадает вовсе. Умолчание обязано быть в коде.
    """
    assert Settings(_env_file=None).llm_provider == "fal"


def test_gateway_refuses_a_tier_override_without_a_vendor_prefix():
    """Дефект №2: `gemini-3.5-flash` вместо `google/gemini-3.5-flash`.

    Шлюз отвечает на такое имя 400, 400 не ретраится — и ролик молча выходит
    без акцентов и без вставок, а бот докладывает «готово».
    """
    import pydantic

    with pytest.raises(pydantic.ValidationError) as e:
        Settings(_env_file=None, llm_provider="fal", llm_tier_smart="gemini-3.5-flash")
    # Отказ обязан называть переменную и готовую правку: его читает оператор.
    assert "LLM_TIER_SMART" in str(e.value)
    assert "google/gemini-3.5-flash" in str(e.value)


@pytest.mark.parametrize("tier", ["llm_tier_cheap", "llm_tier_smart", "llm_tier_vision"])
def test_every_tier_override_is_checked_not_just_smart(tier):
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        Settings(_env_file=None, llm_provider="fal", **{tier: "gemini-3.5-flash"})
    ok = Settings(_env_file=None, llm_provider="fal", **{tier: "google/gemini-3.5-flash"})
    assert ok.model_for_tier(tier.removeprefix("llm_tier_")) == "google/gemini-3.5-flash"


def test_empty_override_is_fine_for_a_gateway():
    """Пусто — это не «без префикса»: берётся модель по умолчанию, она с вендором."""
    s = Settings(_env_file=None, llm_provider="fal", llm_tier_smart="", llm_tier_vision="  ")
    assert s.model_for_tier("smart").startswith("google/")
    assert s.model_for_tier("vision").startswith("google/")


@pytest.mark.parametrize("provider", ["anthropic", "deepseek", "gemini"])
def test_direct_providers_still_take_bare_model_names(provider):
    """У прямых провайдеров вендора в имени нет и быть не должно."""
    s = Settings(_env_file=None, llm_provider=provider, llm_tier_smart="какая-то-модель")
    assert s.model_for_tier("smart") == "какая-то-модель"


def test_llm_ready_requires_key_of_the_selected_provider():
    def s(**kw):
        return Settings(_env_file=None, llm_provider="anthropic", **kw)

    assert s(anthropic_api_key="", llm_enabled=True).llm_ready is False
    assert s(anthropic_api_key="k", llm_enabled=False).llm_ready is False
    assert s(anthropic_api_key="k", llm_enabled=True).llm_ready is True
    # ключ другого провайдера не считается
    assert Settings(_env_file=None, llm_provider="deepseek", anthropic_api_key="k").llm_ready is False


# ------------------------------------------------------------------ пресеты


def test_all_shipped_presets_load():
    assert available()
    for name in available():
        p = load_preset(name)
        assert isinstance(p, Preset)
        assert p.name == name
        assert p.font.size > 0


def test_preset_rejects_unknown_fields():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        Preset(name="x", unknown_field=1)


def test_missing_preset_lists_available():
    with pytest.raises(FileNotFoundError) as e:
        load_preset("no-such-preset")
    assert "glass" in str(e.value)


def test_scaling_is_proportional():
    p = load_preset("glass")
    half = p.scaled(960)
    assert half.font.size == round(p.font.size * 0.5)
    assert half.caption.margin == round(p.caption.margin * 0.5)


def test_scaling_at_native_height_is_identity():
    p = load_preset("glass")
    assert p.scaled(1920) is p


# ------------------------------------------------------------------ цвета


def test_hex_parsing():
    assert hex_to_rgb("#FF8000") == (255, 128, 0)
    assert hex_to_rgb("f80") == (255, 136, 0)


def test_bad_colour_raises():
    with pytest.raises(ValueError):
        hex_to_rgb("#12345")


def test_ass_colour_is_bgr_with_inverted_alpha():
    # ASS: &HAABBGGRR, альфа 00 = непрозрачно
    assert to_ass_colour("#FF8000") == "&H000080FF"
    assert to_ass_colour("#FFFFFF", 0.0) == "&HFFFFFFFF"


def test_ass_alpha_inverted():
    assert to_ass_alpha(1.0) == "&H00&"
    assert to_ass_alpha(0.0) == "&HFF&"


def test_css_rgba():
    assert to_css_rgba("#FF8000", 0.5) == "rgba(255, 128, 0, 0.5)"


# ------------------------------------------------------------------ тема по умолчанию
#
# Дефект №5: умолчание темы было зашито в трёх местах («glass»), и смена
# DEFAULT_PRESET до нового пользователя не доходила.


def test_shipped_default_theme_is_pulse():
    assert Settings(_env_file=None).default_preset == "pulse"
    assert Settings(_env_file=None).default_preset in available()


def test_new_user_is_created_with_the_configured_theme(monkeypatch):
    """`upsert_user` обязан сам положить тему: умолчания колонки ему не хватало."""
    from datetime import datetime, timezone

    from sqlalchemy.dialects import postgresql

    from src.db import repo

    monkeypatch.setattr(repo, "default_settings", Settings(_env_file=None, default_preset="bold"))
    stmt = repo.upsert_user_stmt(
        1, username=None, first_name=None, language_code=None, is_admin=False,
        now=datetime.now(timezone.utc),
    )
    compiled = stmt.compile(dialect=postgresql.dialect())
    assert compiled.params["preset"] == "bold"


def test_a_new_user_gets_the_trial_once_and_only_once(monkeypatch):
    """Проба кладётся на баланс при ПЕРВОЙ записи о человеке и больше никогда.

    В `set_` баланс стоять не должен: `set_` выполняется на каждом сообщении
    существующего пользователя, и проба либо обнуляла бы купленные ролики, либо
    выдавалась бы заново.
    """
    from datetime import datetime, timezone

    from sqlalchemy.dialects import postgresql

    from src.db import repo

    monkeypatch.setattr(repo, "default_settings", Settings(_env_file=None, trial_videos=1))
    stmt = repo.upsert_user_stmt(
        1, username=None, first_name=None, language_code=None, is_admin=False,
        now=datetime.now(timezone.utc),
    )
    compiled = stmt.compile(dialect=postgresql.dialect())
    assert compiled.params["balance"] == 1
    insert_part, rest = str(compiled).split("ON CONFLICT", 1)
    assert "balance" in insert_part
    assert "balance" not in rest.split("RETURNING", 1)[0]

    off = repo.upsert_user_stmt(
        1, username=None, first_name=None, language_code=None, is_admin=False,
        now=datetime.now(timezone.utc), trial=0,
    )
    assert off.compile(dialect=postgresql.dialect()).params["balance"] == 0


def test_every_theme_has_its_own_plain_label():
    """Подпись темы отвечает на вопрос «как это выглядит» — и обязана быть у
    каждой темы на обоих языках. Две темы с одной подписью неразличимы в чате,
    где превью нет вовсе."""
    from src.core.options import THEME_LABELS, theme_label
    from src.core.preset import available

    assert set(THEME_LABELS) == set(available())
    for lang in ("ru", "en"):
        labels = [theme_label(name, lang) for name in available()]
        assert all(labels) and len(set(labels)) == len(labels), lang
    assert theme_label("нет такой", "ru", "Запасное") == "Запасное"


def test_the_first_screen_states_the_terms(monkeypatch):
    """Условия стоят в приветствии: что бесплатно и сколько стоит следующий
    ролик. Раньше о цене человек узнавал отказом после отправки видео."""
    from src.bot.handlers import common

    monkeypatch.setattr(common.settings, "free_videos", False)
    monkeypatch.setattr(common.settings, "star_packages", {1: 50, 5: 200, 20: 600})
    monkeypatch.setattr(common.settings, "brand_name", "Reelmaker")

    monkeypatch.setattr(common.settings, "trial_videos", 1)
    text = common.welcome_text("ru")
    assert "Reelmaker" in text and "{brand}" not in text
    assert "Первый ролик — бесплатно" in text and "50 ⭐" in text and "от 30 ⭐" in text

    monkeypatch.setattr(common.settings, "trial_videos", 0)
    assert "бесплатно" not in common.offer_text("ru") and "50 ⭐" in common.offer_text("ru")

    monkeypatch.setattr(common.settings, "free_videos", True)
    assert "⭐" not in common.offer_text("ru")


def test_upsert_does_not_overwrite_the_theme_of_an_existing_user():
    """`set_` выполняется на каждом сообщении — тема из конфига затёрла бы выбор."""
    from datetime import datetime, timezone

    from sqlalchemy.dialects import postgresql

    from src.db.repo import upsert_user_stmt

    stmt = upsert_user_stmt(
        1, username="u", first_name="f", language_code="ru", is_admin=False,
        now=datetime.now(timezone.utc), preset="bold",
    )
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    insert_part, rest = sql.split("ON CONFLICT", 1)
    # RETURNING отрезаем: там перечислены все колонки, в том числе preset.
    update_part = rest.split("RETURNING", 1)[0]
    assert "preset" in insert_part
    assert "preset" not in update_part


def test_user_table_default_follows_config(monkeypatch):
    """Вставка мимо `upsert_user` (тесты, скрипты) тоже не должна давать glass."""
    from src.db import models

    monkeypatch.setattr(models, "default_settings", Settings(_env_file=None, default_preset="bold"))
    column_default = models.User.__table__.c.preset.default
    assert column_default.is_callable  # значение берётся на вставку, а не при импорте
    assert models.default_preset() == "bold"


def test_edl_without_an_explicit_theme_takes_it_from_config(monkeypatch):
    from src.core import edl as edl_mod

    monkeypatch.setattr(edl_mod, "default_settings", Settings(_env_file=None, default_preset="bold"))
    edl = edl_mod.EDL(source_duration=1.0, words=[], cuts=[], captions=[])
    assert edl.preset == "bold"
