"""Опции монтажа и пакеты звёзд: то, что пользователь выбирает в Mini App."""

from __future__ import annotations

import pytest

from src.core.config import Settings
from src.core.options import ACCENTS, JobOptions, apply_accent
from src.core.preset import load_preset


def test_a_job_without_options_behaves_as_before():
    """Задачи из чата и из времени до Mini App опций не несут — всё включено."""
    o = JobOptions.from_stored(None)
    assert (o.accent, o.cut, o.sfx, o.broll, o.camera) == ("theme", True, True, True, True)


def test_an_unknown_accent_falls_back_to_the_theme():
    """Палитру в приложении могут урезать, а старые задачи в БД останутся.

    Упасть на такой задаче значило бы потерять оплаченный ролик из-за переименования
    цвета, поэтому неизвестный ключ читается как «цвет темы».
    """
    assert JobOptions.from_stored({"accent": "ultraviolet"}).accent == "theme"


def test_unknown_fields_from_a_newer_client_are_ignored():
    assert JobOptions.from_stored({"cut": False, "hook": True}).cut is False


@pytest.mark.parametrize("key", [k for k, (_, color) in ACCENTS.items() if color])
def test_the_accent_recolours_both_highlight_colours(key: str):
    glass = load_preset("glass")
    painted = apply_accent(glass, key)
    assert painted.colors.accent == painted.colors.shout == ACCENTS[key][1]
    # Исходный пресет — общий объект кеша, трогать его нельзя.
    assert glass.colors.accent == "#F3D28F"


def test_the_theme_accent_leaves_the_preset_untouched():
    bold = load_preset("bold")
    assert apply_accent(bold, "theme") is bold


def test_star_packages_parse_from_the_env_string():
    cfg = Settings(_env_file=None, star_packages='{"20": 600, "1": 50, "5": 200}')
    assert cfg.star_packages == {1: 50, 5: 200, 20: 600}
    assert list(cfg.star_packages) == [1, 5, 20], "пакеты идут по возрастанию"


@pytest.mark.parametrize("raw", ['{"0": 50}', '{"1": 0}', '{"-1": 50}'])
def test_a_free_or_empty_package_is_a_config_error(raw: str):
    with pytest.raises(ValueError):
        Settings(_env_file=None, star_packages=raw)
