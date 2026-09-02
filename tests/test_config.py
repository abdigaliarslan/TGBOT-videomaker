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


def test_llm_ready_requires_key_and_flag():
    assert Settings(anthropic_api_key="", llm_enabled=True).llm_ready is False
    assert Settings(anthropic_api_key="k", llm_enabled=False).llm_ready is False
    assert Settings(anthropic_api_key="k", llm_enabled=True).llm_ready is True


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
