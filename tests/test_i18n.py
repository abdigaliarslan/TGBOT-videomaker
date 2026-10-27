"""Язык интерфейса: каждая фраза есть на обоих языках и подставляет одно и то же."""

from __future__ import annotations

import json
import re
import string
import subprocess
from pathlib import Path

import pytest

from src.core.i18n import (
    ERROR_CODES, LANGS, STAGE_KEYS, TEXTS, error_text, guess_lang, lang_of, stage_title, t, videos_word,
)
from src.pipeline.job import STAGES

APP_JS = Path(__file__).resolve().parents[1] / "src" / "webapp" / "static" / "app.js"


def _fields(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


@pytest.mark.parametrize("key", sorted(TEXTS))
def test_every_phrase_exists_in_both_languages_with_the_same_placeholders(key: str):
    entry = TEXTS[key]
    assert set(entry) == set(LANGS), f"{key}: нет перевода"
    assert _fields(entry["ru"]) == _fields(entry["en"]), f"{key}: подстановки разошлись"


def test_every_pipeline_stage_and_error_code_has_a_phrase():
    """Новая стадия или код ошибки без фразы показали бы пользователю голый ключ."""
    assert {key for key, _, _ in STAGES} <= set(STAGE_KEYS)
    for key in STAGE_KEYS:
        assert f"stage.{key}" in TEXTS
    for code in ERROR_CODES:
        assert f"err.{code}" in TEXTS


def test_error_codes_raised_by_the_pipeline_are_known():
    """Код, поднятый в `JobError`, обязан быть в словаре, иначе фраза молча станет «failed»."""
    root = Path(__file__).resolve().parents[1] / "src"
    raised = set()
    for path in root.rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(r"raise JobError\(", src):
            # Вызов бывает многострочным: код — последний строковый литерал до «)» в конце вызова.
            call = src[m.end(): m.end() + 400].split("\n        )")[0].split(")\n")[0]
            codes = re.findall(r'"([a-z_]+)"\s*,?\s*$', call.strip())
            if codes:
                raised.add(codes[-1])
    assert raised, "не нашёл ни одного кода — регулярка сломалась"
    assert raised <= set(ERROR_CODES), raised - set(ERROR_CODES)


def test_translation_and_fallbacks():
    assert t("video.done", "en") == "Done."
    assert t("video.done", "de") == "Готово.", "неизвестный язык — русский, а не падение"
    assert t("no.such.key", "en") == "no.such.key"
    assert stage_title("asr", "en") == "Transcribing speech"
    assert stage_title("nope", "en") is None
    assert error_text("no_speech", "en", min_s=15, max_m=3).startswith("No speech")
    assert error_text("weird", "ru", min_s=15, max_m=3) == t("err.failed", "ru")
    assert "15" in error_text("too_short", "ru", min_s=15, max_m=3)


def test_the_language_is_the_users_choice_first_and_the_client_second():
    class U:
        def __init__(self, lang, code):
            self.lang, self.language_code = lang, code

    assert lang_of(U("en", "ru")) == "en", "выбор руками важнее языка клиента"
    assert lang_of(U(None, "de")) == "en"
    assert lang_of(U(None, "kk")) == "ru"
    assert lang_of(None) == "ru"
    assert guess_lang("pt-br") == "en"


@pytest.mark.parametrize("n,ru,en", [(1, "ролик", "video"), (3, "ролика", "videos"), (5, "роликов", "videos"), (11, "роликов", "videos"), (21, "ролик", "videos")])
def test_plurals(n, ru, en):
    assert videos_word(n, "ru") == ru
    assert videos_word(n, "en") == en


def test_the_app_dictionary_has_the_same_keys_in_both_languages():
    """Словарь приложения живёт в JS — сверяем его оттуда же, где он исполняется."""
    script = (
        "const src = require('fs').readFileSync(process.argv[1], 'utf8');"
        "const start = src.indexOf('const I18N = {');"
        "const end = src.indexOf('\\nconst LANG_NAMES');"
        "const I18N = eval('(' + src.slice(start + 'const I18N = '.length, end).trim().replace(/;$/, '') + ')');"
        "console.log(JSON.stringify({ru: Object.keys(I18N.ru), en: Object.keys(I18N.en)}));"
    )
    out = subprocess.run(["node", "-e", script, str(APP_JS)], capture_output=True, text=True, check=True)
    keys = json.loads(out.stdout)
    assert set(keys["ru"]) == set(keys["en"]), set(keys["ru"]) ^ set(keys["en"])

    html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
    used = set(re.findall(r'data-i18n(?:-aria)?="([^"]+)"', html))
    assert used <= set(keys["ru"]), used - set(keys["ru"])


@pytest.mark.parametrize(
    "text,action",
    [("🎬 Начать монтаж", "start"), ("💎 Баланс", "balance"), ("📊 Моя статистика", "stats"),
     ("💡 Как это работает", "help"), ("привет", None), (None, None)],
)
def test_the_old_reply_keyboard_buttons_are_recognised(text, action):
    """Старая клавиатура осталась в чатах; её кнопки обязаны что-то делать, а не молчать."""
    from src.bot.handlers.fallback import legacy_action

    assert legacy_action(text) == action
