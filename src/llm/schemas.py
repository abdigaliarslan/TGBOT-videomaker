"""JSON-схемы для structured output (§4).

Ключевой инвариант: ни одна схема не содержит поля со временем. Модель оперирует
ТОЛЬКО индексами слов (§0.1). Таймкоды считает Python из вывода ASR.

Схемы строгие: `additionalProperties: false` и закрытые enum — чтобы модель не
изобрела эффект, которого нет в рендерере.
"""

from __future__ import annotations

from typing import Any

# ------------------------------------------------------------------ direct (главный вызов)

DIRECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["blocks", "drop_words", "hook_end_word"],
    "properties": {
        "hook_end_word": {
            "type": ["integer", "null"],
            "description": (
                "Индекс последнего слова хука — первой фразы, которая держит зрителя. "
                "null, если явного хука нет."
            ),
        },
        "drop_words": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "Индексы слов-паразитов и оговорок на вырезание ('ну', 'эээ', "
                "повторы). Только слова, без которых смысл не меняется."
            ),
        },
        "blocks": {
            "type": "array",
            "minItems": 1,
            "description": (
                "Субтитры, разбитые на блоки по 3-5 слов. Каждое НЕ вырезанное "
                "слово должно попасть ровно в один блок, порядок сохранён."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["words", "accent_word", "accent", "position", "follow_face"],
                "properties": {
                    "words": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "items": {"type": "integer"},
                    },
                    "follow_face": {
                        "type": "boolean",
                        "description": (
                            "Блок едет за головой говорящего вместо того, чтобы "
                            "стоять на месте. Редкий приём для ОДНОГО смыслового "
                            "пика: не больше двух блоков на ролик, лишние мы "
                            "снимаем сами."
                        ),
                    },
                    "accent_word": {
                        "type": ["integer", "null"],
                        "description": (
                            "Индекс ГЛАВНОГО слова блока — того, что человек "
                            "выделил бы голосом. Нужно в каждом блоке: кегль "
                            "внутри блока это иерархия, главное слово набирается "
                            "крупнее соседей. null допустим, но тогда слово "
                            "выберет программа по длине, без разбора смысла."
                        ),
                    },
                    "accent": {
                        "type": ["string", "null"],
                        "enum": ["accent", "shout", None],
                    },
                    "position": {
                        "type": "string",
                        "enum": ["top", "center", "bottom", "auto"],
                    },
                },
            },
        },
    },
}

# ------------------------------------------------------------------ clean_transcript

CLEAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fixes"],
    "properties": {
        "fixes": {
            "type": "array",
            "description": (
                "Точечные правки распознавания по индексу. НЕ переписывание текста: "
                "только явные ошибки ASR в терминах, именах и числах."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["i", "fix"],
                "properties": {
                    "i": {"type": "integer"},
                    "fix": {"type": "string", "maxLength": 40},
                },
            },
        }
    },
}

# ------------------------------------------------------------------ plan_broll (vision)

BROLL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["word_idx", "duration", "query", "prompt", "anim", "anchor"],
                "properties": {
                    "word_idx": {"type": "integer"},
                    "duration": {"type": "number", "minimum": 0.8, "maximum": 3.0},
                    "query": {
                        "type": "string",
                        "description": "Короткий поисковый запрос НА АНГЛИЙСКОМ для стока.",
                        "maxLength": 60,
                    },
                    "prompt": {
                        "type": ["string", "null"],
                        "description": (
                            "Промпт для генерации, если стока не найдётся. Без текста и "
                            "надписей на картинке."
                        ),
                        "maxLength": 300,
                    },
                    "anim": {
                        "type": "string",
                        "enum": ["pop_in", "slide_up", "fullscreen_to_inset"],
                    },
                    "anchor": {
                        "type": "string",
                        "enum": [
                            "top_left",
                            "top_right",
                            "bottom_left",
                            "bottom_right",
                            "center",
                            "fullscreen",
                        ],
                    },
                },
            },
        }
    },
}

# ------------------------------------------------------------------ metadata

METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "description", "hashtags"],
    "properties": {
        "title": {"type": "string", "maxLength": 80},
        "description": {"type": "string", "maxLength": 400},
        "hashtags": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string", "maxLength": 30},
        },
    },
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "direct": DIRECT_SCHEMA,
    "clean_transcript": CLEAN_SCHEMA,
    "plan_broll": BROLL_SCHEMA,
    "metadata": METADATA_SCHEMA,
}

# Тир на вызов (§4). Меняется здесь, а не в вызывающем коде.
TIERS: dict[str, str] = {
    "clean_transcript": "cheap",
    "direct": "smart",
    "plan_broll": "vision",
    "metadata": "cheap",
}
