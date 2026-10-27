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
        # Стилевая памятка на весь ролик: по ней генерируются ВСЕ вставки, и
        # поэтому две картинки одного ролика выглядят комплектом, а не двумя
        # случайными находками. Пишется тем же вызовом, который и так смотрит
        # на кадры, — отдельного запроса и отдельных денег не стоит.
        #
        # Необязательна намеренно: модель без принуждения к схеме (DeepSeek)
        # может её не вернуть, и терять из-за этого весь план вставок нельзя.
        "style": {
            "type": "object",
            "additionalProperties": False,
            "required": ["palette", "light", "mood"],
            "properties": {
                "palette": {
                    "type": "string",
                    "description": "Палитра кадров ролика, НА АНГЛИЙСКОМ: 3-5 слов.",
                    "maxLength": 120,
                },
                "light": {
                    "type": "string",
                    "description": "Свет: направление, мягкость, температура. Английский.",
                    "maxLength": 120,
                },
                "mood": {
                    "type": "string",
                    "description": (
                        "Обработка кадра, НЕ место: фактура, зерно, контраст. "
                        "Английский."
                    ),
                    "maxLength": 120,
                },
            },
        },
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
                        "description": (
                            "Запасной поисковый запрос НА АНГЛИЙСКОМ для стока, 2-4 слова. "
                            "Нужен, только если генерация не удастся."
                        ),
                        "maxLength": 60,
                    },
                    "prompt": {
                        "type": ["string", "null"],
                        "description": (
                            "ГЛАВНОЕ поле: описание кадра НА АНГЛИЙСКОМ, по которому "
                            "картинка рисуется. Предметно, с композицией под вертикаль "
                            "9:16. Без текста и надписей на картинке."
                        ),
                        "maxLength": 600,
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

# ------------------------------------------------------------------ check_image (vision)

# Приёмка сгенерированной картинки. Двух полей достаточно: показано ли то, что
# просили, и нет ли на картинке букв. Второе спрашивается отдельно, потому что
# именно на нём модели врут чаще всего (кириллица — всегда, латиница — на
# вывесках и упаковках), а заметить это в готовом ролике уже поздно.
IMAGE_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["shows_subject", "has_text", "fix"],
    "properties": {
        "shows_subject": {"type": "boolean"},
        "has_text": {"type": "boolean"},
        "fix": {
            "type": ["string", "null"],
            "description": (
                "Если что-то не так — одна фраза НА АНГЛИЙСКОМ, что поправить в "
                "промпте. Иначе null."
            ),
            "maxLength": 200,
        },
    },
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "direct": DIRECT_SCHEMA,
    "clean_transcript": CLEAN_SCHEMA,
    "plan_broll": BROLL_SCHEMA,
    "check_image": IMAGE_CHECK_SCHEMA,
    "metadata": METADATA_SCHEMA,
}

# Тир на вызов (§4). Меняется здесь, а не в вызывающем коде.
TIERS: dict[str, str] = {
    "clean_transcript": "cheap",
    "direct": "smart",
    "plan_broll": "vision",
    "check_image": "vision",
    "metadata": "cheap",
}
