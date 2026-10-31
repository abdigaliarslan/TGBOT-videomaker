"""Тесты проверки формы ответа.

Нужна там, где провайдер не принуждает к схеме: без неё `{"blocks": "ой"}`
дошло бы до сборки EDL и упало там, где причину уже не видно.
"""

from __future__ import annotations

import pytest

from src.llm import jsonshape
from src.llm.schemas import CLEAN_SCHEMA, DIRECT_SCHEMA


def ok(value, schema) -> bool:
    return jsonshape.check(value, schema) == []


# ------------------------------------------------------------------ типы


@pytest.mark.parametrize(
    "value,schema,valid",
    [
        ("текст", {"type": "string"}, True),
        (5, {"type": "string"}, False),
        (5, {"type": "integer"}, True),
        (5.5, {"type": "integer"}, False),
        (5.5, {"type": "number"}, True),
        (True, {"type": "integer"}, False),  # bool это не число
        (True, {"type": "boolean"}, True),
        (None, {"type": "null"}, True),
        (None, {"type": ["integer", "null"]}, True),
        (7, {"type": ["integer", "null"]}, True),
        ("x", {"type": ["integer", "null"]}, False),
    ],
)
def test_type_checks(value, schema, valid):
    assert ok(value, schema) is valid


# ------------------------------------------------------------------ объекты


def test_required_field_missing():
    errs = jsonshape.check({}, {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}})
    assert any("обязательного поля" in e for e in errs)


def test_extra_field_when_closed():
    schema = {"type": "object", "additionalProperties": False, "properties": {"a": {"type": "integer"}}}
    assert any("лишнее поле" in e for e in jsonshape.check({"a": 1, "b": 2}, schema))


def test_nested_error_reports_path():
    schema = {
        "type": "object",
        "properties": {"inner": {"type": "object", "properties": {"n": {"type": "integer"}}}},
    }
    errs = jsonshape.check({"inner": {"n": "нет"}}, schema)
    assert errs and "$.inner.n" in errs[0]


# ------------------------------------------------------------------ массивы


def test_array_item_type():
    errs = jsonshape.check([1, "два"], {"type": "array", "items": {"type": "integer"}})
    assert errs and "[1]" in errs[0]


def test_array_bounds():
    schema = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 3}
    assert any("минимум" in e for e in jsonshape.check([1], schema))
    assert any("максимум" in e for e in jsonshape.check([1, 2, 3, 4], schema))


# ------------------------------------------------------------------ значения


def test_enum():
    assert any("не из набора" in e for e in jsonshape.check("rainbow", {"enum": ["accent", "shout"]}))


def test_numeric_bounds():
    schema = {"type": "number", "minimum": 1.0, "maximum": 1.25}
    assert any("больше максимума" in e for e in jsonshape.check(99.0, schema))
    assert any("меньше минимума" in e for e in jsonshape.check(0.5, schema))
    assert ok(1.1, schema)


def test_max_length():
    assert any("длина" in e for e in jsonshape.check("х" * 50, {"type": "string", "maxLength": 40}))


# ------------------------------------------------------------------ реальные схемы


def test_valid_direct_plan_passes():
    plan = {
        "hook_end_word": 3,
        "drop_words": [0, 7],
        "blocks": [{"words": [1, 2, 3], "register": "poster",
                    "accent_word": 2, "accent": "accent",
                    "position": "auto", "follow_face": False}],
    }
    assert ok(plan, DIRECT_SCHEMA)


def test_direct_plan_with_null_accent_passes():
    plan = {
        "hook_end_word": None,
        "drop_words": [],
        "blocks": [{"words": [0], "register": "poster",
                    "accent_word": None, "accent": None, "position": "auto", "follow_face": False}],
    }
    assert ok(plan, DIRECT_SCHEMA)


def test_blocks_as_string_is_caught():
    plan = {"hook_end_word": None, "drop_words": [], "blocks": "ой"}
    assert any("$.blocks" in e for e in jsonshape.check(plan, DIRECT_SCHEMA))


def test_invented_effect_is_caught():
    plan = {
        "hook_end_word": None, "drop_words": [],
        "blocks": [{"words": [0], "register": "poster",
                    "accent_word": 0, "accent": "радуга",
                    "position": "auto", "follow_face": False}],
    }
    assert any("не из набора" in e for e in jsonshape.check(plan, DIRECT_SCHEMA))


def test_too_many_words_in_block_is_caught():
    """Потолок схемы — потолок ТИХОГО регистра (8 слов, §1.4.1).

    Плакату больше пяти нельзя, но это связь между полями (длина × позиция ×
    акцент), а не форма одного поля: её держит `validate_edl`. Схема ловит
    только блок, который не бывает ни тем, ни другим.
    """
    plan = {
        "hook_end_word": None, "drop_words": [],
        "blocks": [{"words": list(range(9)), "register": "quiet",
                    "accent_word": None, "accent": None,
                    "position": "top", "follow_face": False}],
    }
    assert any("максимум" in e for e in jsonshape.check(plan, DIRECT_SCHEMA))


def test_a_quiet_length_block_fits_the_schema():
    """Восемь слов схему не нарушают: это тихий блок, а не перебор."""
    plan = {
        "hook_end_word": None, "drop_words": [],
        "blocks": [{"words": list(range(8)), "register": "quiet",
                    "accent_word": None, "accent": None,
                    "position": "top", "follow_face": False}],
    }
    assert jsonshape.check(plan, DIRECT_SCHEMA) == []


def test_clean_schema_accepts_empty_fixes():
    assert ok({"fixes": []}, CLEAN_SCHEMA)
