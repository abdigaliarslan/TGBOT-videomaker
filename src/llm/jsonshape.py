"""Проверка ответа модели на соответствие JSON-схеме.

Нужна там, где провайдер не умеет принуждать к схеме (DeepSeek): промпт со
схемой — это просьба, а не гарантия. Без проверки формы `{"blocks": "ой"}`
дошло бы до сборки EDL и упало бы там, где причину уже не видно.

Покрыт ровно тот поднабор JSON Schema, который используется в `schemas.py`:
type (в том числе списком), required, properties, additionalProperties, items,
enum, minItems/maxItems, minimum/maximum, maxLength. Полноценный валидатор
здесь был бы лишней зависимостью.
"""

from __future__ import annotations

from typing import Any

TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    # bool — подкласс int в Python, но JSON-число и JSON-булево это разные вещи
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def check(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Возвращает список несоответствий; пустой список — форма верна."""
    errors: list[str] = []

    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not any(TYPE_CHECKS.get(t, lambda _v: True)(value) for t in types):
            got = "null" if value is None else type(value).__name__
            return [f"{path}: ожидался тип {'|'.join(types)}, пришёл {got}"]

    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(x) for x in schema["enum"])
        errors.append(f"{path}: значение {value!r} не из набора [{allowed}]")

    if isinstance(value, dict):
        props: dict[str, Any] = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                errors.append(f"{path}: нет обязательного поля {name!r}")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in props:
                    errors.append(f"{path}: лишнее поле {name!r}")
        for name, sub in props.items():
            if name in value:
                errors.extend(check(value[name], sub, f"{path}.{name}"))

    elif isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: элементов {len(value)}, минимум {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: элементов {len(value)}, максимум {schema['maxItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                errors.extend(check(item, item_schema, f"{path}[{i}]"))

    elif isinstance(value, str):
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: длина {len(value)}, максимум {schema['maxLength']}")

    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} меньше минимума {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} больше максимума {schema['maximum']}")

    return errors
