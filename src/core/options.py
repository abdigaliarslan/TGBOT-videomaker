"""Опции одного монтажа — то, что пользователь выбирает перед загрузкой.

Сохраняются в задаче целиком (`jobs.options`) и читаются воркером: чат и Mini
App ставят задачи одним путём, а различаются только набором галочек.

Здесь нет ни одной опции, которую пайплайн не умеет выполнить. Экран-образец
предлагал ещё «хук», «эмодзи в субтитрах» и «плашки»: первых двух в пайплайне
нет (режиссёр хук находит, но не сочиняет, и эмодзи не расставляет), а плашки
сняты с субтитра по решению владельца. Галочка, которая ничего не меняет, хуже
отсутствующей.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.preset import Preset

# Цвет выделенного слова. Ключ — то, что хранится в задаче; значение — подписи
# на языках интерфейса и цвет. Порядок словаря — порядок чипов в приложении. `None` у цвета значит
# «как задумано темой»: у glass это приглушённое золото, у bold — салатовый.
ACCENTS: dict[str, tuple[dict[str, str], str | None]] = {
    "theme": ({"ru": "Как в теме", "en": "Theme default"}, None),
    "gold": ({"ru": "Золото", "en": "Gold"}, "#F3D28F"),
    "turquoise": ({"ru": "Бирюза", "en": "Turquoise"}, "#3DDBD9"),
    "violet": ({"ru": "Фиолет", "en": "Violet"}, "#A38BFF"),
    "neon_pink": ({"ru": "Розовый неон", "en": "Neon pink"}, "#FF4FD8"),
    "lime": ({"ru": "Лайм", "en": "Lime"}, "#C8F560"),
    "coral": ({"ru": "Коралл", "en": "Coral"}, "#FF7A59"),
}


class JobOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    accent: str = "theme"
    cut: bool = Field(default=True, description="резать затяжные паузы и заминки")
    sfx: bool = Field(default=True, description="звуковые акценты на картинках и укрупнениях")
    broll: bool = Field(default=True, description="медиа-карточки по смыслу")
    camera: bool = Field(default=True, description="смена крупности на резах")

    @field_validator("accent")
    @classmethod
    def _known_accent(cls, v: str) -> str:
        return v if v in ACCENTS else "theme"

    @classmethod
    def from_stored(cls, raw: dict[str, Any] | None) -> "JobOptions":
        """Опции из БД. Старые задачи опций не несут — это поведение по умолчанию."""
        return cls.model_validate(raw or {})


def apply_accent(preset: Preset, accent: str) -> Preset:
    """Пресет с перекрашенным выделенным словом. Исходный объект не трогается.

    Крик красится тем же цветом: выбранный цвет — это «цвет выделения» целиком,
    и второй, чужой ему цвет на пике ролика выглядит как ошибка выбора.
    """
    color = ACCENTS.get(accent, ACCENTS["theme"])[1]
    if color is None:
        return preset
    clone = preset.model_copy(deep=True)
    clone.colors.accent = color
    clone.colors.shout = color
    return clone
