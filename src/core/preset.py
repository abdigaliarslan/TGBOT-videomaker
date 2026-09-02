"""Пресет стиля — данные, а не код (§0.3).

Один и тот же JSON читают оба рендерера: ASS-генератор (M3) и Remotion (M5).
Добавление стиля не требует правки компонентов — только новый файл в presets/.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PRESETS_DIR = Path(__file__).resolve().parent / "presets"


class FontSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str = "Montserrat ExtraBold"
    # Имя файла в assets/fonts — рендер обязан ставить шрифт сам, иначе Chromium
    # подставит дефолт (§5 M5 ⚠).
    file: str | None = None
    weight: int = 800
    size: int = 84  # px при высоте кадра 1920
    uppercase: bool = False
    letter_spacing: float = -0.01  # em


class Colors(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active: str = "#FFFFFF"  # слово, которое звучит сейчас
    past: str = "#FFFFFF"  # уже прозвучало
    ghost: str = "#FFFFFF"  # ещё не прозвучало
    past_opacity: float = 0.72
    ghost_opacity: float = 0.34
    accent: str = "#FFD54A"
    shout: str = "#FF6A3D"
    outline: str = "#000000"
    shadow: str = "#000000"


class Panel(BaseModel):
    """Glass-панель под текстом. Контейнер растёт под контент сам (M5)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    color: str = "#0B0F14"
    opacity: float = 0.34
    blur: int = 18
    radius: int = 30
    padding_x: int = 34
    padding_y: int = 22
    border_color: str = "#FFFFFF"
    border_opacity: float = 0.12
    border_width: int = 1


class Caption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: Literal["top", "center", "bottom"] = "bottom"
    # Отступ от края кадра в пикселях при высоте 1920.
    margin: int = 360
    max_line_chars: int = 15
    line_height: float = 1.06
    align: Literal["center", "left"] = "center"


class Animation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Появление слова: пружина (Remotion) / линейный масштаб (ASS).
    pop_scale: float = 1.16
    pop_ms: int = 160
    spring_damping: float = 12.0
    spring_stiffness: float = 190.0
    spring_mass: float = 0.7
    block_in_ms: int = 180
    block_out_ms: int = 140
    accent_scale: float = 1.18
    shout_scale: float = 1.34
    shout_rotate: float = -4.0  # градусы


class Preset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    title: str = ""
    font: FontSpec = Field(default_factory=FontSpec)
    colors: Colors = Field(default_factory=Colors)
    panel: Panel = Field(default_factory=Panel)
    caption: Caption = Field(default_factory=Caption)
    animation: Animation = Field(default_factory=Animation)
    background: str = "#000000"

    def scaled(self, height: int) -> "Preset":
        """Размеры заданы для 1920 по высоте; под другой кадр — пропорционально."""
        if height == 1920:
            return self
        k = height / 1920
        clone = self.model_copy(deep=True)
        clone.font.size = max(12, round(self.font.size * k))
        clone.caption.margin = round(self.caption.margin * k)
        clone.panel.padding_x = round(self.panel.padding_x * k)
        clone.panel.padding_y = round(self.panel.padding_y * k)
        clone.panel.radius = round(self.panel.radius * k)
        clone.panel.blur = round(self.panel.blur * k)
        return clone


def available() -> list[str]:
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))


@lru_cache(maxsize=32)
def load_preset(name: str) -> Preset:
    path = PRESETS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"нет пресета {name!r}; доступны: {', '.join(available()) or '—'}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("name", name)
    return Preset(**data)


# ------------------------------------------------------------------ цвета


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) not in (6, 8):
        raise ValueError(f"нераспознанный цвет: {value!r}")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def to_ass_colour(value: str, opacity: float = 1.0) -> str:
    """#RRGGBB + непрозрачность → &HAABBGGRR (в ASS альфа инвертирована)."""
    r, g, b = hex_to_rgb(value)
    alpha = max(0, min(255, round((1.0 - max(0.0, min(1.0, opacity))) * 255)))
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def to_ass_alpha(opacity: float) -> str:
    alpha = max(0, min(255, round((1.0 - max(0.0, min(1.0, opacity))) * 255)))
    return f"&H{alpha:02X}&"


def to_css_rgba(value: str, opacity: float = 1.0) -> str:
    r, g, b = hex_to_rgb(value)
    return f"rgba({r}, {g}, {b}, {round(max(0.0, min(1.0, opacity)), 3)})"
