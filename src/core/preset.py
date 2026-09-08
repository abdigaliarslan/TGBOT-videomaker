"""Пресет стиля — данные, а не код (§0.3).

Один и тот же JSON читают оба рендерера: ASS-генератор (M3) и Remotion (M5).
Добавление стиля не требует правки компонентов — только новый файл в presets/.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    # Ф1: место под ещё не прозвучавшее слово резервируется, но самого слова не
    # видно — иначе вся реплика читается вперёд диктора и «пшик» подсвечивает не
    # появление слова, а смену его яркости. Ноль гасит слово, не убирая его из
    # потока: панель не меняет размер, когда слово зазвучит.
    ghost_opacity: float = 0.0
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


# Поля субтитра, которых в модели больше нет, но которые лежат в СОХРАНЁННЫХ
# планах: пресет уезжает в план целиком, а `extra="forbid"` роняет чтение такого
# плана валидацией (`samples/plan-baseline.json` перестал грузиться собственным
# загрузчиком). `max_line_chars` удалён в Ф5: перенос считается в пикселях, а не
# в символах (QUALITY_PLAN). Выкидываем его при чтении молча, а не воскрешаем
# полем-заглушкой — заглушку пресеты начали бы задавать снова, и она бы врала.
REMOVED_CAPTION_FIELDS = ("max_line_chars",)


class Caption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_removed_fields(cls, data: object) -> object:
        if isinstance(data, dict) and any(k in data for k in REMOVED_CAPTION_FIELDS):
            return {k: v for k, v in data.items() if k not in REMOVED_CAPTION_FIELDS}
        return data

    position: Literal["top", "center", "bottom"] = "bottom"
    # Отступ от края кадра в пикселях при высоте 1920.
    margin: int = 360
    line_height: float = 1.06
    align: Literal["center", "left"] = "center"
    # Сколько строк максимум в блоке. Не влезло — блок ужимается целиком, а не
    # переполняет контейнер (Ф5).
    max_lines: int = 3
    # Пол кегля при ужатии: доля от кегля строки. Ниже него текст перестаёт
    # быть субтитром, поэтому переполнение честнее, чем микроскопия. 0.4 — не
    # вкус, а замер: самый широкий глиф кириллицы в Montserrat ExtraBold — «Щ»
    # (1.135 em; «Ж» 1.109, «Ш» 1.099), и на поле 0.4 в бюджет строки bold
    # влезает 18 таких подряд, а девятнадцатый вылезает
    # (см. test_word_wider_than_the_floor_overflows_knowingly).
    min_word_scale: float = 0.4


class Geometry(BaseModel):
    """Геометрия кадра в долях, а не в пикселях TSX.

    Раньше эти числа жили константами в компонентах, и Python не мог предсказать,
    куда попадёт текст. Здесь они данные, которые читают и Python, и рендерер.
    """

    model_config = ConfigDict(extra="forbid")

    # Доля ширины кадра, которую может занять строка субтитра.
    max_line_width_ratio: float = 0.78
    # Безопасная зона у края кадра (был SAFE_AREA_RATIO в Captions.tsx).
    safe_area_ratio: float = 0.07
    # Картинка в углу: доля ширины кадра и отступ от края (были INSET_RATIO, MARGIN).
    broll_inset_ratio: float = 0.44
    broll_margin_ratio: float = 0.05
    broll_top_ratio: float = 0.16
    broll_bottom_ratio: float = 0.58


class Grade(BaseModel):
    """Лёгкий цветокор. Телефонное видео приходит тёмным и плоским."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    contrast: float = 1.08
    saturation: float = 1.12  # выше 1.2 на лицах даёт красноту
    gamma: float = 0.98
    vignette: float = 0.0  # сила виньетки в радианах для ffmpeg, 0 — выключена


class Crop(BaseModel):
    """Правила кадрирования: крупность задаётся долей лица, а не вписыванием bbox."""

    model_config = ConfigDict(extra="forbid")

    face_height_ratio: float = 0.22  # крупность «wide»
    tight_face_height_ratio: float = 0.28  # крупность «tight» на резе
    eye_line_ratio: float = 0.36  # линия глаз по высоте кадра, не центр
    headroom_ratio: float = 0.05  # запас над макушкой
    # Больше 1.0 — это апскейл исходника: вертикальный вход зумиться не должен.
    max_scale: float = 1.0


class Audio(BaseModel):
    """Звуковой пост: громкость речи, подложка, дакинг, эффекты."""

    model_config = ConfigDict(extra="forbid")

    target_lufs: float = -16.0
    music_gain_db: float = -20.0
    duck_db: float = -12.0
    sfx_gain_db: float = -6.0
    music_fade_s: float = 0.5
    music: str | None = None  # имя файла в assets/music, None — без подложки


class Contrast(BaseModel):
    """Контраст как ограничение, а не как вкус (Ф9).

    Панель фиксированной прозрачности не справляется над светлым фоном: белый
    текст на светлой стене читается хуже, чем на тёмной, при одних и тех же
    числах пресета. Препроцесс кладёт в блок `bg_luma` — среднюю яркость под
    ним, — а рендерер по этим двум опорным точкам ведёт панель и цвет текста
    между «над тёмным» и «над светлым».

    Опорные точки — это стиль. Порог `min_ratio` — это закон: если между
    точками контраст всё равно ниже него, рендерер поднимает прозрачность
    панели ровно настолько, чтобы порог был взят. `bg_luma = null` (никто не
    мерил) выключает всё правило целиком, и пресет играет как играл.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Контраст по WCAG. Для крупного жирного текста порог AA равен 3.0; берём
    # 4.5 — субтитр читают в движении и часто на телефоне на солнце.
    min_ratio: float = 4.5
    # Яркости опорных точек (0..1). Между ними всё интерполируется линейно,
    # снаружи — зажимается.
    luma_dark: float = 0.2
    luma_light: float = 0.8
    panel_opacity_dark: float = 0.5
    panel_opacity_light: float = 0.86
    # Цвет обычного слова (active/past/ghost) в опорных точках. Акцент и крик
    # сохраняют свой цвет: это опознавательный знак стиля, а не читаемость.
    text_dark: str = "#FFFFFF"
    text_light: str = "#FFFFFF"


class Animation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Появление слова: пружина (Remotion) / линейный масштаб (ASS).
    pop_scale: float = 1.16
    pop_ms: int = 160
    spring_damping: float = 12.0
    spring_stiffness: float = 190.0
    spring_mass: float = 0.7
    block_in_ms: int = 180
    # Сколько блок живёт ПОСЛЕ своего конца: столько же длится его уход и
    # столько же уходящий и приходящий блоки лежат в кадре вместе. Пресет просит
    # 100-140 мс, но нахлёст короче `block_overlap_min_ms` даёт между репликами
    # пустой кадр, поэтому фактический хвост — максимум из этих двух чисел.
    # Оба живут здесь, а не константами в TSX: §0.3, пресет — данные.
    block_out_ms: int = 140
    block_overlap_min_ms: int = 220
    # Насколько круче альфы уходящего блока падает кривая прихода и до какого
    # масштаба он ужимается, пока уходит. Не стилевые числа: они разводят два
    # текста, лежащих в одном месте кадра во время нахлёста. Показатель 1.35
    # опускает максимум одновременной читаемости с 0.50 до 0.448 (замер по
    # кривой, а не по кадровой сетке), а сумма альф в точке пересечения падает
    # с 1.00 всего до 0.90 — то есть кадр не проваливается в пустоту.
    block_leave_falloff: float = 1.35
    block_leave_scale: float = 0.9
    accent_scale: float = 1.18
    shout_scale: float = 1.34
    shout_rotate: float = -4.0  # градусы
    # Ход движения в пикселях при высоте кадра 1920. Раньше эти два числа были
    # константами в TSX (`font.size * 0.28` и `height * 0.09`) и Python о них не
    # знал; здесь они данные, которые пересчитывает scaled().
    block_shift_px: int = 24  # подъём блока субтитров на входе
    # Уход блока на выходе. Больше входа в разы, и это не вкус: блоки идут
    # встык, поэтому уходящий и приходящий блок ЛЕЖАТ ДРУГ НА ДРУГЕ всё время
    # нахлёста. Одной прозрачностью это не лечится: в точке пересечения кривых
    # обе не ниже 0.4 (сумма обязана держаться около единицы, иначе кадр
    # моргает), и два текста читаются одновременно — замерено на кадре 228
    # эталонного прогона. Развести их может только смещение, и оно должно быть
    # порядка междустрочия, а не 24 px входа.
    block_leave_px: int = 64
    broll_slide_px: int = 173  # ход карточки в slide_up


# Все поля пресета, измеренные в пикселях при высоте кадра 1920. Держим их одним
# списком: появилось новое пиксельное поле — впиши сюда, иначе на превью 720p оно
# останется в масштабе 1920 и композиция поедет. Всё, чего здесь нет, от высоты
# кадра не зависит по своей природе: миллисекунды, множители, доли, цвета.
_PIXEL_FIELDS: tuple[tuple[str, str], ...] = (
    ("font", "size"),
    ("caption", "margin"),
    ("panel", "padding_x"),
    ("panel", "padding_y"),
    ("panel", "radius"),
    ("panel", "blur"),
    ("panel", "border_width"),
    ("animation", "block_shift_px"),
    ("animation", "block_leave_px"),
    ("animation", "broll_slide_px"),
)

# Поля, которые округляются ВНИЗ, а не к ближайшему.
#
# Кегль решает перенос строк, а бюджет строки (`geometry.caption_text_width`)
# считается от ширины кадра в float и масштабируется точно. Округлённый вверх
# кегль растёт относительно бюджета — 88 → 59 при k = 0.6667 это +0.57 % — и
# строка, влезавшая на 1080p с запасом в доли процента, на превью 720p
# переносится. Замерено на эталонном прогоне: «не стоит бежать» — одна строка в
# 1080p и две в 720p, то есть композиция превью не совпадает с финалом.
# Вниз кегль относительно бюджета только уменьшается, поэтому перенос на меньшем
# кадре может стать разве что свободнее, но не теснее.
_ROUND_DOWN: frozenset[tuple[str, str]] = frozenset({("font", "size")})


class Preset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    title: str = ""
    font: FontSpec = Field(default_factory=FontSpec)
    colors: Colors = Field(default_factory=Colors)
    panel: Panel = Field(default_factory=Panel)
    caption: Caption = Field(default_factory=Caption)
    animation: Animation = Field(default_factory=Animation)
    geometry: Geometry = Field(default_factory=Geometry)
    contrast: Contrast = Field(default_factory=Contrast)
    grade: Grade = Field(default_factory=Grade)
    crop: Crop = Field(default_factory=Crop)
    audio: Audio = Field(default_factory=Audio)
    background: str = "#000000"

    def scaled(self, height: int) -> "Preset":
        """Размеры заданы для 1920 по высоте; под другой кадр — пропорционально.

        Пересчитывается всё пиксельное разом по `_PIXEL_FIELDS`. Раньше список
        был зашит в тело метода и покрывал шесть полей из девяти: рамка панели и
        ход анимаций оставались в масштабе 1920 и на превью 720p прыгали в
        полтора раза сильнее остальной композиции.
        """
        if height == 1920:
            return self
        k = height / 1920
        clone = self.model_copy(deep=True)
        for section, name in _PIXEL_FIELDS:
            value = getattr(getattr(self, section), name)
            # Ноль — это «выключено», он остаётся нулём. А единица это не размер,
            # а волосок: округление в ноль стирает край панели совсем.
            if value == 0:
                fresh = 0
            elif (section, name) in _ROUND_DOWN:
                fresh = max(1, int(value * k))
            else:
                fresh = max(1, round(value * k))
            setattr(getattr(clone, section), name, fresh)
        clone.font.size = max(12, clone.font.size)
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
