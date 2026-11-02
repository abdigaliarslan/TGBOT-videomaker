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
    # Ниже — добавленные позже. Порядок словаря это порядок чипов в приложении,
    # поэтому новые идут в конец: у выбравших цвет ничего не переставляется.
    "red": ({"ru": "Красный", "en": "Red"}, "#FF3B30"),
    "sky": ({"ru": "Небо", "en": "Sky"}, "#4DA6FF"),
    "orange": ({"ru": "Апельсин", "en": "Orange"}, "#FF9F0A"),
    "mint": ({"ru": "Мята", "en": "Mint"}, "#30D158"),
    "amber": ({"ru": "Янтарь", "en": "Amber"}, "#FFD60A"),
    "ice": ({"ru": "Лёд", "en": "Ice"}, "#EAF2FF"),
}


class FontChoice(BaseModel):
    """Шрифт из библиотеки. Имя семейства — ровно то, что лежит внутри TTF.

    `family` сверяется с выгруженными метриками тестом: ASS-бэкенд ищет шрифт
    по имени семейства через fontconfig, и расхождение дало бы чужой шрифт без
    единого сообщения об ошибке.
    """

    model_config = ConfigDict(extra="forbid")

    family: str
    file: str
    weight: int


# Библиотека шрифтов субтитра. Ключ хранится в задаче, значение — подписи и сам
# шрифт. `None` значит «как задумано темой».
#
# Добавить шрифт — это четыре шага, и пропуск любого ломает молча:
#   1. `python tools/font_instance.py <url|файл> <вес> --name <Имя>` (нарезать
#      статическое начертание: выгрузка метрик читает дефолтный инстанс, а у
#      переменных шрифтов он почти всегда Regular);
#   2. `python tools/font_metrics.py assets/fonts/<Имя>.ttf` и коммит метрик;
#   3. строка здесь;
#   4. прогон `pytest -m slow -k bundled` — сверка раскладки с Chromium.
FONTS: dict[str, tuple[dict[str, str], FontChoice | None]] = {
    "theme": ({"ru": "Как в теме", "en": "Theme default"}, None),
    "grotesk": (
        {"ru": "Гротеск", "en": "Grotesque"},
        FontChoice(family="Montserrat ExtraBold", file="Montserrat-ExtraBold.ttf", weight=800),
    ),
    "serif": (
        {"ru": "Антиква", "en": "Serif"},
        FontChoice(
            family="Playfair Display ExtraBold",
            file="PlayfairDisplay-ExtraBold.ttf",
            weight=800,
        ),
    ),
    "neutral": (
        {"ru": "Нейтральный", "en": "Neutral"},
        FontChoice(family="Golos Text", file="GolosText-Bold.ttf", weight=700),
    ),
    "display": (
        {"ru": "Характерный", "en": "Display"},
        FontChoice(family="Unbounded ExtraBold", file="Unbounded-ExtraBold.ttf", weight=800),
    ),
    "narrow": (
        {"ru": "Узкий", "en": "Condensed"},
        FontChoice(family="Oswald SemiBold", file="Oswald-SemiBold.ttf", weight=600),
    ),
}


class JobOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    accent: str = "theme"
    font: str = "theme"
    cut: bool = Field(default=True, description="резать затяжные паузы и заминки")
    sfx: bool = Field(default=True, description="звуковые акценты на картинках и укрупнениях")
    broll: bool = Field(default=True, description="медиа-карточки по смыслу")
    camera: bool = Field(default=True, description="смена крупности на резах")

    @field_validator("accent")
    @classmethod
    def _known_accent(cls, v: str) -> str:
        return v if v in ACCENTS else "theme"

    @field_validator("font")
    @classmethod
    def _known_font(cls, v: str) -> str:
        return v if v in FONTS else "theme"

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


def apply_font(preset: Preset, font: str) -> Preset:
    """Пресет с другим шрифтом субтитра. Исходный объект не трогается.

    Кегли пересчитываются, и это главное здесь. Кегль задан в пикселях, а глаз
    меряет высоту прописных: у Oswald SemiBold она 0.810 em против 0.700 у
    Montserrat, то есть те же 88 px дали бы букву на 16 % крупнее задуманного
    темой. Нормируем по `cap_height` из ВЫГРУЖЕННЫХ метрик — тех самых, что
    сверены с Chromium, — а не по числу, переписанному руками в каталог:
    второй источник правды тут разошёлся бы молча.

    Пересчитываются ровно три кегля: основной, тихого регистра и титра. Это те
    же поля, что `Preset.scaled` округляет ВНИЗ, и по той же причине — кегль
    решает перенос строк, и округление вверх теснит строку относительно
    бюджета.

    Чего НЕ трогаем: `letter_spacing` задан в em и едет за кеглем сам, а
    `geometry.min_word_scale` — свойство темы, выведенное под ширины
    Montserrat (см. его докстроку).
    """
    choice = FONTS.get(font, FONTS["theme"])[1]
    if choice is None:
        return preset

    from src.core.geometry import load_metrics

    was, now = load_metrics(preset.font.file or ""), load_metrics(choice.file)
    ratio = 1.0
    if was.cap_height and now.cap_height:
        ratio = (was.cap_height / was.units_per_em) / (now.cap_height / now.units_per_em)

    clone = preset.model_copy(deep=True)
    clone.font.family = choice.family
    clone.font.file = choice.file
    clone.font.weight = choice.weight
    clone.font.size = max(12, int(preset.font.size * ratio))
    clone.caption.quiet.font_size = max(12, int(preset.caption.quiet.font_size * ratio))
    clone.caption.title_card.font_size = max(
        12, int(preset.caption.title_card.font_size * ratio)
    )
    return clone
