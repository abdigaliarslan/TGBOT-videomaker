"""M3: генератор .ass из плана рендера — быстрый путь к картинке до Remotion.

Три состояния слова делаются через альфу и `\\c`, появление — через `\\t` на
масштабе, позиция — через `\\an`. Glass-панель и b-roll здесь намеренно не
делаются: в ASS это выходит плохо, это работа M5.

Одна реплика на одно активное слово. Караоке-теги `\\k` дают только два
состояния (спето/не спето), а нам нужно три (ghost / active / past).

Привязка блока к голове (М2, `PlanBlock.anchor_path`) здесь тоже намеренно не
делается: путь задан по кадрам, а в ASS позиция ставится на реплику целиком, и
воспроизвести его можно было бы только разбив каждый блок на одну реплику на
кадр. Быстрый путь на то и быстрый — привязанный блок в нём рисуется статично,
как любой другой, и ролик остаётся смотрибельным.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from src.core.preset import Preset, to_ass_alpha, to_ass_colour
from src.core.proc import ffmpeg
from src.core.render_plan import PlanBlock, PlanWord, RenderPlan

log = logging.getLogger(__name__)

ALIGN = {"bottom": 2, "center": 5, "top": 8}

# Мягкая тень вместо контура (Ф1). Жирный шрифт с чёрным кантом читается как
# дефолтный шаблон монтажки, а под панелью кант вдобавок избыточен. В ASS нет
# многослойного text-shadow, поэтому эквивалент CSS-гало `0 0 Npx` собирается из
# тонкой обводки, размытой `\blur`: это подложка под буквой, а не кант.
# Доли от кегля: было 0.07 канта без размытия.
OUTLINE_RATIO = 0.03
BLUR_RATIO = 0.06
SHADOW_RATIO = 0.045

# Пол длительности анимации в кадрах. То же число и та же причина, что в
# render/src/style.ts (MIN_ANIM_FRAMES): короче восьми кадров переход читается не
# как движение, а как мигание. Пресеты просят 130-160 мс — это 3-5 кадров на
# 24-30 fps, то есть закон Ф8 обходил ASS стороной, пока длительность бралась из
# пресета как есть.
MIN_ANIM_FRAMES = 8
# Доля разгона в «пшике»: `popBump` в style.ts делит окно ровно так же.
POP_RISE = 0.45


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def escape(text: str) -> str:
    """В ASS фигурные скобки открывают блок тегов, а перевод строки значим."""
    return (
        text.replace("\\", "⧵")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", " ")
        .strip()
    )


def _word_scale(word: PlanWord, preset: Preset) -> float:
    """Кегль слова как доля кегля строки.

    Считает его Python в `render_plan.wrap_block` — он же им мерил строку и решал
    перенос, и длинное слово, ужатое под ширину строки, известно только ему.
    Поэтому размер берём из плана, а из выделения выводим только там, где плана
    нет (пресет без метрик шрифта): тогда оба рендерера падают на один и тот же
    запасной путь.
    """
    if word.size and preset.font.size:
        return word.size / float(preset.font.size)
    if word.emphasis == "accent":
        return preset.animation.accent_scale
    if word.emphasis == "shout":
        return preset.animation.shout_scale
    return 1.0


def pop_ms(preset: Preset, fps: int) -> int:
    """Желаемая длительность «пшика» в миллисекундах, но не короче восьми кадров."""
    # Вверх, а не к ближайшему: round(8/24·1000) = 333 мс — это 7.992 кадра,
    # то есть пол, который сам себя не берёт.
    floor = math.ceil(MIN_ANIM_FRAMES / max(1, fps) * 1000)
    return max(preset.animation.pop_ms, floor)


def pop_window_ms(preset: Preset, fps: int, dialogue_ms: int) -> int:
    """Сколько «пшик» длится ВНУТРИ своей реплики. Ноль — не анимируем вовсе.

    Теги `\\t` отсчитываются от начала реплики, а реплика живёт от слова i до
    слова i+1 — то есть длительность анимации ограничена сверху не пресетом, а
    речью. Пол в восемь кадров, взятый из пресета как есть, этого не знает: на
    24 fps он просит 334 мс, а 15 реплик эталонного ролика из 35 короче — «пшик»
    не успевает вернуть слову его кегль, реплика кончается, следующая рисует то
    же слово базовым кеглем, и выходит ЩЕЛЧОК ровно там, где Ф8 его убирала.
    Хуже того, `\\fscx` в libass меняет раскладку, поэтому доля времени, когда
    строка переливается по горизонтали, росла с 30 % до 54 %.

    Поэтому окно считается по реплике, а пол остаётся законом, а не пожеланием:
    не помещается восемь кадров — не рисуем ничего. Огрызок анимации хуже её
    отсутствия, так же решает Broll.tsx при `travel < MIN_ANIM_FRAMES`.
    """
    total = min(pop_ms(preset, fps), max(0, dialogue_ms))
    return total if total * max(1, fps) >= MIN_ANIM_FRAMES * 1000 else 0


def _word_colour(word: PlanWord, preset: Preset) -> str:
    c = preset.colors
    if word.emphasis == "accent":
        return to_ass_colour(c.accent)
    if word.emphasis == "shout":
        return to_ass_colour(c.shout)
    return to_ass_colour(c.active)


def _render_word(word: PlanWord, state: str, preset: Preset, fps: int, dialogue_ms: int) -> str:
    """Один токен с инлайновыми тегами под своё состояние.

    Кегль слова ОДИН И ТОТ ЖЕ во всех трёх состояниях. Раньше выделение жило
    только в `active`, а в `past` и `ghost` слово возвращалось к 100 %: libass
    меряет `\\fscx` при раскладке, поэтому строка меняла ширину каждый раз,
    когда акцентное слово меняло состояние, и весь блок прыгал по горизонтали.
    Инвариант «контейнер не прыгает» (Ф1) обязан держаться в обоих рендерерах,
    а не только в Remotion.

    Что в ASS вылечить нечем: `\\fscx` меняет и ширину пролёта глифа, поэтому
    строка переливается по горизонтали на время «пшика». В Remotion этого нет —
    там кегль это layout, а «пшик» это `transform` поверх него. Амплитуда мала
    (`pop_scale` 1.05-1.06 на одно слово), и ASS — быстрый предпросмотр, а не
    финальный рендер, поэтому живём с этим.

    Второе известное расхождение с Remotion, тоже неустранимое здесь: там «пшик»
    считается от начала слова и продолжается, когда слово стало past, поэтому на
    24-30 fps одновременно пульсируют до трёх слов подряд. В ASS активное слово
    по построению ровно одно: реплика перерисовывает строку целиком, и у past
    нет способа доиграть свою анимацию. Гасить «пшик» на переходе active→past в
    Remotion ради симметрии нельзя — слово короче восьми кадров оборвало бы
    масштаб на полпути, то есть щёлкнуло бы ровно так, как щёлкала эта функция.
    """
    c = preset.colors
    a = preset.animation
    text = escape(word.text)
    if word.emoji:
        text = f"{text} {escape(word.emoji)}"

    base = max(1, round(_word_scale(word, preset) * 100))
    size = f"\\fscx{base}\\fscy{base}"

    if state == "active":
        # Транзиентный «пшик» ровно как в Remotion (`popBump`): вверх к пику и
        # обратно к своему кеглю. Раньше слово ПОЯВЛЯЛОСЬ уже на пике — то есть
        # скачком, который Ф8 и убирала на стороне TSX. Окно — своя реплика:
        # `\t` обязан вернуть слову его кегль до её конца, иначе следующая
        # реплика делает это скачком (см. `pop_window_ms`).
        total = pop_window_ms(preset, fps, dialogue_ms)
        tags = [to_ass_alpha(1.0), f"\\c{_word_colour(word, preset)}", size]
        if total:
            rise = max(1, round(total * POP_RISE))
            peak = max(1, round(base * a.pop_scale))
            tags += [
                f"\\t(0,{rise},\\fscx{peak}\\fscy{peak})",
                f"\\t({rise},{total},{size})",
            ]
        if word.emphasis == "shout" and a.shout_rotate:
            tags.append(f"\\frz{a.shout_rotate:g}")
    elif state == "past":
        tags = [to_ass_alpha(c.past_opacity), f"\\c{to_ass_colour(c.past)}", size]
    else:  # ghost — слово ещё не прозвучало
        tags = [to_ass_alpha(c.ghost_opacity), f"\\c{to_ass_colour(c.ghost)}", size]

    joined = "".join(t if t.startswith("\\") else f"\\alpha{t}" for t in tags)
    return "{" + joined + "}" + text


def _block_dialogues(
    block: PlanBlock, preset: Preset, fps: int
) -> list[tuple[float, float, str, int]]:
    """Блок → список реплик (начало, конец, текст, выравнивание)."""
    words = block.words
    if not words:
        return []
    align = ALIGN.get(block.position, 2)
    out: list[tuple[float, float, str, int]] = []

    # Позиция слова в блоке — по счётчику, а не поиском по значению: два
    # одинаковых слова подряд («очень очень») сломали бы index().
    positions: list[list[int]] = []
    counter = 0
    for line in block.lines:
        positions.append(list(range(counter, counter + len(line))))
        counter += len(line)

    for i in range(len(words)):
        start = block.start if i == 0 else words[i].start
        end = words[i + 1].start if i + 1 < len(words) else block.end
        if end <= start:
            continue
        # Окно самой реплики: `\t` живёт внутри него и наружу не выходит.
        dialogue_ms = round((end - start) * 1000)
        chunks: list[str] = []
        for line, line_pos in zip(block.lines, positions):
            rendered = [
                _render_word(
                    w,
                    "active" if j == i else ("past" if j < i else "ghost"),
                    preset,
                    fps,
                    dialogue_ms,
                )
                for w, j in zip(line, line_pos)
            ]
            chunks.append(" ".join(rendered))
        out.append((start, end, "\\N".join(chunks), align))
    return out


def build_ass(plan: RenderPlan) -> str:
    preset = Preset(**plan.preset)
    c = preset.colors
    f = preset.font

    margin = max(10, preset.caption.margin)
    style = ",".join(
        [
            "Main",
            f.family,
            str(f.size),
            to_ass_colour(c.active),  # PrimaryColour
            to_ass_colour(c.active),  # SecondaryColour (караоке не используем)
            to_ass_colour(c.outline),  # OutlineColour
            to_ass_colour(c.shadow, 0.55),  # BackColour (тень)
            "-1" if f.weight >= 600 else "0",  # Bold
            "0",  # Italic
            "0",  # Underline
            "0",  # StrikeOut
            "100",  # ScaleX
            "100",  # ScaleY
            f"{round(f.letter_spacing * f.size, 1)}",  # Spacing
            "0",  # Angle
            "1",  # BorderStyle: обводка + тень
            f"{max(1, round(f.size * OUTLINE_RATIO))}",  # Outline — гало, а не кант
            f"{max(1, round(f.size * SHADOW_RATIO))}",  # Shadow
            "2",  # Alignment (перебивается \an в репликах)
            "80",  # MarginL
            "80",  # MarginR
            str(margin),  # MarginV
            "1",  # Encoding
        ]
    )

    header = f"""[Script Info]
; Сгенерировано ReelBot из EDL. Правки руками потеряются при перерендере.
Title: ReelBot {preset.name}
ScriptType: v4.00+
PlayResX: {plan.width}
PlayResY: {plan.height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: {style}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # `\blur` ставится один раз в начале реплики: теги ASS живут до конца строки,
    # а размывает он и обводку, и тень — то есть ровно то, что должно быть мягким.
    blur = max(2, round(f.size * BLUR_RATIO))

    lines: list[str] = []
    for block in plan.blocks:
        for start, end, text, align in _block_dialogues(block, preset, plan.fps):
            body = "{" + f"\\an{align}\\blur{blur}" + "}" + text
            lines.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Main,,0,0,0,,{body}"
            )
    return header + "\n".join(lines) + "\n"


def write_ass(plan: RenderPlan, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_ass(plan), encoding="utf-8")
    return path


def burn(
    plan: RenderPlan,
    ass_path: str | Path,
    out_path: str | Path,
    *,
    fonts_dir: str | Path | None = None,
    crf: int = 20,
) -> Path:
    """Вшивание субтитров в видео. Вход — уже нарезанное и обрезанное видео."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ffmpeg-фильтры разбирают путь как выражение: двоеточия и запятые надо экранировать.
    escaped = str(ass_path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    vf = f"subtitles='{escaped}'"
    if fonts_dir:
        fd = str(fonts_dir).replace("\\", "/").replace(":", r"\:")
        vf += f":fontsdir='{fd}'"

    args = ["-i", str(plan.video), "-vf", vf, "-c:v", "libx264", "-preset", "medium",
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if plan.audio:
        args = ["-i", str(plan.video), "-i", str(plan.audio), "-map", "0:v", "-map", "1:a",
                "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    args += ["-c:a", "aac", "-b:a", "160k", str(out_path)]
    ffmpeg(args, timeout=3600)
    log.info("ASS-рендер готов: %s", out_path)
    return out_path
