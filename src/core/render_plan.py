"""EDL → план рендера. Единственное место, где считается время для картинки.

Рендерер (хоть ASS, хоть Remotion) не принимает решений и не считает арифметику:
он получает готовые секунды, готовые строки и готовый пресет и рисует буквально
(§0). Поэтому обе реализации рендера едят одну и ту же структуру.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.core.config import Settings, settings as default_settings
from src.core.edl import EDL, Emphasis, Timeline
from src.core.preset import Preset, load_preset
from src.core.text import display_text


@dataclass(slots=True)
class PlanWord:
    text: str
    start: float  # секунды на ВЫХОДНОМ таймлайне
    end: float
    emphasis: str = "none"
    emoji: str | None = None
    word_idx: int = -1


@dataclass(slots=True)
class PlanBlock:
    start: float
    end: float
    position: Literal["top", "center", "bottom"]
    lines: list[list[PlanWord]]

    @property
    def words(self) -> list[PlanWord]:
        return [w for line in self.lines for w in line]


@dataclass(slots=True)
class PlanBroll:
    start: float
    end: float
    anim: str
    anchor: str
    asset: str | None
    query: str


@dataclass(slots=True)
class RenderPlan:
    version: int
    fps: int
    width: int
    height: int
    duration: float
    video: str
    preset: dict[str, Any]
    blocks: list[PlanBlock] = field(default_factory=list)
    broll: list[PlanBroll] = field(default_factory=list)
    audio: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RenderPlan":
        return cls(
            version=data.get("version", 1),
            fps=data["fps"],
            width=data["width"],
            height=data["height"],
            duration=data["duration"],
            video=data["video"],
            audio=data.get("audio"),
            preset=data["preset"],
            blocks=[
                PlanBlock(
                    start=b["start"],
                    end=b["end"],
                    position=b["position"],
                    lines=[[PlanWord(**w) for w in line] for line in b["lines"]],
                )
                for b in data.get("blocks", [])
            ],
            broll=[PlanBroll(**x) for x in data.get("broll", [])],
            meta=data.get("meta", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration": round(self.duration, 3),
            "video": self.video,
            "audio": self.audio,
            "preset": self.preset,
            "blocks": [
                {
                    "start": round(b.start, 3),
                    "end": round(b.end, 3),
                    "position": b.position,
                    "lines": [[asdict(w) for w in line] for line in b.lines],
                }
                for b in self.blocks
            ],
            "broll": [asdict(x) for x in self.broll],
            "meta": self.meta,
        }


# --------------------------------------------------------------------------- перенос строк


def wrap_words(words: list[PlanWord], max_chars: int, max_lines: int = 3) -> list[list[PlanWord]]:
    """Жадный перенос по символам. Считает Python, чтобы рендерер не думал."""
    if not words:
        return []
    lines: list[list[PlanWord]] = [[]]
    length = 0
    for w in words:
        wl = len(w.text)
        if lines[-1] and length + 1 + wl > max_chars and len(lines) < max_lines:
            lines.append([])
            length = 0
        lines[-1].append(w)
        length += wl + (1 if length else 0)
    return [line for line in lines if line]


# --------------------------------------------------------------------------- сборка


BOTTOM_ANCHORS = {"bottom_left", "bottom_right"}
TOP_ANCHORS = {"top_left", "top_right"}


def _resolve_position(
    requested: str,
    preset_position: str,
    *,
    collides_bottom: bool = False,
    collides_top: bool = False,
) -> Literal["top", "center", "bottom"]:
    """Явную позицию уважаем, «auto» решаем сами.

    Единственное настоящее решение здесь — разъехаться с b-roll: если картинка
    висит в той же половине кадра, блок уходит в противоположную. Иначе
    субтитр и картинка перекроют друг друга (§7 DoD).
    """
    if requested in ("top", "center", "bottom"):
        return requested  # type: ignore[return-value]
    base = preset_position if preset_position in ("top", "center", "bottom") else "bottom"
    if base == "bottom" and collides_bottom and not collides_top:
        return "top"
    if base == "top" and collides_top and not collides_bottom:
        return "bottom"
    return base  # type: ignore[return-value]


def build_plan(
    edl: EDL,
    video_path: str,
    *,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    preset: Preset | None = None,
    apply_cuts: bool = True,
    audio_path: str | None = None,
    cfg: Settings | None = None,
    meta: dict[str, Any] | None = None,
) -> RenderPlan:
    """EDL → RenderPlan.

    `apply_cuts=True` означает, что видео УЖЕ нарезано по EDL.cuts и время надо
    пересчитать на выходной таймлайн. `False` — рисуем поверх исходника.
    """
    cfg = cfg or default_settings
    fps = fps or cfg.render_fps
    width = width or cfg.render_width
    height = height or cfg.render_height
    ps = (preset or load_preset(edl.preset)).scaled(height)

    tl = edl.timeline() if apply_cuts else Timeline([], edl.source_duration)
    duration = tl.duration
    by_idx = {w.idx: w for w in edl.words}

    # B-roll считаем раньше субтитров: блокам нужно знать, какая половина кадра
    # занята картинкой, чтобы уехать в свободную.
    broll: list[PlanBroll] = []
    for b in edl.broll:
        w = by_idx.get(b.word_idx)
        if w is None:
            continue
        window = tl.word_window(w)
        if window is None:
            continue
        broll.append(
            PlanBroll(
                start=round(window[0], 3),
                end=round(min(window[0] + b.duration, duration), 3),
                anim=b.anim.value if hasattr(b.anim, "value") else str(b.anim),
                anchor=b.anchor,
                asset=b.asset,
                query=b.query,
            )
        )

    def occupied(start: float, end: float) -> tuple[bool, bool]:
        """Какие половины кадра заняты картинкой в этом промежутке."""
        bottom = top = False
        for item in broll:
            if item.end <= start or item.start >= end:
                continue
            if item.anchor in BOTTOM_ANCHORS or item.anchor in ("center", "fullscreen"):
                bottom = True
            if item.anchor in TOP_ANCHORS or item.anchor in ("center", "fullscreen"):
                top = True
        return bottom, top

    blocks: list[PlanBlock] = []
    for block in edl.captions:
        pwords: list[PlanWord] = []
        for cw in block.words:
            w = by_idx.get(cw.word_idx)
            if w is None:
                continue
            window = tl.word_window(w)
            if window is None:
                continue
            text = display_text(w.text, uppercase=ps.font.uppercase)
            pwords.append(
                PlanWord(
                    text=text,
                    start=round(window[0], 3),
                    end=round(window[1], 3),
                    emphasis=cw.emphasis.value if isinstance(cw.emphasis, Emphasis) else str(cw.emphasis),
                    emoji=cw.emoji,
                    word_idx=cw.word_idx,
                )
            )
        if not pwords:
            continue

        start = pwords[0].start
        end = pwords[-1].end + cfg.sub_tail
        hits_bottom, hits_top = occupied(start, end)
        blocks.append(
            PlanBlock(
                start=start,
                end=end,
                position=_resolve_position(
                    block.position,
                    ps.caption.position,
                    collides_bottom=hits_bottom,
                    collides_top=hits_top,
                ),
                lines=wrap_words(pwords, ps.caption.max_line_chars),
            )
        )

    # Блоки не должны накладываться: следующий начинается — предыдущий уходит.
    for a, b in zip(blocks, blocks[1:]):
        if a.end > b.start:
            a.end = b.start
    if blocks and duration > 0:
        blocks[-1].end = min(blocks[-1].end, duration)

    return RenderPlan(
        version=1,
        fps=fps,
        width=width,
        height=height,
        duration=duration,
        video=video_path,
        audio=audio_path,
        preset=ps.model_dump(),
        blocks=blocks,
        broll=broll,
        meta=meta or {},
    )
