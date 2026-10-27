"""Композиция этапов в один прогон: файл на входе → готовый ролик на выходе.

Здесь же собираются метрики (§7): время каждого этапа, стоимость LLM, доля
падений на валидации EDL. Без них экономика вскроется поздно.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from src.core.config import Settings, settings as default_settings
from src.core.edl import EDL, validate_edl
from src.core.options import JobOptions, apply_accent
from src.core.preset import load_preset
from src.core.render_plan import build_plan
from src.core.track import load_track
from src.llm.client import LLMClient
from src.pipeline import broll as broll_mod
from src.pipeline.asr import ASRResult, transcribe
from src.pipeline.audio_post import build_audio
from src.pipeline.crop import autocrop, framing_spans
from src.pipeline.cut import apply_cuts
from src.pipeline.director import clean_transcript, direct, make_metadata
from src.pipeline.luma import measure_bg_luma
from src.pipeline.probe import MediaInfo, is_playable, probe
from src.pipeline.render import arender_remotion, render_ass

log = logging.getLogger(__name__)

Progress = Callable[[str, float], Awaitable[None] | None]

STAGES = [
    ("probe", "Читаю файл", 0.03),
    ("asr", "Распознаю речь", 0.32),
    ("clean", "Проверяю расшифровку", 0.36),
    ("direct", "Собираю монтаж", 0.45),
    ("broll", "Подбираю картинки", 0.52),
    ("cut", "Режу паузы", 0.66),
    ("audio", "Свожу звук", 0.70),
    ("crop", "Кадрирую 9:16", 0.78),
    # Рендер — самый долгий этап, поэтому ему отдан диапазон 0.80…0.97: полоса
    # стартует с нижней границы и идёт по мере готовых кадров (`render_share`).
    ("render", "Рисую субтитры", 0.80),
    ("done", "Готово", 1.0),
]
STAGE_TITLES = {key: title for key, title, _ in STAGES}
RENDER_SHARE_END = 0.97


def render_share(done: float) -> float:
    """Доля всего монтажа, когда рендер готов на `done` (0..1)."""
    start = dict((k, v) for k, _, v in STAGES)["render"]
    return round(start + (RENDER_SHARE_END - start) * max(0.0, min(1.0, done)), 3)
# Обратный ход: воркер получает в прогресс русский заголовок, а интерфейсу нужен
# ключ, чтобы показать стадию на языке пользователя.
STAGE_KEYS_BY_TITLE = {title: key for key, title, _ in STAGES}
STAGE_SHARE = {key: share for key, _, share in STAGES}


class JobError(RuntimeError):
    """Ошибка, которую можно показать пользователю.

    Текст исключения — русский, для логов и поля `jobs.error`. Пользователь его
    не видит: интерфейс показывает фразу по `code` на языке пользователя
    (`core.i18n.ERROR_CODES`).
    """

    def __init__(self, message: str, code: str = "failed") -> None:
        super().__init__(message)
        self.code = code


@dataclass
class JobResult:
    output: Path
    edl: EDL
    asr: ASRResult
    info: MediaInfo
    timings: dict[str, float] = field(default_factory=dict)
    llm_report: dict[str, Any] = field(default_factory=dict)
    edl_from_llm: bool = False
    broll_used: int = 0
    metadata: dict[str, Any] | None = None
    transcript_fixes: int = 0
    audio: dict[str, Any] | None = None  # что сделал звуковой пост, None — не запускался

    def summary(self) -> dict[str, Any]:
        return {
            "output": str(self.output),
            "source_duration": self.info.duration,
            "out_duration": round(self.edl.out_duration, 2),
            "words": len(self.asr.words),
            "blocks": len(self.edl.captions),
            "cuts_dropped": sum(1 for c in self.edl.cuts if not c.kept),
            "broll": self.broll_used,
            "edl_from_llm": self.edl_from_llm,
            "transcript_fixes": self.transcript_fixes,
            "audio": self.audio,
            "metadata": self.metadata,
            "timings": self.timings,
            "llm": self.llm_report,
        }


def _measure_luma(
    edl: EDL, staged: Path, *, preset: Any, cut_pauses: bool, cfg: Settings
) -> None:
    """Ф9: положить в EDL яркость фона под каждым субтитром.

    Место жёсткое: после кропа (под субтитром окажется именно этот кадр) и до
    `build_plan` (план уносит число в рендерер). Замер необязателен — без него
    панель просто останется той, что в пресете, и ролик всё равно выйдет.
    """
    try:
        measure_bg_luma(edl, staged, preset=preset, apply_cuts=cut_pauses, cfg=cfg)
    except Exception as e:  # noqa: BLE001 — контраст важен, но ролик важнее
        log.warning("замер яркости фона пропущен: %s", e)


async def _notify(progress: Progress | None, stage: str) -> None:
    if progress is None:
        return
    res = progress(STAGE_TITLES.get(stage, stage), STAGE_SHARE.get(stage, 0.0))
    if hasattr(res, "__await__"):
        await res  # type: ignore[misc]


async def process_video(
    src: str | Path,
    work_dir: str | Path,
    *,
    preset: str | None = None,
    llm: LLMClient | None = None,
    progress: Progress | None = None,
    cut_pauses: bool = True,
    vertical: bool = True,
    backend: str | None = None,
    cfg: Settings | None = None,
    keep_work: bool | None = None,
    options: JobOptions | None = None,
) -> JobResult:
    """Полный прогон: probe → asr → direct → broll → cut → crop → render.

    `options` — галочки пользователя (Mini App). Без них поведение прежнее: всё
    включено, цвет выделения — как в теме. Нарезка по-прежнему управляется
    `cut_pauses`; вызывающий сводит одно с другим сам.
    """
    cfg = cfg or default_settings
    options = options or JobOptions()
    src = Path(src)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    # Рабочую папку удаляет вызывающий (воркер) — здесь только помечаем режим.
    keep = cfg.debug if keep_work is None else keep_work

    def mark(stage: str, t0: float) -> None:
        timings[stage] = round(time.perf_counter() - t0, 2)

    try:
        # ---------------------------------------------------------- восприятие
        await _notify(progress, "probe")
        t = time.perf_counter()
        info = probe(src)
        mark("probe", t)

        if info.duration < cfg.min_video_seconds:
            raise JobError(
                f"Видео короче {cfg.min_video_seconds:.0f} с — монтировать нечего.", "too_short"
            )
        if info.duration > cfg.max_video_seconds:
            raise JobError(
                f"Видео длиннее {cfg.max_video_minutes:.0f} мин. "
                f"Пришлите кусок покороче.",
                "too_long",
            )
        if not info.has_audio:
            raise JobError("В видео нет звуковой дорожки — субтитры делать не из чего.", "no_audio")

        await _notify(progress, "asr")
        t = time.perf_counter()
        asr = transcribe(src)
        mark("asr", t)
        if not asr.words:
            raise JobError("Речь не распознана. Возможно, в ролике только музыка или шум.", "no_speech")

        # ---------------------------------------------------------- правка ASR
        words = asr.words
        fixes = 0
        if llm is not None and llm.ready:
            await _notify(progress, "clean")
            t = time.perf_counter()
            words, fixes = await clean_transcript(
                asr.words, llm=llm, language=asr.language, cfg=cfg
            )
            mark("clean", t)

        # ---------------------------------------------------------- режиссёр
        await _notify(progress, "direct")
        t = time.perf_counter()
        edl = await direct(
            words,
            info.duration,
            llm=llm,
            language=asr.language,
            preset=preset,
            cfg=cfg,
        )
        mark("direct", t)
        errors = validate_edl(edl)
        if errors:
            raise JobError("Не удалось собрать монтажный лист: " + "; ".join(errors[:3]), "edl_invalid")
        edl_from_llm = bool(llm and llm.ready and any(u.prompt == "direct" for u in llm.usages))
        style = apply_accent(load_preset(preset or edl.preset), options.accent)
        if not options.camera:
            # Движение камеры выключено: один план на весь ролик. Правится EDL, а
            # не кроп, — тогда и звуковые акценты на укрупнениях пропадают сами,
            # и сохранённый лист честно описывает то, что видно в кадре.
            for c in edl.cuts:
                c.framing = "wide"

        # ---------------------------------------------------------- b-roll
        broll_used = 0
        if cfg.broll_enabled and options.broll and llm is not None and llm.ready:
            await _notify(progress, "broll")
            t = time.perf_counter()
            try:
                broll_used = await broll_mod.plan_and_fetch(
                    edl, src, work / "broll", llm=llm, info=info, cfg=cfg
                )
            except Exception as e:  # noqa: BLE001 — b-roll необязателен, ролик важнее
                log.warning("b-roll пропущен: %s", e)
            mark("broll", t)

        # ---------------------------------------------------------- монтаж
        await _notify(progress, "cut")
        t = time.perf_counter()
        cut_path = work / "cut.mp4"
        if cut_pauses:
            apply_cuts(edl, src, cut_path, info=info, cfg=cfg)
        else:
            shutil.copy2(src, cut_path)
        if not is_playable(cut_path):
            # Оборванный ffmpeg (упавший воркер, kill) оставляет mp4 без
            # moov-атома. Дальше он поедет в рендер и упадёт там, где причину
            # уже не видно.
            raise JobError("Нарезка оборвалась — файл повреждён. Пришлите видео ещё раз.", "cut_broken")
        mark("cut", t)

        # ---------------------------------------------------------- звук (Ф2)
        # Между нарезкой и рендером, на cut.mp4: дорожка обязана лежать на том же
        # выходном таймлайне, что и план. Кроп длительность не меняет.
        audio_path: str | None = None
        audio_report: dict[str, Any] | None = None
        if cfg.audio_post_enabled:
            await _notify(progress, "audio")
            t = time.perf_counter()
            try:
                track = build_audio(
                    edl, cut_path, work / "audio.wav", preset=style,
                    apply_cuts=cut_pauses, sfx=options.sfx, cfg=cfg,
                )
                if track is not None:
                    audio_path = str(track.path)
                    audio_report = {
                        "music": track.music,
                        "sfx": track.sfx,
                        "lufs_in": track.lufs_in,
                        "target_lufs": track.target_lufs,
                    }
            except Exception as e:  # noqa: BLE001 — без звукового поста ролик всё равно едет
                log.warning("звуковой пост пропущен: %s", e)
            mark("audio", t)

        await _notify(progress, "crop")
        t = time.perf_counter()
        staged = cut_path
        if vertical:
            staged = autocrop(
                cut_path, work / "vertical.mp4", work_dir=work,
                out_width=cfg.render_width, out_height=cfg.render_height,
                # Пресет несёт цветокор (Ф3), крупность — смену плана на резе (Ф6).
                # Без них обе фазы написаны, но выключены: кроп берёт пресет из
                # конфига и снимает весь ролик одной крупностью.
                preset=style, framing=framing_spans(edl, applied_cuts=cut_pauses),
                cfg=cfg, track_out=work / "track.json",
            )
        if not is_playable(staged):
            raise JobError("Кадрирование оборвалось — файл повреждён. Пришлите видео ещё раз.", "crop_broken")
        mark("crop", t)

        t = time.perf_counter()
        _measure_luma(edl, staged, preset=style, cut_pauses=cut_pauses, cfg=cfg)
        mark("luma", t)

        # ---------------------------------------------------------- рендер
        await _notify(progress, "render")
        t = time.perf_counter()
        staged_info = probe(staged)
        plan = build_plan(
            edl,
            str(staged),
            fps=int(round(staged_info.fps)) or cfg.render_fps,
            width=staged_info.width,
            height=staged_info.height,
            preset=style,
            apply_cuts=cut_pauses,
            audio_path=audio_path,
            cfg=cfg,
            track=load_track(work / "track.json"),
        )
        (work / "plan.json").write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (work / "edl.json").write_text(edl.model_dump_json(indent=1), encoding="utf-8")
        # Слова с таймингами — единственный источник правды о времени (§3). Без них
        # разбор «подсветка уехала на слово вперёд» упирается в то, что EDL хранит
        # только индексы, а тайминги живут в памяти процесса и умирают вместе с ним.
        # Пишем ПОСЛЕ clean_transcript: именно эти слова уехали в режиссёра и рендер.
        (work / "words.json").write_text(
            json.dumps(
                {"language": asr.language, "backend": asr.backend, "model": asr.model,
                 "words": [w.model_dump() for w in words]},
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )

        out_path = work / "result.mp4"
        chosen = backend or cfg.render_backend
        if chosen == "ass":
            render_ass(plan, out_path, work_dir=work, cfg=cfg)
        else:
            async def on_render(done: float) -> None:
                if progress is None:
                    return
                res = progress(STAGE_TITLES["render"], render_share(done))
                if hasattr(res, "__await__"):
                    await res  # type: ignore[misc]

            await arender_remotion(
                plan, out_path, assets_dir=work, job_key=work.name, cfg=cfg,
                on_progress=on_render,
            )
        mark("render", t)

        metadata = await make_metadata(
            asr.text, edl.out_duration, llm=llm, language=asr.language, cfg=cfg
        )
        await _notify(progress, "done")

        result = JobResult(
            output=out_path,
            edl=edl,
            asr=asr,
            info=info,
            timings=timings,
            llm_report=llm.report() if llm else {},
            edl_from_llm=edl_from_llm,
            broll_used=broll_used,
            metadata=metadata,
            transcript_fixes=fixes,
            audio=audio_report,
        )
        log.info("джоб готов: %s", json.dumps(result.summary(), ensure_ascii=False))
        return result

    except Exception:
        if keep:
            log.info("DEBUG: рабочая папка оставлена для разбора — %s", work)
        raise


async def process_from_edl(
    src: str | Path,
    work_dir: str | Path,
    edl: EDL,
    *,
    preset: str | None = None,
    backend: str | None = None,
    cut_pauses: bool = True,
    vertical: bool = True,
    cfg: Settings | None = None,
) -> JobResult:
    """Пересборка ролика из готового EDL: без ASR и без LLM (§0.4).

    Нужна для двух вещей: перерендер с другим пресетом и честное сравнение фаз.
    Режиссёр недетерминирован, поэтому «до и после» имеет смысл только на одном и
    том же монтажном листе — иначе меряется разброс LLM, а не правка в коде.
    """
    cfg = cfg or default_settings
    src = Path(src)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def mark(stage: str, t0: float) -> None:
        timings[stage] = round(time.perf_counter() - t0, 2)

    t = time.perf_counter()
    info = probe(src)
    mark("probe", t)

    if preset:
        edl.preset = preset
    errors = validate_edl(edl)
    if errors:
        raise JobError("Монтажный лист не проходит валидацию: " + "; ".join(errors[:3]), "edl_invalid")

    t = time.perf_counter()
    cut_path = work / "cut.mp4"
    if cut_pauses:
        apply_cuts(edl, src, cut_path, info=info, cfg=cfg)
    else:
        shutil.copy2(src, cut_path)
    if not is_playable(cut_path):
        raise JobError("Нарезка оборвалась — файл повреждён.", "cut_broken")
    mark("cut", t)

    style = load_preset(edl.preset)
    audio_path: str | None = None
    audio_report: dict[str, Any] | None = None
    if cfg.audio_post_enabled:
        t = time.perf_counter()
        try:
            track = build_audio(edl, cut_path, work / "audio.wav", preset=style, cfg=cfg)
            if track is not None:
                audio_path = str(track.path)
                audio_report = {
                    "music": track.music,
                    "sfx": track.sfx,
                    "lufs_in": track.lufs_in,
                    "target_lufs": track.target_lufs,
                }
        except Exception as e:  # noqa: BLE001 — сравнение фаз важнее звука
            log.warning("звуковой пост пропущен: %s", e)
        mark("audio", t)

    t = time.perf_counter()
    staged = cut_path
    if vertical:
        staged = autocrop(
            cut_path, work / "vertical.mp4", work_dir=work,
            out_width=cfg.render_width, out_height=cfg.render_height,
            preset=style, framing=framing_spans(edl, applied_cuts=cut_pauses),
            cfg=cfg, track_out=work / "track.json",
        )
    if not is_playable(staged):
        raise JobError("Кадрирование оборвалось — файл повреждён.", "crop_broken")
    mark("crop", t)

    t = time.perf_counter()
    _measure_luma(edl, staged, preset=style, cut_pauses=cut_pauses, cfg=cfg)
    mark("luma", t)

    t = time.perf_counter()
    staged_info = probe(staged)
    plan = build_plan(
        edl,
        str(staged),
        fps=int(round(staged_info.fps)) or cfg.render_fps,
        width=staged_info.width,
        height=staged_info.height,
        preset=style,
        apply_cuts=cut_pauses,
        audio_path=audio_path,
        cfg=cfg,
        track=load_track(work / "track.json"),
    )
    (work / "plan.json").write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (work / "edl.json").write_text(edl.model_dump_json(indent=1), encoding="utf-8")

    out_path = work / "result.mp4"
    chosen = backend or cfg.render_backend
    if chosen == "ass":
        render_ass(plan, out_path, work_dir=work, cfg=cfg)
    else:
        await arender_remotion(plan, out_path, assets_dir=work, job_key=work.name, cfg=cfg)
    mark("render", t)

    return JobResult(
        output=out_path,
        edl=edl,
        asr=ASRResult(words=list(edl.words), language="", backend="edl", model="—"),
        info=info,
        timings=timings,
        audio=audio_report,
    )
