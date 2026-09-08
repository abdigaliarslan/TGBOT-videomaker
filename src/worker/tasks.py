"""M4: arq-воркеры и композиция job'ов.

Три пула (§5 M4): `gpu` — ASR, `cpu` — ffmpeg и рендер, `light` — LLM и мелочь.
Пул выбирается очередью, в которую положили задачу; здесь описаны сами задачи.

Прогресс живёт в Redis, а не в БД: бот читает его раз в 3 секунды, и делать ради
этого запись в постгрес на каждый шаг незачем.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arq import cron, create_pool
from arq.connections import RedisSettings

from src.core.config import settings
from src.core.logging import setup_logging
from src.db import repo
from src.llm.client import LLMClient, Usage
from src.pipeline.job import JobError, process_video
from src.storage.s3 import RESULTS, SOURCES, storage

log = logging.getLogger(__name__)

QUEUE_GPU = "arq:queue:gpu"
QUEUE_CPU = "arq:queue:cpu"
QUEUE_LIGHT = "arq:queue:light"

PROGRESS_TTL = 3 * 3600


def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(settings.redis_url)


async def get_queue(name: str = QUEUE_CPU):
    return await create_pool(redis_settings(), default_queue_name=name)


def progress_key(job_id: str) -> str:
    return f"reelbot:progress:{job_id}"


async def fetch_source(job: Any, dst: Path) -> Path:
    """Кладёт исходник джоба в `dst`, откуда бы он ни был доступен.

    Локальный Bot API оставляет файл на общем томе — тогда качать из S3 незачем.
    """
    if job.source_local_path and Path(job.source_local_path).exists():
        shutil.copy2(job.source_local_path, dst)
        return dst
    if job.source_key:
        await storage.download(job.source_key, dst)
        return dst
    raise JobError("Исходный файл потерялся. Пришлите видео ещё раз.")


# --------------------------------------------------------------------------- задача


async def process_job(ctx: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Полный прогон одного ролика: скачать → обработать → залить → отметить."""
    redis = ctx["redis"]
    job = await repo.get_job(job_id)
    if job is None:
        raise RuntimeError(f"задача {job_id} не найдена в БД")

    work = Path(settings.work_dir) / str(job.id)
    work.mkdir(parents=True, exist_ok=True)
    await repo.mark_running(job.id)

    async def publish(stage: str, share: float) -> None:
        await redis.set(
            progress_key(str(job.id)),
            json.dumps({"stage": stage, "progress": share}, ensure_ascii=False),
            ex=PROGRESS_TTL,
        )
        await repo.update_job(job.id, stage=stage[:32], progress=share)

    llm = LLMClient(usage_sink=_usage_sink(job.id))

    try:
        await publish("Скачиваю файл", 0.01)
        src_path = await fetch_source(job, work / "source.mp4")

        result = await process_video(
            src_path,
            work,
            preset=job.preset,
            llm=llm,
            progress=publish,
            cfg=settings,
        )

        await publish("Загружаю результат", 0.99)
        result_key = storage.key_for(RESULTS, str(job.id), "result.mp4")
        await storage.upload(result.output, result_key, content_type="video/mp4")

        await repo.mark_done(
            job.id,
            result_key=result_key,
            result_duration=round(result.edl.out_duration, 2),
            source_duration=result.info.duration,
            edl=json.loads(result.edl.model_dump_json()),
            words={"words": [w.model_dump() for w in result.asr.words], "language": result.asr.language},
            timings=result.timings,
            edl_from_llm=result.edl_from_llm,
            cost_usd=llm.total_cost(),
            stage="done",
        )
        await repo.bump_videos_done(job.user_id)
        await redis.set(
            progress_key(str(job.id)),
            json.dumps({"stage": "done", "progress": 1.0, "result_key": result_key}),
            ex=PROGRESS_TTL,
        )
        return {"job_id": str(job.id), "result_key": result_key, **result.summary()}

    except JobError as e:
        # Понятная пользователю причина: показываем как есть.
        log.warning("задача %s отклонена: %s", job.id, e)
        await repo.mark_failed(job.id, str(e))
        await redis.set(
            progress_key(str(job.id)),
            json.dumps({"stage": "failed", "progress": 0.0, "error": str(e)}, ensure_ascii=False),
            ex=PROGRESS_TTL,
        )
        raise
    except Exception as e:  # noqa: BLE001 — любая другая поломка тоже должна дойти до юзера
        log.exception("задача %s упала", job.id)
        await repo.mark_failed(job.id, f"{type(e).__name__}: {e}")
        await redis.set(
            progress_key(str(job.id)),
            json.dumps(
                {
                    "stage": "failed",
                    "progress": 0.0,
                    "error": "Не получилось смонтировать этот ролик. Мы записали ошибку.",
                },
                ensure_ascii=False,
            ),
            ex=PROGRESS_TTL,
        )
        raise
    finally:
        if not settings.debug:
            shutil.rmtree(work, ignore_errors=True)
        else:
            log.info("DEBUG=1: рабочая папка сохранена — %s", work)


def _usage_sink(job_id: uuid.UUID):
    async def sink(usage: Usage) -> None:
        try:
            await repo.record_llm_call(usage, job_id)
        except Exception as e:  # noqa: BLE001 — учёт не должен ронять рендер
            log.warning("не записал расход токенов: %s", e)

    return sink


async def rerender_job(ctx: dict[str, Any], job_id: str, preset: str) -> dict[str, Any]:
    """Перерендер из сохранённого EDL с другим пресетом — без ASR и LLM (§0.4)."""
    from src.core.edl import EDL
    from src.core.preset import load_preset
    from src.core.render_plan import build_plan
    from src.core.track import load_track
    from src.pipeline.crop import autocrop, framing_spans
    from src.pipeline.cut import apply_cuts
    from src.pipeline.probe import probe
    from src.pipeline.render import arender_remotion, render_ass

    redis = ctx["redis"]
    job = await repo.get_job(job_id)
    if job is None:
        raise RuntimeError(f"задача {job_id} не найдена")
    if not job.edl:
        raise JobError("Монтажный лист не сохранился — пересобрать нечем.")

    work = Path(settings.work_dir) / f"{job.id}-rerender"
    work.mkdir(parents=True, exist_ok=True)
    try:
        src_path = await fetch_source(job, work / "source.mp4")
        edl = EDL(**job.edl)
        edl.preset = preset

        info = probe(src_path)
        style = load_preset(preset)
        cut_path = apply_cuts(edl, src_path, work / "cut.mp4", info=info, cfg=settings)
        # Пресет и крупность передаются ЯВНО. Без них перерендер брал цветокор из
        # `cfg.default_preset`, а не из пресета задачи, и снимал весь ролик одной
        # крупностью — то есть §0.4 «сохранённый EDL + другой пресет = новое
        # видео» выполнялся только для субтитров. `autocrop` про эту деградацию
        # предупреждает в своём докстринге; правка была за владельцем вызова.
        vertical = autocrop(
            cut_path, work / "vertical.mp4", work_dir=work, cfg=settings,
            preset=style, framing=framing_spans(edl),
            track_out=work / "track.json",
        )
        staged_info = probe(vertical)

        # Трек считается заново, а не берётся из кеша задачи. MOTION_PLAN просит
        # кешировать его в MinIO и не пересчитывать, но здесь всё равно заново
        # идёт и нарезка, и кроп — а трек живёт в координатах ВЫХОДНОГО кадра,
        # которые от пресета зависят (другая крупность — другое окно). Кеш дал бы
        # экономию на одном проходе YuNet ценой трека, который врёт про кадр.
        plan = build_plan(
            edl, str(vertical), fps=int(round(staged_info.fps)),
            width=staged_info.width, height=staged_info.height,
            preset=style, cfg=settings,
            track=load_track(work / "track.json"),
        )
        out = work / "result.mp4"
        if settings.render_backend == "ass":
            render_ass(plan, out, work_dir=work, cfg=settings)
        else:
            await arender_remotion(plan, out, assets_dir=work, job_key=work.name, cfg=settings)

        key = storage.key_for(RESULTS, f"{job.id}-{preset}", "result.mp4")
        await storage.upload(out, key, content_type="video/mp4")
        await redis.set(
            progress_key(f"{job_id}:{preset}"),
            json.dumps({"stage": "done", "progress": 1.0, "result_key": key}),
            ex=PROGRESS_TTL,
        )
        return {"job_id": str(job.id), "preset": preset, "result_key": key}
    finally:
        if not settings.debug:
            shutil.rmtree(work, ignore_errors=True)


async def cleanup_old_files(ctx: dict[str, Any]) -> int:
    """Периодическая уборка рабочих папок, переживших падение воркера."""
    removed = 0
    cutoff = datetime.now(timezone.utc).timestamp() - 24 * 3600
    root = Path(settings.work_dir)
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and child.stat().st_mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    if removed:
        log.info("уборка: удалено %d старых рабочих папок", removed)
    return removed


# --------------------------------------------------------------------------- воркеры


async def startup(ctx: dict[str, Any]) -> None:
    setup_logging(settings.log_level)
    repo.init_engine(settings)
    await storage.ensure_bucket_async()
    import asyncio

    # Всё тяжёлое качаем на старте, а не внутри пользовательской задачи:
    # иначе первый ролик на новом языке упирается в 1.2 ГБ загрузки, а
    # оборвавшаяся закачка вешает задачу целиком.
    from src.pipeline.asr import prewarm

    try:
        await asyncio.to_thread(prewarm)
    except Exception as e:  # noqa: BLE001 — без прогрева работать можно, просто медленнее
        log.warning("прогрев ASR не удался (%s) — модели подтянутся на первом ролике", e)

    if settings.render_backend == "remotion":
        # Chromium скачивается один раз при старте, а не на каждый рендер (M5 ⚠).
        from src.pipeline.render import ensure_browser, ensure_bundle

        await asyncio.to_thread(ensure_browser)
        await asyncio.to_thread(ensure_bundle)
    log.info("воркер поднят: очередь=%s", ctx.get("queue_name", "?"))


async def shutdown(ctx: dict[str, Any]) -> None:
    await repo.dispose_engine()


class WorkerSettings:
    """Пул `cpu`: ffmpeg и рендер. Тяжёлый, поэтому параллелизм маленький."""

    functions = [process_job, rerender_job]
    cron_jobs = []
    on_startup = startup
    on_shutdown = shutdown
    queue_name = QUEUE_CPU
    max_jobs = settings.max_concurrent_jobs
    job_timeout = 3 * 3600
    keep_result = 3600
    max_tries = 1  # ретраить полный монтаж бессмысленно и дорого
    # arq читает это как атрибут-значение, а не как метод.
    redis_settings = redis_settings()


class LightWorkerSettings(WorkerSettings):
    """Пул `light`: уборка и мелкие задачи, не занимает CPU-слот."""

    functions = [cleanup_old_files]
    # Раз в час: подчищаем папки джобов, переживших падение воркера.
    cron_jobs = [cron(cleanup_old_files, minute=17, run_at_startup=False)]
    queue_name = QUEUE_LIGHT
    max_jobs = 4
    job_timeout = 600


class GpuWorkerSettings(WorkerSettings):
    """Пул `gpu`: ASR. На машине без GPU совпадает с cpu по поведению."""

    functions = [process_job]
    queue_name = QUEUE_GPU
    max_jobs = 1
    job_timeout = 3 * 3600
