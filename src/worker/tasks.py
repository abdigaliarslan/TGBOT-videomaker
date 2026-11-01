"""M4: arq-воркеры и композиция job'ов.

Три пула (§5 M4): `gpu` — ASR, `cpu` — ffmpeg и рендер, `light` — LLM и мелочь.
Пул выбирается очередью, в которую положили задачу; здесь описаны сами задачи.

Прогресс живёт в Redis, а не в БД: бот читает его раз в 3 секунды, и делать ради
этого запись в постгрес на каждый шаг незачем.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from arq import cron, create_pool
from arq.connections import RedisSettings

from src.core.config import CLAIMED_PREFIX, UPLOAD_PREFIX, settings
from src.core.logging import setup_logging
from src.db import repo
from src.db.models import JobStatus
from src.core.options import JobOptions, apply_accent
from src.llm.client import LLMClient, Usage
from src.pipeline.job import STAGE_KEYS_BY_TITLE, JobError, process_video
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


# Пульс задачи: пока воркер жив, он подновляет этот ключ. Сторож (`reap_stuck_jobs`)
# по его отсутствию отличает мёртвую задачу от долгой — больше ничем они в базе не
# различаются. Срок жизни с запасом на четыре пропущенных удара: loop занят
# вытеснением тяжёлых этапов в поток, и ровный такт ему никто не гарантирует.
JOB_HEARTBEAT_TTL = 120
JOB_HEARTBEAT_EVERY = 30


def heartbeat_key(job_id: str) -> str:
    return f"reelbot:alive:{job_id}"


@asynccontextmanager
async def heartbeat(redis: Any, job_id: str) -> AsyncIterator[None]:
    """Держит пульс задачи, пока идёт работа внутри блока.

    Первый удар — ДО входа в блок, синхронно: иначе сторож, проснувшийся между
    `mark_running` и первым ударом, снимет живую задачу. Последний ключ
    снимается на выходе, чтобы доделанная задача не выглядела живой ещё две
    минуты.
    """
    key = heartbeat_key(job_id)
    await redis.set(key, "1", ex=JOB_HEARTBEAT_TTL)

    async def beat() -> None:
        while True:
            await asyncio.sleep(JOB_HEARTBEAT_EVERY)
            try:
                await redis.set(key, "1", ex=JOB_HEARTBEAT_TTL)
            except Exception as e:  # noqa: BLE001 — потеря удара не повод ронять монтаж
                log.warning("пульс задачи %s не поставлен: %s", job_id, e)

    task = asyncio.create_task(beat())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        with suppress(Exception):
            await redis.delete(key)


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
    raise JobError("Исходный файл потерялся. Пришлите видео ещё раз.", "source_lost")


# --------------------------------------------------------------------------- задача


async def process_job(ctx: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Полный прогон одного ролика: скачать → обработать → залить → отметить."""
    redis = ctx["redis"]
    job = await repo.get_job(job_id)
    if job is None:
        raise RuntimeError(f"задача {job_id} не найдена в БД")

    if not await repo.mark_running(job.id):
        # В работу берётся только задача из очереди. Сюда попадает та, которую
        # arq вернул в очередь после смерти воркера (ключ `in-progress` живёт
        # `job_timeout`) и которую сторож уже освободил, вернув ролик на баланс.
        # Собрать её второй раз значило бы отдать видео бесплатно.
        log.warning("задача %s уже не в очереди (%s) — второй прогон не нужен", job.id, job.status)
        return {"job_id": str(job.id), "skipped": "not_queued"}

    work = Path(settings.work_dir) / str(job.id)
    work.mkdir(parents=True, exist_ok=True)

    async def publish(stage: str, share: float, key: str | None = None) -> None:
        # `stage` — русский заголовок для логов и старых читателей, `stage_key` —
        # то, по чему интерфейс покажет стадию на языке пользователя.
        key = key or STAGE_KEYS_BY_TITLE.get(stage)
        await redis.set(
            progress_key(str(job.id)),
            json.dumps({"stage": stage, "stage_key": key, "progress": share}, ensure_ascii=False),
            ex=PROGRESS_TTL,
        )
        await repo.update_job(job.id, stage=stage[:32], progress=share)

    llm = LLMClient(usage_sink=_usage_sink(job.id))

    try:
        # Пульс — вокруг всей работы: пока он есть, сторож зависших задач знает,
        # что воркер жив, и не трогает задачу, сколько бы она ни шла.
        async with heartbeat(redis, str(job.id)):
            await publish("Скачиваю файл", 0.01, "download")
            src_path = await fetch_source(job, work / "source.mp4")

            options = JobOptions.from_stored(job.options)
            result = await process_video(
                src_path,
                work,
                preset=job.preset,
                llm=llm,
                progress=publish,
                cut_pauses=options.cut,
                cfg=settings,
                options=options,
            )

            await publish("Загружаю результат", 0.99, "upload")
            result_key = storage.key_for(RESULTS, str(job.id), "result.mp4")
            await storage.upload(result.output, result_key, content_type="video/mp4")

            await repo.mark_done(
                job.id,
                result_key=result_key,
                result_duration=round(result.edl.out_duration, 2),
                source_duration=result.info.duration,
                edl=json.loads(result.edl.model_dump_json()),
                words={
                    "words": [w.model_dump() for w in result.asr.words],
                    "language": result.asr.language,
                },
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
        await repo.mark_failed(job.id, str(e), code=e.code)
        await _refund(job.id)
        await redis.set(
            progress_key(str(job.id)),
            json.dumps(
                {"stage": "failed", "progress": 0.0, "error": str(e), "error_code": e.code},
                ensure_ascii=False,
            ),
            ex=PROGRESS_TTL,
        )
        raise
    except Exception as e:  # noqa: BLE001 — любая другая поломка тоже должна дойти до юзера
        log.exception("задача %s упала", job.id)
        await repo.mark_failed(job.id, f"{type(e).__name__}: {e}", code="failed")
        await _refund(job.id)
        await redis.set(
            progress_key(str(job.id)),
            json.dumps(
                {
                    "stage": "failed",
                    "progress": 0.0,
                    "error": "Не получилось смонтировать этот ролик. Мы записали ошибку.",
                    "error_code": "failed",
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


async def _refund(job_id: uuid.UUID) -> None:
    """Ролик не получился — оплата за него возвращается, в том числе за «короткое
    видео» и «нет речи». Платят за готовый монтаж, а не за попытку."""
    try:
        if await repo.refund_job(job_id):
            log.info("задача %s: ролик возвращён на баланс", job_id)
    except Exception as e:  # noqa: BLE001 — возврат не должен скрыть исходную ошибку
        log.error("задача %s: НЕ удалось вернуть ролик на баланс: %s", job_id, e)


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
        raise JobError("Монтажный лист не сохранился — пересобрать нечем.", "edl_missing")

    work = Path(settings.work_dir) / f"{job.id}-rerender"
    work.mkdir(parents=True, exist_ok=True)
    try:
        src_path = await fetch_source(job, work / "source.mp4")
        edl = EDL(**job.edl)
        edl.preset = preset

        # Тяжёлое — в рабочий поток, по той же причине, что и в `process_video`:
        # loop обязан дышать, иначе воркер не слышит SIGTERM от выкладки.
        info = await asyncio.to_thread(probe, src_path)
        # Перерендер меняет тему, а выбранный пользователем цвет выделения
        # сохраняет: он выбирал цвет, а не «цвет конкретной темы».
        options = JobOptions.from_stored(job.options)
        style = apply_accent(load_preset(preset), options.accent)
        # Нарезка — решение пользователя, и перерендер обязан его повторить: без
        # этого ролик, смонтированный без вырезания пауз, при смене темы
        # нарезался бы, а субтитры и звук разъехались бы с картинкой.
        cut_path = work / "cut.mp4"
        if options.cut:
            await asyncio.to_thread(apply_cuts, edl, src_path, cut_path, info=info, cfg=settings)
        else:
            await asyncio.to_thread(shutil.copy2, src_path, cut_path)
        # Пресет и крупность передаются ЯВНО. Без них перерендер брал цветокор из
        # `cfg.default_preset`, а не из пресета задачи, и снимал весь ролик одной
        # крупностью — то есть §0.4 «сохранённый EDL + другой пресет = новое
        # видео» выполнялся только для субтитров. `autocrop` про эту деградацию
        # предупреждает в своём докстринге; правка была за владельцем вызова.
        vertical = await asyncio.to_thread(
            autocrop,
            cut_path, work / "vertical.mp4", work_dir=work, cfg=settings,
            preset=style, framing=framing_spans(edl, applied_cuts=options.cut),
            track_out=work / "track.json",
        )
        staged_info = await asyncio.to_thread(probe, vertical)

        # Трек считается заново, а не берётся из кеша задачи. MOTION_PLAN просит
        # кешировать его в MinIO и не пересчитывать, но здесь всё равно заново
        # идёт и нарезка, и кроп — а трек живёт в координатах ВЫХОДНОГО кадра,
        # которые от пресета зависят (другая крупность — другое окно). Кеш дал бы
        # экономию на одном проходе YuNet ценой трека, который врёт про кадр.
        plan = build_plan(
            edl, str(vertical), fps=int(round(staged_info.fps)),
            width=staged_info.width, height=staged_info.height,
            preset=style, apply_cuts=options.cut, cfg=settings,
            track=load_track(work / "track.json"),
        )
        out = work / "result.mp4"
        if settings.render_backend == "ass":
            await asyncio.to_thread(render_ass, plan, out, work_dir=work, cfg=settings)
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


# Запас от последнего пульса до приговора. Больше срока жизни ключа — иначе
# сторож успеет застать живую задачу в промежутке между двумя ударами.
STUCK_GRACE = 5 * 60
# Ожидание в очереди, после которого задача считается потерянной. Длинное
# намеренно: пул `cpu` берёт один ролик за раз при потолке очереди в несколько
# задач, и честное ожидание здесь измеряется часами. Короткая отсечка снимала бы
# задачи, которые просто стоят за долгим роликом.
QUEUED_MAX_WAIT = 6 * 3600


async def reap_stuck_jobs(ctx: dict[str, Any]) -> dict[str, int]:
    """Освобождает задачи, за которыми больше некому следить.

    Воркера снимают не по своей воле: выкладка перезапускает службу, OOM-killer
    убивает процесс на тяжёлом ролике. Запись в БД остаётся RUNNING навсегда, и
    пользователь получает двойной штраф — `active_jobs_count` больше никогда не
    пустит его прислать видео, а списанный ролик не вернётся на баланс.

    Живую задачу от мёртвой отличает только пульс в Redis. Поэтому: сначала
    отсечка по времени (кандидаты, которым пульс уже пора было поставить),
    потом проверка пульса — и лишь затем приговор.

    ⚠ Потеря самого Redis (flush, переезд) выглядит отсюда как смерть всех
    воркеров: задачи в работе будут сняты и возвращены на баланс. Это
    осознанный перекос в пользу пользователя — в той же аварии теряется и
    очередь, то есть почти всегда он прав.
    """
    redis = ctx["redis"]
    now = datetime.now(timezone.utc)
    jobs = await repo.jobs_to_reap(
        running_before=now - timedelta(seconds=STUCK_GRACE),
        queued_before=now - timedelta(seconds=QUEUED_MAX_WAIT),
    )
    reaped = {"running": 0, "queued": 0}
    for job in jobs:
        queued = job.status == JobStatus.QUEUED
        if not queued and await redis.exists(heartbeat_key(str(job.id))):
            continue  # воркер жив, ролик просто долгий
        try:
            # Приговор условный: статус обязан быть всё тем же, по которому
            # задачу выбрали. Пульс снимается в конце прогона, поэтому между
            # выборкой и этой строкой ролик мог как раз доделаться — и возврат
            # за него был бы подарком вдобавок к видео.
            if not await repo.fail_if_still(
                job.id,
                "воркер не довёл задачу до конца: пульс пропал"
                if not queued
                else f"задача простояла в очереди больше {QUEUED_MAX_WAIT // 3600} ч",
                code="worker_lost",
                expect=job.status,
            ):
                log.info("сторож: задача %s разошлась с выборкой, оставляю как есть", job.id)
                continue
            await _refund(job.id)
            # Наблюдатель в чате читает прогресс из Redis, а не из БД: без этой
            # записи пользователь так и смотрел бы на полосу.
            await redis.set(
                progress_key(str(job.id)),
                json.dumps(
                    {"stage": "failed", "progress": 0.0, "error_code": "worker_lost"},
                    ensure_ascii=False,
                ),
                ex=PROGRESS_TTL,
            )
        except Exception as e:  # noqa: BLE001 — одна задача не должна оставить висеть остальные
            log.error("сторож: задачу %s освободить не удалось: %s", job.id, e)
            continue
        reaped["queued" if queued else "running"] += 1
        log.warning("сторож: задача %s освобождена (%s)", job.id, job.status)
    return reaped


# Общие папки внутри WORK_DIR: они не принадлежат ни одной задаче и живут
# своим сроком, поэтому у них смотрятся ФАЙЛЫ, а не сама папка. Снести их
# целиком по mtime папки означало бы выбросить живой кеш вместе с содержимым.
#
# `media-cache` — копии готовых роликов для перемотки в приложении.
# `broll-cache` — скачанные и НАРИСОВАННЫЕ вставки. Второе стоит денег:
# суточный срок сжигал оплаченные генерации каждый день заново, поэтому у него
# свой срок из конфига.
def _shared_caches() -> dict[str, float]:
    day = 24 * 3600
    return {
        "media-cache": day,
        "broll-cache": max(1, settings.broll_cache_days) * day,
    }


async def cleanup_old_files(ctx: dict[str, Any]) -> int:
    """Периодическая уборка рабочих папок, переживших падение воркера."""
    removed = 0
    now = datetime.now(timezone.utc).timestamp()
    cutoff = now - 24 * 3600
    # Брошенный приём файла из приложения — не папка задачи: за ним никто не
    # придёт, а держит он до гигабайта на свободном диске в 14 ГБ. Сутки для
    # него слишком долго. Живому приёму два часа не помеха: папка создаётся в
    # начале загрузки, и двух часов хватает даже гигабайту на мобильной сети.
    upload_cutoff = now - 2 * 3600
    root = Path(settings.work_dir)
    shared = _shared_caches()
    for child in root.iterdir() if root.exists() else []:
        if child.name in shared:
            continue  # не папка задачи — чистится пофайлово ниже
        if not child.is_dir():
            continue
        upload = child.name.startswith((UPLOAD_PREFIX, CLAIMED_PREFIX))
        if child.stat().st_mtime < (upload_cutoff if upload else cutoff):
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    for name, ttl in shared.items():
        cache = root / name
        stale = now - ttl
        for f in cache.rglob("*") if cache.is_dir() else []:
            if f.is_file() and f.stat().st_mtime < stale:
                f.unlink(missing_ok=True)
                removed += 1
    if removed:
        log.info("уборка: удалено %d старых рабочих папок и файлов кеша", removed)
    return removed


# Удаление по сроку: запас к сроку для прохода по самому хранилищу. Файлы задач
# удаляются точно в срок по базе, а проход по хранилищу подбирает то, чего база
# не знает (ролики перерендера), — и запас гарантирует, что он никогда не
# опередит учёт и не сотрёт файл задачи, пока та ещё не помечена.
STORAGE_SWEEP_GRACE = 24 * 3600


async def purge_expired_files(ctx: dict[str, Any]) -> dict[str, int]:
    """Удаляет исходники и готовые ролики старше `STORAGE_RETENTION_DAYS`.

    Ролик у пользователя уже лежит в чате — Telegram хранит его сам. На сервере
    копия нужна только для перерендера в другой стиль и плеера в приложении, и
    держать её вечно при 14 ГБ свободного диска нельзя.
    """
    from datetime import timedelta

    days = settings.storage_retention_days
    if days <= 0:
        return {"jobs": 0, "files": 0, "swept": 0}
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    jobs = await repo.jobs_with_expired_files(cutoff)
    files = 0
    for job in jobs:
        for key in {job.source_key, job.result_key} - {None}:
            try:
                await storage.delete(key)
                files += 1
            except Exception as e:  # noqa: BLE001 — один битый ключ не должен стопорить уборку
                log.warning("задача %s: не удалил %s: %s", job.id, key, e)
        await repo.mark_files_purged(job.id)

    swept = await storage.delete_older_than(cutoff - timedelta(seconds=STORAGE_SWEEP_GRACE))
    if jobs or swept:
        log.info(
            "срок хранения %d дн.: задач %d, файлов задач %d, прочих файлов %d",
            days, len(jobs), files, swept,
        )
    return {"jobs": len(jobs), "files": files, "swept": swept}


# --------------------------------------------------------------------------- воркеры


async def startup(ctx: dict[str, Any]) -> None:
    setup_logging(settings.log_level)
    repo.init_engine(settings)
    await storage.ensure_bucket_async()

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


async def light_startup(ctx: dict[str, Any]) -> None:
    """Старт пула уборки: без прогрева распознавания и Chromium.

    Унаследованный `startup` грузил модель Whisper и браузер рендера — уборщику
    они не нужны, а в его лимит памяти на сервере это просто не помещается.
    """
    setup_logging(settings.log_level)
    repo.init_engine(settings)
    log.info("уборщик поднят")


class LightWorkerSettings(WorkerSettings):
    """Пул `light`: уборка и мелкие задачи, не занимает CPU-слот.

    ⚠ arq читает настройки из `__dict__` класса, а не через наследование: всё,
    что не объявлено здесь явно, для него не существует. Без явного
    `redis_settings` пул подключался к Redis по умолчанию, localhost:6379, — на
    общем сервере это оказался Redis соседнего проекта.
    """

    redis_settings = WorkerSettings.redis_settings
    on_shutdown = WorkerSettings.on_shutdown
    max_tries = WorkerSettings.max_tries
    keep_result = WorkerSettings.keep_result
    on_startup = light_startup
    functions = [cleanup_old_files, purge_expired_files, reap_stuck_jobs]
    # Раз в час: папки задач, переживших падение воркера, и файлы старше срока
    # хранения. Второе — сразу при старте тоже: после долгого простоя ждать час
    # незачем, а работа идемпотентна.
    #
    # Сторож зависших задач — раз в пять минут и обязательно при старте: самая
    # частая причина его работы это выкладка, а она перезапускает и этот пул.
    # Час ожидания означал бы час, в котором пользователь не может прислать
    # видео и не видит своих денег.
    cron_jobs = [
        cron(cleanup_old_files, minute=17, run_at_startup=False),
        cron(purge_expired_files, minute=47, run_at_startup=True),
        cron(reap_stuck_jobs, minute={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55},
             run_at_startup=True),
    ]
    queue_name = QUEUE_LIGHT
    max_jobs = 4
    job_timeout = 600


class GpuWorkerSettings(WorkerSettings):
    """Пул `gpu`: ASR. На машине без GPU совпадает с cpu по поведению."""

    # Явно, а не наследованием: arq читает только `__dict__` класса.
    redis_settings = WorkerSettings.redis_settings
    on_startup = WorkerSettings.on_startup
    on_shutdown = WorkerSettings.on_shutdown
    max_tries = WorkerSettings.max_tries
    keep_result = WorkerSettings.keep_result

    functions = [process_job]
    queue_name = QUEUE_GPU
    max_jobs = 1
    job_timeout = 3 * 3600
