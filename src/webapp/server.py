"""HTTP-сторона Mini App: статика и JSON API. Живёт в процессе бота.

Почему в процессе бота, а не отдельным сервисом: приложению нужен тот же `Bot` —
выставить счёт на звёзды, написать пользователю статус, отдать ролик в чат, — и
те же наблюдатели прогресса, что у загрузки из чата. Второй процесс держал бы
второе соединение с Telegram ради того, что здесь уже есть.

Кто пришёл, решает только подпись `initData` (заголовок `Authorization: tma …`).
Ни одному полю из тела запроса про пользователя сервер не верит.

Загрузка идёт кусками. Бесплатный туннель Cloudflare режет тело запроса на
100 МБ, а трёхминутное видео с телефона бывает больше; заодно оборванную на
мобильной сети загрузку можно продолжить с того байта, на котором она встала.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from aiogram import Bot
from aiogram.types import LabeledPrice
from aiohttp import web

from src.core.config import UPLOAD_PREFIX, Settings
from src.core.i18n import LANGS, error_text, lang_of, stage_title, t, videos_word
from src.core.options import ACCENTS, JobOptions
from src.core.preset import available, load_preset
from src.db import repo
from src.db.models import Job, JobStatus
from src.storage.s3 import SOURCES, storage
from src.webapp.auth import AuthError, WebAppUser, validate_init_data

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
_UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")

BOT_KEY = web.AppKey("bot", Bot)
CFG_KEY = web.AppKey("cfg", Settings)
USER_KEY = web.RequestKey("webapp_user", WebAppUser)
LANG_KEY = web.RequestKey("webapp_lang", str)


# ------------------------------------------------------------------ вспомогательное


def _json_error(status: int, message: str, **extra: Any) -> web.Response:
    return web.json_response({"error": message, **extra}, status=status)


async def _lang(request: web.Request) -> str:
    """Язык интерфейса пользователя. Читается из БД один раз на запрос и только когда нужен."""
    if LANG_KEY in request:
        return request[LANG_KEY]
    u: WebAppUser = request[USER_KEY]
    try:
        db_user = await repo.get_user(u.id)
    except Exception:  # noqa: BLE001 — нет базы: язык по клиенту Telegram
        db_user = None
    lang = lang_of(db_user) if db_user else lang_of(u)
    request[LANG_KEY] = lang
    return lang


async def _fail(request: web.Request, status: int, key: str, *, extra: dict[str, Any] | None = None,
                **params: Any) -> web.Response:
    return _json_error(status, t(key, await _lang(request), **params), **(extra or {}))


def media_token(job_id: str, secret: str, *, ttl_s: int = 6 * 3600, now: float | None = None) -> str:
    """Подписанная ссылка на ролик для тега <video>.

    Браузер грузит видео сам и заголовок авторизации к нему не приложит, поэтому
    право посмотреть ролик едет в адресе: id задачи, срок и подпись ключом бота.
    """
    exp = int((time.time() if now is None else now) + ttl_s)
    return f"{exp}.{_media_sig(job_id, exp, secret)}"


def _media_sig(job_id: str, exp: int, secret: str) -> str:
    # Ключ выводится из токена, а не равен ему: подпись ссылки — не то место,
    # где сырой токен бота должен работать ключом.
    key = hashlib.sha256(b"reelbot-media:" + secret.encode()).digest()
    return hmac.new(key, f"{job_id}:{exp}".encode(), hashlib.sha256).hexdigest()[:32]


def check_media_token(job_id: str, token: str, secret: str, *, now: float | None = None) -> bool:
    try:
        exp_s, sig = token.split(".", 1)
        exp = int(exp_s)
    except ValueError:
        return False
    if exp < (time.time() if now is None else now):
        return False
    return hmac.compare_digest(_media_sig(job_id, exp, secret), sig)


def _upload_dir(cfg: Settings, upload_id: str) -> Path:
    # Папка лежит прямо в work_dir: её подберёт почасовая уборка старых папок,
    # если пользователь закрыл приложение посреди загрузки.
    return Path(cfg.work_dir) / f"{UPLOAD_PREFIX}{upload_id}"


# Сколько незавершённых приёмов разрешено одному пользователю. Приём занимает
# диск до самого конца загрузки, а брошенную папку убирает только уборщик.
# При разрешённом гигабайте на файл десяток брошенных приёмов съедает весь
# свободный диск сервера — вместе с местом под Postgres, который там же.
# Два — чтобы повтор на оборванной мобильной сети не упирался в отказ.
MAX_OPEN_UPLOADS = 2
# Запас свободного места, который приёму не отдаётся ни при каких условиях:
# под базу, журналы и рабочие папки самого монтажа.
DISK_RESERVE_MB = 4096
# Сколько живёт заявка на место без единого нового куска. Заявка держит место
# за собой (иначе учёт обещанного ничего не стоит), а объявить гигабайт и
# ничего не прислать стоит одного запроса: пяти аккаунтов хватило бы, чтобы
# закрыть загрузку всем до прихода уборщика. Срок продлевает каждый кусок,
# поэтому живой загрузке он не мешает: кусок — 8 МБ, и четверть часа на него
# не уходит даже на плохой мобильной сети.
UPLOAD_LEASE = 15 * 60


def _sweep_stale_uploads(cfg: Settings) -> list[tuple[Path, dict[str, Any]]]:
    """Сносит заявки с истёкшим сроком и отдаёт живые, старые первыми.

    Снос здесь, а не только в уборщике: просроченная заявка иначе держала бы и
    место, и квоту своего хозяина — то есть пользователь, у которого
    приложение закрылось посреди загрузки, не смог бы начать заново.
    """
    root = Path(cfg.work_dir)
    alive: list[tuple[Path, dict[str, Any]]] = []
    stale_before = time.time() - UPLOAD_LEASE
    for folder in root.glob(f"{UPLOAD_PREFIX}*") if root.exists() else []:
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            data = folder / "data"
            # Срок считается по файлу данных: его трогает каждый кусок.
            touched = (data if data.exists() else folder).stat().st_mtime
        except (OSError, ValueError):
            continue  # папка недописана или уже убрана — не наше дело
        if not isinstance(meta, dict):
            continue
        if touched < stale_before:
            shutil.rmtree(folder, ignore_errors=True)
            continue
        alive.append((folder, meta))
    return sorted(alive, key=lambda pair: pair[0].stat().st_mtime)


def _promised_bytes(open_uploads: list[tuple[Path, dict[str, Any]]]) -> int:
    """Сколько байт уже обещано приёмам, но ещё не легло на диск.

    Без этого счёта проверка места бессмысленна: байты незавершённого приёма на
    диске не лежат, и `disk_usage` показывает их место свободным. Лимит на
    пользователя тут не помогает — десять человек по два приёма обещают
    двадцать гигабайт на диске, где их нет.
    """
    promised = 0
    for folder, meta in open_uploads:
        try:
            have = (folder / "data").stat().st_size
        except OSError:
            have = 0
        promised += max(0, int(meta.get("size") or 0) - have)
    return promised


def _job_view(job: Job, *, live: dict[str, Any] | None, secret: str, lang: str, cfg: Settings) -> dict[str, Any]:
    from src.pipeline.job import STAGE_KEYS_BY_TITLE

    progress = job.progress
    status = job.status.value if isinstance(job.status, JobStatus) else str(job.status)
    stage_key = STAGE_KEYS_BY_TITLE.get(job.stage or "")
    if live and status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
        stage_key = live.get("stage_key") or STAGE_KEYS_BY_TITLE.get(live.get("stage") or "") or stage_key
        progress = float(live.get("progress") or progress or 0.0)
    stage = stage_title(stage_key, lang)
    if stage is None and status == JobStatus.QUEUED.value:
        stage = t("stage.queued", lang)
    error = None
    if status == JobStatus.FAILED.value:
        # Пользователю — фраза по коду на его языке, а не русский текст из `jobs.error`.
        error = error_text(job.error_code, lang, min_s=cfg.min_video_seconds, max_m=cfg.max_video_minutes)
    view: dict[str, Any] = {
        "id": str(job.id),
        "status": status,
        "stage": stage,
        "progress": round(float(progress or 0.0), 3),
        "preset": job.preset,
        "options": JobOptions.from_stored(job.options).model_dump(),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "source_duration": job.source_duration,
        "result_duration": job.result_duration,
        "error": error,
        "delivered": job.delivered,
        "video_url": None,
        "purged": job.files_purged_at is not None,
    }
    if status == JobStatus.DONE.value and job.result_key:
        view["video_url"] = f"/media/{job.id}.mp4?t={media_token(str(job.id), secret)}"
    return view


# ------------------------------------------------------------------ авторизация


@web.middleware
async def auth_middleware(request: web.Request, handler):
    if not request.path.startswith("/api/"):
        return await handler(request)
    cfg = request.app[CFG_KEY]
    header = request.headers.get("Authorization", "")
    init_data = header[4:] if header.startswith("tma ") else ""
    try:
        user = validate_init_data(init_data, cfg.bot_token, max_age_s=cfg.webapp_auth_max_age_s)
    except AuthError as e:
        return _json_error(401, str(e))
    request[USER_KEY] = user
    return await handler(request)


async def _db_user(request: web.Request):
    u: WebAppUser = request[USER_KEY]
    cfg = request.app[CFG_KEY]
    return await repo.upsert_user(
        u.id,
        username=u.username,
        first_name=u.first_name,
        language_code=u.language_code,
        is_admin=u.id in cfg.admin_ids,
    )


# ------------------------------------------------------------------ API


async def bootstrap(request: web.Request) -> web.Response:
    """Всё, что приложению нужно для первого экрана, одним запросом."""
    cfg = request.app[CFG_KEY]
    bot = request.app[BOT_KEY]
    user = await _db_user(request)
    lang = lang_of(user)
    request[LANG_KEY] = lang
    me = await bot.me()

    presets = []
    # Тема по умолчанию — первой: это первая вкладка, которую видит пользователь.
    names = sorted(available(), key=lambda n: (n != cfg.default_preset, n))
    for name in names:
        try:
            p = load_preset(name)
        except Exception:  # noqa: BLE001 — битый пресет не должен ронять приложение
            continue
        presets.append(
            {
                "name": name,
                "title": p.title or name,
                "font": p.font.family,
                "font_file": p.font.file,
                "letter_spacing": p.font.letter_spacing,
                "uppercase": p.font.uppercase,
                "colors": {"text": p.colors.active, "accent": p.colors.accent},
                # Числа движения — чтобы превью показывало ту тему, которую
                # выбрали, а не одно и то же для всех. Подъём приводится к
                # ДОЛЕ кегля: превью рисуется своим шрифтом в своём масштабе,
                # и пиксели кадра 1920 там ничего не значат.
                "motion": {
                    "word_in_ms": p.animation.word_in_ms,
                    "word_rise_em": round(p.animation.word_rise_px / max(1, p.font.size), 3),
                },
            }
        )
    return web.json_response(
        {
            "user": {
                "id": user.id,
                "first_name": user.first_name,
                "username": user.username,
                "is_admin": user.is_admin,
                "balance": user.balance,
                "videos_done": user.videos_done,
                "preset": user.preset,
                "lang": lang,
            },
            "bot_username": me.username,
            "bot_name": me.first_name,
            "presets": presets,
            "accents": [
                {"key": k, "title": titles.get(lang, titles["ru"]), "color": color}
                for k, (titles, color) in ACCENTS.items()
            ],
            "packages": [{"videos": v, "stars": s} for v, s in cfg.star_packages.items()],
            "limits": {
                "min_seconds": cfg.min_video_seconds,
                "max_minutes": cfg.max_video_minutes,
                "max_upload_mb": cfg.webapp_max_upload_mb,
                "chunk_bytes": cfg.webapp_chunk_mb * 1024 * 1024,
            },
            "defaults": JobOptions().model_dump(),
            "retention_days": cfg.storage_retention_days,
        }
    )


async def upload_start(request: web.Request) -> web.Response:
    cfg = request.app[CFG_KEY]
    user: WebAppUser = request[USER_KEY]
    body = await request.json()
    size = int(body.get("size") or 0)
    if size <= 0:
        return await _fail(request, 400, "api.empty_file")
    if size > cfg.webapp_max_upload_mb * 1024 * 1024:
        return await _fail(request, 413, "api.too_big", mb=cfg.webapp_max_upload_mb)
    open_uploads = _sweep_stale_uploads(cfg)
    if sum(1 for _, meta in open_uploads if meta.get("user_id") == user.id) >= MAX_OPEN_UPLOADS:
        return await _fail(request, 429, "api.too_many_uploads", n=MAX_OPEN_UPLOADS)
    # Отказ до первого байта, а не на середине: пользователь с гигабайтным
    # файлом иначе узнаёт о нехватке места через десять минут загрузки.
    #
    # Из свободного места вычитается обещанное другим приёмам: их байты ещё не
    # пришли, и без этого вычета каждый новый приём получал бы одно и то же
    # место заново. Диск может кончиться и снаружи — машина общая, — тогда
    # запись упадёт по ходу загрузки; это уже не наш учёт, а чужой сосед.
    free = shutil.disk_usage(cfg.work_dir).free - DISK_RESERVE_MB * 1024 * 1024
    if size > free - _promised_bytes(open_uploads):
        return await _fail(request, 507, "api.no_space")

    upload_id = uuid.uuid4().hex
    folder = _upload_dir(cfg, upload_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data").touch()
    (folder / "meta.json").write_text(
        json.dumps({"user_id": user.id, "size": size, "name": str(body.get("name") or "")[:200]}),
        encoding="utf-8",
    )
    return web.json_response(
        {"upload_id": upload_id, "chunk_bytes": cfg.webapp_chunk_mb * 1024 * 1024, "received": 0}
    )


async def _own_upload(request: web.Request) -> tuple[Path, dict[str, Any]] | web.Response:
    cfg = request.app[CFG_KEY]
    user: WebAppUser = request[USER_KEY]
    upload_id = request.match_info["upload_id"]
    if not _UPLOAD_ID.match(upload_id):
        return await _fail(request, 404, "api.upload_not_found")
    folder = _upload_dir(cfg, upload_id)
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return await _fail(request, 404, "api.upload_not_found")
    if meta.get("user_id") != user.id:
        return await _fail(request, 404, "api.upload_not_found")
    return folder, meta


async def upload_chunk(request: web.Request) -> web.Response:
    own = await _own_upload(request)
    if isinstance(own, web.Response):
        return own
    folder, meta = own
    data_path = folder / "data"
    have = data_path.stat().st_size
    try:
        offset = int(request.query.get("offset", "-1"))
    except ValueError:
        return await _fail(request, 400, "api.bad_offset")

    chunk = await request.read()
    end = offset + len(chunk)
    if end > meta["size"]:
        return await _fail(request, 400, "api.chunk_too_far")
    if end <= have:
        # Повтор уже принятого куска: ответ потерялся на мобильной сети, а байты дошли.
        return web.json_response({"received": have})
    if offset != have:
        # Клиент и сервер разошлись в том, сколько принято. Ответ говорит, откуда
        # продолжать, — клиент перешлёт с нужного места, а не с нуля.
        return await _fail(request, 409, "api.offset_mismatch", extra={"received": have})

    def append() -> None:
        with data_path.open("ab") as fh:
            fh.write(chunk)

    await asyncio.to_thread(append)
    return web.json_response({"received": end})


async def upload_finish(request: web.Request) -> web.Response:
    """Файл целиком у нас → проверка → списание → очередь. Порядок важен.

    Проверка видео идёт ДО списания: не-видео и видео не той длины отбиваются,
    не трогая баланс. Всё, что может сломаться после списания (хранилище,
    очередь), возвращает ролик на баланс.
    """
    from src.bot.handlers.video import watch_job  # отложенно: хендлеры тянут весь бот
    from src.pipeline.probe import probe
    from src.worker.tasks import QUEUE_CPU, get_queue

    cfg = request.app[CFG_KEY]
    bot = request.app[BOT_KEY]
    own = await _own_upload(request)
    if isinstance(own, web.Response):
        return own
    folder, meta = own
    data_path = folder / "data"
    if data_path.stat().st_size != meta["size"]:
        return await _fail(request, 409, "api.incomplete", extra={"received": data_path.stat().st_size})
    # Продлеваем срок заявки на время самой сборки: дальше идут ffprobe и
    # заливка в хранилище, кусков больше не будет, и чужой `upload_start` иначе
    # имеет право снести папку у нас из-под рук.
    os.utime(data_path)

    body = await request.json()
    preset = str(body.get("preset") or "")
    if preset not in available():
        return await _fail(request, 400, "api.unknown_style")
    options = JobOptions.model_validate(body.get("options") or {})

    user = await _db_user(request)
    lang = lang_of(user)
    request[LANG_KEY] = lang
    limits = {"min_s": cfg.min_video_seconds, "max_m": cfg.max_video_minutes}
    if user.is_blocked:
        return await _fail(request, 403, "api.blocked")
    if not user.is_admin and await repo.active_jobs_count(user.id) >= 1:
        return await _fail(request, 429, "video.one_in_flight")
    if await repo.active_jobs_count() >= cfg.queue_max_size:
        return await _fail(request, 503, "video.queue_full")

    try:
        info = await asyncio.to_thread(probe, data_path)
    except Exception:  # noqa: BLE001 — ffprobe не прочитал: это не видео или битый файл
        shutil.rmtree(folder, ignore_errors=True)
        return await _fail(request, 415, "api.unreadable")
    if info.duration < cfg.min_video_seconds:
        shutil.rmtree(folder, ignore_errors=True)
        return _json_error(400, error_text("too_short", lang, **limits))
    if info.duration > cfg.max_video_seconds:
        shutil.rmtree(folder, ignore_errors=True)
        return _json_error(400, error_text("too_long", lang, **limits))
    if not info.has_audio:
        shutil.rmtree(folder, ignore_errors=True)
        return _json_error(400, error_text("no_audio", lang, **limits))

    try:
        job = await repo.create_job(
            user.id,
            source_file_id=None,
            source_key=None,
            source_size=meta["size"],
            preset=preset,
            chat_id=user.id,
            options=options.model_dump(),
            charge=not user.is_admin,
        )
    except repo.InsufficientBalance:
        return await _fail(request, 402, "api.no_balance", extra={"balance": user.balance})

    try:
        key = storage.key_for(SOURCES, str(job.id), "source.mp4")
        await storage.upload(data_path, key, content_type="video/mp4")
        await repo.update_job(job.id, source_key=key, source_duration=info.duration, preset=preset)
        await repo.set_user_preset(user.id, preset)

        status_id: int | None = None
        try:
            status = await bot.send_message(user.id, t("video.accepted_app", lang))
            status_id = status.message_id
            await repo.update_job(job.id, status_message_id=status_id)
        except Exception as e:  # noqa: BLE001 — нет чата с ботом: прогресс увидят в приложении
            log.warning("задача %s: не смог написать в чат %s: %s", job.id, user.id, e)

        queue = await get_queue(QUEUE_CPU)
        try:
            await queue.enqueue_job("process_job", str(job.id), _queue_name=QUEUE_CPU)
        finally:
            await queue.close()
    except Exception as e:  # noqa: BLE001
        log.exception("задача %s: не поставилась в очередь", job.id)
        await repo.mark_failed(job.id, f"постановка в очередь: {e}")
        await repo.refund_job(job.id)
        return await _fail(request, 500, "api.enqueue_failed")
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    if status_id is not None:
        watch_job(bot, user.id, status_id, str(job.id), preset)
    fresh = await repo.get_job(job.id)
    return web.json_response(
        {"job": _job_view(fresh or job, live=None, secret=cfg.bot_token, lang=lang, cfg=cfg)}
    )


async def _live_progress(job_ids: list[str]) -> dict[str, dict[str, Any]]:
    from src.worker.tasks import get_queue, progress_key

    if not job_ids:
        return {}
    redis = await get_queue()
    try:
        raws = await redis.mget([progress_key(j) for j in job_ids])
    finally:
        await redis.close()
    out: dict[str, dict[str, Any]] = {}
    for job_id, raw in zip(job_ids, raws):
        if raw:
            try:
                out[job_id] = json.loads(raw if isinstance(raw, str) else raw.decode())
            except ValueError:
                pass
    return out


async def jobs_list(request: web.Request) -> web.Response:
    cfg = request.app[CFG_KEY]
    user: WebAppUser = request[USER_KEY]
    jobs = await repo.user_jobs(user.id)
    active = [
        str(j.id) for j in jobs if j.status in (JobStatus.QUEUED, JobStatus.RUNNING)
    ]
    live = await _live_progress(active)
    db_user = await repo.get_user(user.id)
    lang = lang_of(db_user) if db_user else lang_of(user)
    return web.json_response(
        {
            "jobs": [
                _job_view(j, live=live.get(str(j.id)), secret=cfg.bot_token, lang=lang, cfg=cfg)
                for j in jobs
            ],
            "balance": db_user.balance if db_user else 0,
        }
    )


async def job_send(request: web.Request) -> web.Response:
    """Прислать готовый ролик в чат ещё раз — если сообщение потерялось или удалено."""
    from src.bot.handlers.video import send_result

    user: WebAppUser = request[USER_KEY]
    job = await _own_job(request, user)
    if job is None:
        return await _fail(request, 404, "api.video_not_found")
    if job.files_purged_at is not None:
        return await _fail(request, 410, "api.purged")
    if job.status != JobStatus.DONE or not job.result_key:
        return await _fail(request, 409, "api.not_ready")
    try:
        await send_result(request.app[BOT_KEY], user.id, job)
    except Exception as e:  # noqa: BLE001
        log.warning("задача %s: повторная отправка не удалась: %s", job.id, e)
        return await _fail(request, 502, "api.resend_failed")
    return web.json_response({"ok": True})


async def _own_job(request: web.Request, user: WebAppUser) -> Job | None:
    try:
        job = await repo.get_job(request.match_info["job_id"])
    except ValueError:
        return None
    if job is None or job.user_id != user.id:
        return None
    return job


async def invoice(request: web.Request) -> web.Response:
    cfg = request.app[CFG_KEY]
    bot = request.app[BOT_KEY]
    lang = lang_of(await _db_user(request))
    request[LANG_KEY] = lang
    body = await request.json()
    try:
        videos = int(body.get("videos"))
    except (TypeError, ValueError):
        return await _fail(request, 400, "api.no_package")
    stars = cfg.star_packages.get(videos)
    if stars is None:
        return await _fail(request, 400, "api.unknown_package")
    link = await bot.create_invoice_link(
        title=t("pay.invoice_title", lang, videos=videos, word=videos_word(videos, lang)),
        description=t("pay.invoice_description", lang),
        payload=f"videos:{videos}",
        currency="XTR",
        prices=[LabeledPrice(label=t("pay.invoice_label", lang, videos=videos), amount=stars)],
    )
    return web.json_response({"link": link})


async def set_language(request: web.Request) -> web.Response:
    """Смена языка из приложения. Тот же выбор, что кнопками в чате, — и кнопка меню тоже."""
    from src.bot.handlers.common import apply_menu_button

    body = await request.json()
    lang = str(body.get("lang") or "")
    if lang not in LANGS:
        return _json_error(400, "lang")
    user = await _db_user(request)
    await repo.set_user_lang(user.id, lang)
    await apply_menu_button(request.app[BOT_KEY], user.id, lang)
    return web.json_response({"lang": lang})


# ------------------------------------------------------------------ медиа и статика


def _browser_lang(request: web.Request) -> str:
    # У плеера и шрифта нет подписи Telegram, а значит и пользователя — только браузер.
    return "ru" if request.headers.get("Accept-Language", "").lower().startswith("ru") else "en"


async def media(request: web.Request) -> web.StreamResponse:
    """Готовый ролик для плеера в приложении. Файл кешируется на диске ради Range."""
    cfg = request.app[CFG_KEY]
    job_id = request.match_info["job_id"]
    if not check_media_token(job_id, request.query.get("t", ""), cfg.bot_token):
        return _json_error(403, t("api.link_expired", _browser_lang(request)))
    try:
        job = await repo.get_job(job_id)
    except ValueError:
        job = None
    if job is None or not job.result_key:
        return _json_error(404, t("api.video_not_found", _browser_lang(request)))
    cache = Path(cfg.work_dir) / "media-cache" / f"{job.id}.mp4"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        part = cache.with_suffix(".part")
        await storage.download(job.result_key, part)
        part.replace(cache)
    return web.FileResponse(cache, headers={"Cache-Control": "private, max-age=3600"})


async def font(request: web.Request) -> web.StreamResponse:
    """Шрифт темы для превью. Отдаём только файлы, которые назвал какой-то пресет."""
    cfg = request.app[CFG_KEY]
    name = request.match_info["name"]
    allowed = set()
    for preset in available():
        try:
            allowed.add(load_preset(preset).font.file)
        except Exception:  # noqa: BLE001
            continue
    path = Path(cfg.fonts_dir) / name
    if name not in allowed or not path.is_file():
        return _json_error(404, t("api.font_not_found", _browser_lang(request)))
    return web.FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


async def index(request: web.Request) -> web.StreamResponse:
    # Без кеша: адрес туннеля постоянный в рамках запуска, а приложение правится
    # чаще, чем Telegram сбрасывает свой кеш WebView.
    return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@web.middleware
async def no_cache_static(request: web.Request, handler):
    resp = await handler(request)
    if request.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


def create_app(bot: Bot, cfg: Settings) -> web.Application:
    # Кусок загрузки плюс запас на заголовки; целиком файл сюда не приходит никогда.
    app = web.Application(
        middlewares=[auth_middleware, no_cache_static],
        client_max_size=(cfg.webapp_chunk_mb + 1) * 1024 * 1024,
    )
    app[BOT_KEY] = bot
    app[CFG_KEY] = cfg
    app.router.add_get("/", index)
    app.router.add_static("/static/", STATIC)
    app.router.add_get("/media/{job_id}.mp4", media)
    app.router.add_get("/fonts/{name}", font)
    app.router.add_get("/api/bootstrap", bootstrap)
    app.router.add_post("/api/uploads", upload_start)
    app.router.add_put("/api/uploads/{upload_id}", upload_chunk)
    app.router.add_post("/api/uploads/{upload_id}/finish", upload_finish)
    app.router.add_get("/api/jobs", jobs_list)
    app.router.add_post("/api/jobs/{job_id}/send", job_send)
    app.router.add_post("/api/invoice", invoice)
    app.router.add_post("/api/language", set_language)
    return app


async def start_server(bot: Bot, cfg: Settings) -> web.AppRunner:
    runner = web.AppRunner(create_app(bot, cfg), access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, cfg.webapp_host, cfg.webapp_port)
    await site.start()
    log.info("Mini App слушает http://%s:%d", cfg.webapp_host, cfg.webapp_port)
    return runner
