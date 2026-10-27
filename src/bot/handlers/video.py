"""Приём видео, постановка в очередь, прогресс, отдача результата."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
from aiogram.types import CallbackQuery, FSInputFile, Message

from src.bot.keyboards import app_keyboard, result_keyboard
from src.core.config import settings
from src.core.i18n import error_text, lang_of, stage_title, t
from src.db import repo
from src.storage.s3 import SOURCES, storage
from src.worker.tasks import QUEUE_CPU, get_queue, progress_key

log = logging.getLogger(__name__)
router = Router(name="video")

POLL_SECONDS = 3.0  # бот редактирует сообщение раз в 3 секунды (§5 M4)
BAR_WIDTH = 12

# Сколько раз пытаться отдать готовый ролик и сколько ждать между попытками.
#
# Заливка десятков мегабайт рвётся: проверено на живом ролике — полностью
# смонтированное видео не дошло из-за одного «Connection reset by peer».
# Обрыв связи не должен стоить пользователю всего монтажа.
DELIVERY_ATTEMPTS = 4
DELIVERY_BACKOFF = 5.0


def bar(share: float) -> str:
    filled = max(0, min(BAR_WIDTH, round(share * BAR_WIDTH)))
    return "▰" * filled + "▱" * (BAR_WIDTH - filled)


async def _resolve_source(bot: Bot, file_id: str, dst: Path) -> tuple[Path | None, str | None]:
    """Достаёт файл. Возвращает (локальный путь, путь Bot API).

    ⚠ Локальный Bot API server отдаёт путь к файлу на диске, а не URL.
    `bot.download_file()` с ним не работает — читаем напрямую с общего тома (M0 ⚠).
    """
    tg_file = await bot.get_file(file_id)
    path = tg_file.file_path
    if not path:
        return None, None

    if settings.bot_api_url:
        local = settings.bot_api_path_to_local(path)
        if local.exists():
            return local, str(local)
        log.warning("локальный Bot API отдал %s, но файла там нет — качаю через API", path)

    dst.parent.mkdir(parents=True, exist_ok=True)
    await bot.download_file(path, destination=dst)
    return dst, None


@router.message(F.video | F.document | F.video_note)
async def on_video(message: Message, bot: Bot) -> None:
    user = message.from_user
    if user is None:
        return

    media = message.video or message.video_note or message.document
    if media is None:
        return
    db_user = await repo.upsert_user(
        user.id,
        username=user.username,
        first_name=user.first_name,
        language_code=user.language_code,
        is_admin=user.id in settings.admin_ids,
    )
    lang = lang_of(db_user)

    mime = getattr(media, "mime_type", None)
    if message.document is not None and mime and not mime.startswith("video/"):
        await message.answer(t("video.not_video", lang))
        return

    in_flight = await repo.active_jobs_count(user.id)
    if in_flight >= 1 and not db_user.is_admin:
        await message.answer(t("video.one_in_flight", lang))
        return
    if await repo.active_jobs_count() >= settings.queue_max_size:
        await message.answer(t("video.queue_full", lang))
        return

    duration = getattr(media, "duration", None)
    if duration and duration > settings.max_video_seconds:
        await message.answer(t("video.too_long", lang, max_m=f"{settings.max_video_minutes:.0f}"))
        return

    # Пейволл один на оба входа: иначе оплату в приложении обходили бы простой
    # отправкой файла в чат. Владельцы из ADMIN_IDS не платят.
    try:
        job = await repo.create_job(
            user.id,
            source_file_id=media.file_id,
            source_key=None,
            source_size=getattr(media, "file_size", None),
            preset=db_user.preset,
            chat_id=message.chat.id,
            charge=not db_user.is_admin,
        )
    except repo.InsufficientBalance:
        await message.answer(t("video.no_balance", lang), reply_markup=app_keyboard("balance", lang))
        return

    status = await message.answer(t("video.accepted", lang))
    await repo.update_job(job.id, status_message_id=status.message_id)

    try:
        tmp = Path(settings.work_dir) / str(job.id) / "source.mp4"
        local, bot_api_path = await _resolve_source(bot, media.file_id, tmp)
        if local is None:
            raise RuntimeError("Bot API не отдал путь к файлу")

        if bot_api_path:
            await repo.update_job(job.id, source_local_path=bot_api_path)
        key = storage.key_for(SOURCES, str(job.id), "source.mp4")
        await storage.upload(local, key, content_type="video/mp4")
        await repo.update_job(job.id, source_key=key)
    except Exception as e:  # noqa: BLE001 — юзеру нужен внятный ответ, а не трейс
        log.exception("не удалось принять файл")
        await repo.mark_failed(job.id, f"приём файла: {e}")
        await repo.refund_job(job.id)
        await status.edit_text(t("video.download_failed", lang))
        return

    queue = await get_queue(QUEUE_CPU)
    try:
        await queue.enqueue_job("process_job", str(job.id), _queue_name=QUEUE_CPU)
    finally:
        await queue.close()

    asyncio.create_task(_watch(bot, message.chat.id, status.message_id, str(job.id), db_user.preset))


def watch_job(bot: Bot, chat_id: int, message_id: int, job_id: str, preset: str) -> None:
    """Повесить наблюдателя прогресса на задачу, поставленную не из чата (Mini App)."""
    asyncio.create_task(_watch(bot, chat_id, message_id, job_id, preset))


async def send_result(bot: Bot, chat_id: int, job) -> None:
    """Прислать готовый ролик в чат ещё раз — по кнопке в приложении."""
    dst = Path(settings.work_dir) / f"resend-{job.id}.mp4"
    try:
        await storage.download(job.result_key, dst)
        await _send_video(bot, chat_id, job, dst, job.preset, await _lang_for_job(str(job.id)))
        await repo.mark_delivered(job.id)
    finally:
        dst.unlink(missing_ok=True)
        dst.with_suffix(".thumb.jpg").unlink(missing_ok=True)


async def resume_watchers(bot: Bot) -> int:
    """Перецепиться к задачам, которые шли во время рестарта бота.

    Наблюдатели живут в памяти процесса, поэтому без этого пользователь после
    рестарта не дождался бы ролика никогда — джоб бы доделался в тишину.
    """
    from src.db.models import JobStatus

    jobs = await repo.jobs_in_flight()
    for job in jobs:
        if job.status == JobStatus.DONE and job.result_key:
            # Доделалось, пока бот лежал — отдаём немедленно, ждать нечего.
            asyncio.create_task(
                _deliver(bot, job.chat_id, job.status_message_id, str(job.id), job.preset)
            )
        else:
            asyncio.create_task(
                _watch(bot, job.chat_id, job.status_message_id, str(job.id), job.preset)
            )
    if jobs:
        log.info("после рестарта подхвачено задач: %d", len(jobs))
    return len(jobs)


async def _lang_for_job(job_id: str) -> str:
    """Язык владельца задачи. Читается один раз на наблюдателя, а не на каждый шаг."""
    try:
        job = await repo.get_job(job_id)
        return lang_of(await repo.get_user(job.user_id) if job else None)
    except Exception:  # noqa: BLE001 — язык не повод оставить пользователя без ролика
        return lang_of(None)


async def _watch(bot: Bot, chat_id: int, message_id: int, job_id: str, preset: str) -> None:
    """Раз в 3 секунды подтягивает прогресс из Redis и правит одно сообщение."""
    lang = await _lang_for_job(job_id)
    queue = await get_queue()
    redis = queue
    last_text = ""
    try:
        for _ in range(int(3 * 3600 / POLL_SECONDS)):
            await asyncio.sleep(POLL_SECONDS)
            raw = await redis.get(progress_key(job_id))
            if raw is None:
                continue
            data = json.loads(raw if isinstance(raw, str) else raw.decode())
            stage = data.get("stage", "")
            share = float(data.get("progress") or 0.0)

            if stage == "failed":
                text = error_text(
                    data.get("error_code"), lang,
                    min_s=settings.min_video_seconds, max_m=settings.max_video_minutes,
                )
                await _safe_edit(bot, chat_id, message_id, text)
                return
            if stage == "done":
                await _deliver(bot, chat_id, message_id, job_id, preset)
                return

            title = stage_title(data.get("stage_key"), lang) or stage
            text = f"{bar(share)}  {int(share * 100)}%\n{title}…"
            if text != last_text:
                await _safe_edit(bot, chat_id, message_id, text)
                last_text = text
    finally:
        await queue.close()


async def _safe_edit(bot: Bot, chat_id: int, message_id: int, text: str) -> None:
    with contextlib.suppress(Exception):  # сообщение могли удалить — это не ошибка
        await bot.edit_message_text(text=text, chat_id=chat_id, message_id=message_id)


async def _send_video(
    bot: Bot, chat_id: int, job, dst: Path, preset: str, lang: str | None = None
) -> None:
    """Отправка ролика, переживающая обрыв связи.

    Файл уже скачан — повторяем только саму заливку. Ошибки логики (не тот чат,
    слишком большой файл) не ретраятся: они от повтора не пройдут.
    """
    dims = await video_dims(dst)
    thumb = await video_thumbnail(dst, dims)
    last: Exception | None = None
    for attempt in range(1, DELIVERY_ATTEMPTS + 1):
        try:
            await bot.send_video(
                chat_id=chat_id,
                video=FSInputFile(dst, filename="reel.mp4"),
                thumbnail=thumb,
                **dims,
                caption=_caption(job, lang),
                supports_streaming=True,
                reply_markup=result_keyboard(str(job.id), preset, lang),
            )
            dst.with_suffix(".thumb.jpg").unlink(missing_ok=True)
            return
        except TelegramRetryAfter as e:  # флуд-контроль сам сказал, сколько ждать
            last = e
            wait = float(e.retry_after)
        except (TelegramNetworkError, OSError) as e:
            last = e
            wait = DELIVERY_BACKOFF * attempt
        log.warning(
            "попытка %d из %d отдать ролик не удалась: %s", attempt, DELIVERY_ATTEMPTS, last
        )
        if attempt < DELIVERY_ATTEMPTS:
            await asyncio.sleep(wait)
    assert last is not None
    raise last


async def video_thumbnail(path: Path, dims: dict[str, int]) -> FSInputFile | None:
    """Обложка-кадр в пропорциях ролика, до 320 px по длинной стороне (лимит Telegram).

    iPhone строит рамку пузыря по обложке ещё до загрузки видео. Без своей
    обложки он берёт серверную, и вертикальный ролик выходил растянутым вширь.
    """
    if not dims:
        return None
    thumb = path.with_suffix(".thumb.jpg")
    scale = "320:-2" if dims["width"] >= dims["height"] else "-2:320"
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(path),
           "-frames:v", "1", "-vf", f"scale={scale}", "-q:v", "4", str(thumb)]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd)
        await asyncio.wait_for(proc.wait(), 30)
    except Exception:  # noqa: BLE001 — без обложки ролик всё равно уходит
        return None
    return FSInputFile(thumb) if thumb.exists() and thumb.stat().st_size > 0 else None


async def video_dims(path: Path) -> dict[str, int]:
    """Ширина, высота и длительность для `send_video`.

    Без них Telegram угадывает рамку сам, и на iPhone вертикальный 1080×1920
    показывался почти квадратом с растянутой по ширине картинкой — хотя сам
    файл был правильным. Не прочитали файл — отправляем как раньше, без них.
    """
    from src.pipeline.probe import probe

    try:
        info = await asyncio.to_thread(probe, path)
    except Exception:  # noqa: BLE001
        return {}
    if not info.width or not info.height:
        return {}
    return {"width": info.width, "height": info.height, "duration": max(1, round(info.duration))}


async def _deliver(bot: Bot, chat_id: int, message_id: int, job_id: str, preset: str) -> None:
    lang = await _lang_for_job(job_id)
    job = await repo.get_job(job_id)
    if job is None or not job.result_key:
        await _safe_edit(bot, chat_id, message_id, t("video.result_lost", lang))
        return

    dst = Path(settings.work_dir) / f"deliver-{job_id}.mp4"
    try:
        await storage.download(job.result_key, dst)
        await _send_video(bot, chat_id, job, dst, preset, lang)
        # Отмечаем только ПОСЛЕ успешной отправки: если заливка сорвалась,
        # задача должна остаться в очереди на доставку.
        await repo.mark_delivered(job_id)
        await _safe_edit(bot, chat_id, message_id, t("video.done", lang))
    except Exception as e:  # noqa: BLE001
        log.exception("не отдал результат")
        await _safe_edit(bot, chat_id, message_id, t("video.not_sent", lang, error=e))
    finally:
        dst.unlink(missing_ok=True)
        dst.with_suffix(".thumb.jpg").unlink(missing_ok=True)


def _caption(job, lang: str | None = None) -> str:
    parts = []
    if job.source_duration and job.result_duration:
        saved = job.source_duration - job.result_duration
        parts.append(
            t("video.caption_durations", lang,
              src=f"{job.source_duration:.0f}", res=f"{job.result_duration:.0f}")
        )
        if saved > 0.5:
            parts.append(t("video.caption_saved", lang, n=f"{saved:.0f}"))
    parts.append(t("video.caption_style", lang, preset=job.preset))
    return " · ".join(parts)


@router.callback_query(lambda c: c.data and c.data.startswith("rerender:"))
async def on_rerender(callback: CallbackQuery, bot: Bot) -> None:
    if callback.data is None or callback.message is None:
        return
    _, job_id, preset = callback.data.split(":", 2)
    lang = await _lang_for_job(job_id)
    job = await repo.get_job(job_id)
    if job is None or not job.edl:
        await callback.answer(t("rerender.no_edl", lang), show_alert=True)
        return
    if job.files_purged_at is not None or not (job.source_key or job.source_local_path):
        # Без этой проверки воркер падал на «исходник потерялся», а наблюдатель
        # перерендера ждёт только успеха — пользователь не получал ничего.
        await callback.answer(t("rerender.purged", lang), show_alert=True)
        return

    await callback.answer(t("rerender.toast", lang))
    status = await bot.send_message(callback.message.chat.id, t("rerender.status", lang, preset=preset))

    queue = await get_queue(QUEUE_CPU)
    try:
        await queue.enqueue_job("rerender_job", job_id, preset, _queue_name=QUEUE_CPU)
    finally:
        await queue.close()

    asyncio.create_task(
        _watch_rerender(bot, callback.message.chat.id, status.message_id, job_id, preset)
    )


async def _watch_rerender(bot: Bot, chat_id: int, message_id: int, job_id: str, preset: str) -> None:
    lang = await _lang_for_job(job_id)
    queue = await get_queue()
    try:
        for _ in range(int(3 * 3600 / POLL_SECONDS)):
            await asyncio.sleep(POLL_SECONDS)
            raw = await queue.get(progress_key(f"{job_id}:{preset}"))
            if raw is None:
                continue
            data = json.loads(raw if isinstance(raw, str) else raw.decode())
            if data.get("stage") != "done":
                continue
            key = data.get("result_key")
            dst = Path(settings.work_dir) / f"rerender-{job_id}-{preset}.mp4"
            try:
                await storage.download(key, dst)
                await bot.send_video(
                    chat_id=chat_id,
                    video=FSInputFile(dst, filename=f"reel-{preset}.mp4"),
                    thumbnail=await video_thumbnail(dst, dims := await video_dims(dst)),
                    **dims,
                    caption=t("video.caption_style", lang, preset=preset),
                    supports_streaming=True,
                    reply_markup=result_keyboard(job_id, preset, lang),
                )
                await _safe_edit(bot, chat_id, message_id, t("video.done", lang))
            finally:
                dst.unlink(missing_ok=True)
                dst.with_suffix(".thumb.jpg").unlink(missing_ok=True)
            return
    finally:
        await queue.close()
