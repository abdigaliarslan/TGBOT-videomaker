"""Приём видео, постановка в очередь, прогресс, отдача результата."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.types import CallbackQuery, FSInputFile, Message

from src.bot.keyboards import result_keyboard
from src.core.config import settings
from src.db import repo
from src.storage.s3 import SOURCES, storage
from src.worker.tasks import QUEUE_CPU, get_queue, progress_key

log = logging.getLogger(__name__)
router = Router(name="video")

POLL_SECONDS = 3.0  # бот редактирует сообщение раз в 3 секунды (§5 M4)
BAR_WIDTH = 12


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
    mime = getattr(media, "mime_type", None)
    if message.document is not None and mime and not mime.startswith("video/"):
        await message.answer("Это не видео. Пришлите видеофайл.")
        return

    db_user = await repo.upsert_user(
        user.id,
        username=user.username,
        first_name=user.first_name,
        language_code=user.language_code,
        is_admin=user.id in settings.admin_ids,
    )

    in_flight = await repo.active_jobs_count(user.id)
    if in_flight >= 1 and not db_user.is_admin:
        await message.answer("Один ролик уже в работе. Дождитесь его — и присылайте следующий.")
        return
    if await repo.active_jobs_count() >= settings.queue_max_size:
        await message.answer("Очередь забита. Попробуйте через несколько минут.")
        return

    duration = getattr(media, "duration", None)
    if duration and duration > settings.max_video_seconds:
        await message.answer(
            f"Видео длиннее {settings.max_video_minutes:.0f} мин. Пришлите кусок покороче."
        )
        return

    status = await message.answer("Принял. Ставлю в очередь…")

    job = await repo.create_job(
        user.id,
        source_file_id=media.file_id,
        source_key=None,
        source_size=getattr(media, "file_size", None),
        preset=db_user.preset,
        chat_id=message.chat.id,
    )
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
        await status.edit_text("Не смог скачать файл из Telegram. Попробуйте прислать ещё раз.")
        return

    queue = await get_queue(QUEUE_CPU)
    try:
        await queue.enqueue_job("process_job", str(job.id), _queue_name=QUEUE_CPU)
    finally:
        await queue.close()

    asyncio.create_task(_watch(bot, message.chat.id, status.message_id, str(job.id), db_user.preset))


async def resume_watchers(bot: Bot) -> int:
    """Перецепиться к задачам, которые шли во время рестарта бота.

    Наблюдатели живут в памяти процесса, поэтому без этого пользователь после
    рестарта не дождался бы ролика никогда — джоб бы доделался в тишину.
    """
    jobs = await repo.jobs_in_flight()
    for job in jobs:
        asyncio.create_task(
            _watch(bot, job.chat_id, job.status_message_id, str(job.id), job.preset)
        )
    if jobs:
        log.info("перецепился к %d задачам после рестарта", len(jobs))
    return len(jobs)


async def _watch(bot: Bot, chat_id: int, message_id: int, job_id: str, preset: str) -> None:
    """Раз в 3 секунды подтягивает прогресс из Redis и правит одно сообщение."""
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
                await _safe_edit(bot, chat_id, message_id, data.get("error") or "Не получилось.")
                return
            if stage == "done":
                await _deliver(bot, chat_id, message_id, job_id, preset)
                return

            text = f"{bar(share)}  {int(share * 100)}%\n{stage}…"
            if text != last_text:
                await _safe_edit(bot, chat_id, message_id, text)
                last_text = text
    finally:
        await queue.close()


async def _safe_edit(bot: Bot, chat_id: int, message_id: int, text: str) -> None:
    with contextlib.suppress(Exception):  # сообщение могли удалить — это не ошибка
        await bot.edit_message_text(text=text, chat_id=chat_id, message_id=message_id)


async def _deliver(bot: Bot, chat_id: int, message_id: int, job_id: str, preset: str) -> None:
    job = await repo.get_job(job_id)
    if job is None or not job.result_key:
        await _safe_edit(bot, chat_id, message_id, "Ролик готов, но файл потерялся. Напишите нам.")
        return

    dst = Path(settings.work_dir) / f"deliver-{job_id}.mp4"
    try:
        await storage.download(job.result_key, dst)
        await bot.send_video(
            chat_id=chat_id,
            video=FSInputFile(dst, filename="reel.mp4"),
            caption=_caption(job),
            supports_streaming=True,
            reply_markup=result_keyboard(job_id, preset),
        )
        await _safe_edit(bot, chat_id, message_id, "Готово.")
    except Exception as e:  # noqa: BLE001
        log.exception("не отдал результат")
        await _safe_edit(bot, chat_id, message_id, f"Ролик готов, но не отправился: {e}")
    finally:
        dst.unlink(missing_ok=True)


def _caption(job) -> str:
    parts = []
    if job.source_duration and job.result_duration:
        saved = job.source_duration - job.result_duration
        parts.append(f"{job.source_duration:.0f} с → {job.result_duration:.0f} с")
        if saved > 0.5:
            parts.append(f"вырезано {saved:.0f} с пауз")
    parts.append(f"стиль: {job.preset}")
    return " · ".join(parts)


@router.callback_query(lambda c: c.data and c.data.startswith("rerender:"))
async def on_rerender(callback: CallbackQuery, bot: Bot) -> None:
    if callback.data is None or callback.message is None:
        return
    _, job_id, preset = callback.data.split(":", 2)
    job = await repo.get_job(job_id)
    if job is None or not job.edl:
        await callback.answer("Исходный монтаж не сохранился", show_alert=True)
        return

    await callback.answer("Пересобираю — без повторного распознавания")
    status = await bot.send_message(callback.message.chat.id, f"Пересобираю в «{preset}»…")

    queue = await get_queue(QUEUE_CPU)
    try:
        await queue.enqueue_job("rerender_job", job_id, preset, _queue_name=QUEUE_CPU)
    finally:
        await queue.close()

    asyncio.create_task(
        _watch_rerender(bot, callback.message.chat.id, status.message_id, job_id, preset)
    )


async def _watch_rerender(bot: Bot, chat_id: int, message_id: int, job_id: str, preset: str) -> None:
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
                    caption=f"стиль: {preset}",
                    supports_streaming=True,
                    reply_markup=result_keyboard(job_id, preset),
                )
                await _safe_edit(bot, chat_id, message_id, "Готово.")
            finally:
                dst.unlink(missing_ok=True)
            return
    finally:
        await queue.close()
