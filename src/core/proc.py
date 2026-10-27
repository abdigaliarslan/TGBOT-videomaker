"""Запуск внешних процессов (ffmpeg, ffprobe, node). Одна точка, один способ."""

from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class ProcError(RuntimeError):
    def __init__(self, cmd: list[str], code: int, stderr: str) -> None:
        self.cmd = cmd
        self.code = code
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-25:])
        super().__init__(f"{Path(cmd[0]).name} завершился с кодом {code}\n{tail}")


def run(cmd: list[str], *, stdin: str | None = None, timeout: float | None = None) -> str:
    """Синхронный запуск. Возвращает stdout, кидает ProcError на ненулевой код."""
    log.debug("run: %s", shlex.join(cmd))
    p = subprocess.run(
        cmd,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode != 0:
        raise ProcError(cmd, p.returncode, p.stderr or "")
    return p.stdout


async def arun(cmd: list[str], *, stdin: str | None = None, timeout: float | None = None) -> str:
    """Асинхронный запуск — воркер не должен блокировать loop на ffmpeg."""
    log.debug("arun: %s", shlex.join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(stdin.encode() if stdin is not None else None), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode != 0:
        raise ProcError(cmd, proc.returncode or -1, err.decode("utf-8", "replace"))
    return out.decode("utf-8", "replace")


def capture(cmd: list[str], *, timeout: float | None = None) -> tuple[int, str, str]:
    """Как run(), но отдаёт (код, stdout, stderr) и не кидает.

    Нужно для фильтров ffmpeg, которые печатают результат замера в stderr
    (loudnorm print_format=json, silencedetect).
    """
    log.debug("capture: %s", shlex.join(cmd))
    p = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace"
    )
    return p.returncode, p.stdout or "", p.stderr or ""


def _ffmpeg_cmd(args: list[str]) -> list[str]:
    """Общая часть вызова ffmpeg, включая потолок по потокам.

    Без `-threads` ffmpeg берёт все ядра. На тонком ноутбуке (15-ваттный
    Ryzen U) это разом поднимает потребление до предела, и машина с изношенной
    батареей просто выключается — проверено дважды, оба раза журнал обрывается
    на полуслове, без штатного завершения. Ограничение стоит нескольких
    процентов скорости и снимает пик.
    """
    from src.core.config import settings

    base = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    if settings.ffmpeg_threads > 0:
        base += ["-threads", str(settings.ffmpeg_threads)]
    return [*base, *args]


def ffmpeg(args: list[str], *, timeout: float | None = None) -> str:
    return run(_ffmpeg_cmd(args), timeout=timeout)


async def affmpeg(args: list[str], *, timeout: float | None = None) -> str:
    return await arun(_ffmpeg_cmd(args), timeout=timeout)
