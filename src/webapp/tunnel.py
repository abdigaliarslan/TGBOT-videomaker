"""Публичный HTTPS-адрес для локального сервера через Cloudflare quick tunnel.

Telegram открывает Mini App только по https. На машине без домена и сертификата
адрес даёт `cloudflared tunnel --url …`: без аккаунта, без настройки, но на
КАЖДОМ запуске новый (`*.trycloudflare.com`). Поэтому бот после старта туннеля
перевешивает кнопку меню на свежий адрес — см. `bot/main.py`.

Ограничения бесплатного туннеля, под которые подстроен сервер: тело запроса не
больше 100 МБ (загрузка идёт кусками) и никакой гарантии доступности. Для
постоянной работы задайте WEBAPP_URL своего домена — туннель тогда не нужен.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import stat
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

DOWNLOAD_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


class TunnelError(RuntimeError):
    pass


def ensure_binary(target: Path) -> Path:
    """cloudflared из PATH, а если его нет — скачанный в `target`."""
    found = shutil.which("cloudflared")
    if found:
        return Path(found)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("скачиваю cloudflared в %s", target)
    part = target.with_suffix(".part")
    with urllib.request.urlopen(DOWNLOAD_URL, timeout=120) as resp, part.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    part.chmod(part.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    part.replace(target)
    return target


def restart_self() -> None:
    """Штатно завершить бота: он закроет polling, а systemd поднимет его заново."""
    os.kill(os.getpid(), signal.SIGTERM)


class Tunnel:
    def __init__(self, binary: Path, local_url: str) -> None:
        self.binary = binary
        self.local_url = local_url
        self.proc: asyncio.subprocess.Process | None = None
        self.url: str | None = None
        self._pump: asyncio.Task | None = None
        self._stopping = False

    async def start(self, timeout_s: float = 60.0) -> str:
        self.proc = await asyncio.create_subprocess_exec(
            str(self.binary), "tunnel", "--no-autoupdate", "--url", self.local_url,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )
        found: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._pump = asyncio.create_task(self._read(found))
        try:
            self.url = await asyncio.wait_for(asyncio.shield(found), timeout_s)
        except asyncio.TimeoutError as e:
            await self.stop()
            raise TunnelError("cloudflared не выдал адрес за отведённое время") from e
        log.info("туннель поднят: %s → %s", self.url, self.local_url)
        return self.url

    async def _read(self, found: asyncio.Future[str]) -> None:
        # Поток читается до конца жизни процесса, а не до первого адреса: иначе
        # заполненный буфер stderr остановит cloudflared, и туннель встанет.
        assert self.proc is not None and self.proc.stderr is not None
        async for raw in self.proc.stderr:
            line = raw.decode(errors="replace").rstrip()
            if not found.done():
                m = _URL.search(line)
                if m:
                    found.set_result(m.group(0))
            if " ERR " in line:
                log.warning("cloudflared: %s", line)
        code = await self.proc.wait()
        if not found.done():
            found.set_exception(TunnelError(f"cloudflared завершился с кодом {code}"))
        elif not self._stopping:
            # Адрес туннеля живёт ровно столько, сколько процесс cloudflared, и
            # поднять новый внутри работающего бота мало: у всех кнопок останется
            # мёртвый адрес. Поэтому завершаемся целиком — systemd поднимет бота
            # заново, а старт сам возьмёт свежий адрес и перевесит кнопки меню.
            log.error("туннель упал (код %s) — перезапускаю бота ради нового адреса", code)
            restart_self()

    async def stop(self) -> None:
        self._stopping = True
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 10)
            except asyncio.TimeoutError:
                self.proc.kill()
        if self._pump:
            self._pump.cancel()
