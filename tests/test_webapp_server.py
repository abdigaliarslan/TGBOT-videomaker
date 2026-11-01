"""HTTP-сторона Mini App: авторизация, протокол загрузки кусками, ссылки на ролик.

Сюда вынесено то, что проверяется без базы и хранилища. Постановка задачи в
очередь (списание, S3, Redis) проверена прогоном на живом стенде.
"""

from __future__ import annotations

import json
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.core.config import Settings
from src.webapp.auth import sign_init_data
from src.webapp.server import check_media_token, create_app, media_token

TOKEN = "123456:TEST-token"


def init_data(user_id: int = 42) -> str:
    return sign_init_data(
        {"auth_date": str(int(time.time())), "user": json.dumps({"id": user_id, "first_name": "Т"})},
        TOKEN,
    )


def auth(user_id: int = 42) -> dict[str, str]:
    return {"Authorization": f"tma {init_data(user_id)}"}


@pytest.fixture
async def client(tmp_path, monkeypatch):
    from src.db import repo

    async def no_user(_user_id):  # язык ответа — по клиенту; база этим ручкам не нужна
        return None

    monkeypatch.setattr(repo, "get_user", no_user)
    cfg = Settings(_env_file=None, bot_token=TOKEN, work_dir=tmp_path, webapp_chunk_mb=1,
                   webapp_max_upload_mb=5)
    app = create_app(bot=None, cfg=cfg)  # type: ignore[arg-type] — бот этим ручкам не нужен
    async with TestClient(TestServer(app)) as c:
        yield c


async def test_api_without_signature_is_refused(client):
    r = await client.post("/api/uploads", json={"size": 10})
    assert r.status == 401


async def test_api_with_a_foreign_bot_signature_is_refused(client):
    forged = sign_init_data({"auth_date": str(int(time.time())), "user": '{"id": 42}'}, "1:other")
    r = await client.post("/api/uploads", json={"size": 10}, headers={"Authorization": f"tma {forged}"})
    assert r.status == 401


async def test_the_index_page_is_public(client):
    """Страницу отдаём без подписи: подпись появляется только внутри Telegram."""
    r = await client.get("/")
    assert r.status == 200
    assert "telegram-web-app.js" in await r.text()


async def test_a_file_over_the_limit_is_refused_before_any_byte(client):
    r = await client.post("/api/uploads", json={"size": 6 * 1024 * 1024}, headers=auth())
    assert r.status == 413


async def test_chunks_resume_from_where_the_server_stopped(client):
    body = bytes(range(256)) * 12  # 3072 байта
    r = await client.post("/api/uploads", json={"size": len(body), "name": "a.mp4"}, headers=auth())
    upload_id = (await r.json())["upload_id"]
    url = f"/api/uploads/{upload_id}"

    r = await client.put(f"{url}?offset=0", data=body[:1000], headers=auth())
    assert (await r.json())["received"] == 1000

    # Ответ потерялся, клиент шлёт тот же кусок снова — второй раз не дописываем.
    r = await client.put(f"{url}?offset=0", data=body[:1000], headers=auth())
    assert r.status == 200 and (await r.json())["received"] == 1000

    # Клиент «забежал вперёд» — сервер говорит, откуда продолжать, и не портит файл.
    r = await client.put(f"{url}?offset=2000", data=body[2000:], headers=auth())
    assert r.status == 409 and (await r.json())["received"] == 1000

    r = await client.put(f"{url}?offset=1000", data=body[1000:], headers=auth())
    assert (await r.json())["received"] == len(body)

    from src.webapp.server import CFG_KEY, _upload_dir

    data = (_upload_dir(client.server.app[CFG_KEY], upload_id) / "data").read_bytes()
    assert data == body


async def test_a_chunk_past_the_declared_size_is_refused(client):
    r = await client.post("/api/uploads", json={"size": 100}, headers=auth())
    upload_id = (await r.json())["upload_id"]
    r = await client.put(f"/api/uploads/{upload_id}?offset=0", data=b"x" * 101, headers=auth())
    assert r.status == 400


async def test_someone_elses_upload_is_invisible(client):
    r = await client.post("/api/uploads", json={"size": 100}, headers=auth(42))
    upload_id = (await r.json())["upload_id"]
    r = await client.put(f"/api/uploads/{upload_id}?offset=0", data=b"x" * 10, headers=auth(43))
    assert r.status == 404


async def test_a_user_cannot_keep_opening_uploads(client):
    """Приём файла занимает диск до самого конца загрузки, а брошенную папку
    убирает только почасовая уборка. При разрешённом гигабайте на файл десяток
    брошенных приёмов съедает весь свободный диск сервера — вместе с местом под
    Postgres, который на той же машине. Поэтому приёмов на пользователя
    считанное число."""
    from src.webapp.server import MAX_OPEN_UPLOADS

    ids = []
    for _ in range(MAX_OPEN_UPLOADS):
        r = await client.post("/api/uploads", json={"size": 1024}, headers=auth())
        assert r.status == 200
        ids.append((await r.json())["upload_id"])

    r = await client.post("/api/uploads", json={"size": 1024}, headers=auth())
    assert r.status == 429
    # Чужие приёмы в счёт не идут: лимит на пользователя, а не на сервер.
    assert (await client.post("/api/uploads", json={"size": 1024}, headers=auth(43))).status == 200


async def test_an_upload_that_does_not_fit_the_disk_is_refused_up_front(client, monkeypatch):
    """Отказ до первого байта, а не на середине загрузки: пользователь с
    гигабайтным файлом иначе узнаёт о нехватке места через десять минут."""
    import shutil

    from src.webapp import server as mod

    monkeypatch.setattr(
        mod.shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(0, 0, 3 * 1024 * 1024)
    )
    r = await client.post("/api/uploads", json={"size": 2 * 1024 * 1024}, headers=auth())
    assert r.status == 507, "приём на диск, где места нет, разрешили"

    monkeypatch.setattr(
        mod.shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(0, 0, 50 * 1024 * 1024 * 1024)
    )
    assert (await client.post("/api/uploads", json={"size": 2 * 1024 * 1024}, headers=auth())).status == 200


async def test_an_abandoned_upload_is_swept_sooner_than_a_work_folder(tmp_path, monkeypatch):
    """Брошенная папка приёма — не папка задачи: за ней никто не придёт, а
    держит она до гигабайта. Сутки для неё слишком долго."""
    import os
    import time as _time

    from src.worker import tasks

    monkeypatch.setattr(tasks.settings, "work_dir", tmp_path)
    job_dir = tmp_path / "0000-job"
    upload_dir = tmp_path / "upload-deadbeef"
    for d in (job_dir, upload_dir):
        d.mkdir()
        (d / "data").write_bytes(b"x")
    # Обеим по три часа: папка задачи обязана дожить до суток, приём — нет.
    old = _time.time() - 3 * 3600
    for d in (job_dir, upload_dir):
        os.utime(d, (old, old))

    assert await tasks.cleanup_old_files({}) == 1
    assert job_dir.exists(), "уборка снесла папку живой задачи"
    assert not upload_dir.exists(), "брошенный приём остался занимать диск"


@pytest.mark.parametrize("upload_id", ["../../etc", "zz", "0" * 31])
async def test_upload_ids_cannot_escape_the_work_dir(client, upload_id: str):
    r = await client.put(f"/api/uploads/{upload_id}?offset=0", data=b"x", headers=auth())
    assert r.status == 404


def test_media_links_expire_and_bind_to_the_job():
    now = 1_800_000_000
    t = media_token("job-1", TOKEN, ttl_s=60, now=now)
    assert check_media_token("job-1", t, TOKEN, now=now + 30)
    assert not check_media_token("job-1", t, TOKEN, now=now + 61), "ссылка обязана протухать"
    assert not check_media_token("job-2", t, TOKEN, now=now), "чужой ролик по той же ссылке"
    assert not check_media_token("job-1", t, "1:other", now=now)
    assert not check_media_token("job-1", "garbage", TOKEN, now=now)


async def test_errors_speak_the_users_language(client, monkeypatch):
    """Ответ API — на языке пользователя, а не русский всем подряд."""
    from src.db import repo

    class EnglishUser:
        lang = "en"
        language_code = "ru"

    async def english(_user_id):
        return EnglishUser()

    monkeypatch.setattr(repo, "get_user", english)
    r = await client.post("/api/uploads", json={"size": 0}, headers=auth())
    assert r.status == 400
    assert (await r.json())["error"] == "The file is empty."


async def test_a_dead_tunnel_restarts_the_bot_and_a_stopped_one_does_not(monkeypatch, tmp_path):
    """Туннель умер сам — процесс бота завершается, чтобы systemd поднял его с новым адресом.

    Остановленный нами туннель (штатное завершение бота) так делать не должен.
    """
    import os
    import signal

    from src.webapp import tunnel as mod

    fake = tmp_path / "cloudflared"
    fake.write_text("#!/bin/sh\necho 'INF https://abc-def.trycloudflare.com' >&2\nsleep \"$SLEEP\"\n")
    fake.chmod(0o755)
    kills = []
    # Подменяется только перезапуск бота: `os.kill` трогать нельзя — через него
    # asyncio останавливает сам дочерний процесс.
    monkeypatch.setattr(mod, "restart_self", lambda: kills.append((os.getpid(), signal.SIGTERM)))

    monkeypatch.setenv("SLEEP", "0.3")
    t = mod.Tunnel(fake, "http://127.0.0.1:1")
    assert await t.start(timeout_s=5) == "https://abc-def.trycloudflare.com"
    await t._pump
    assert kills == [(os.getpid(), signal.SIGTERM)]

    kills.clear()
    monkeypatch.setenv("SLEEP", "5")
    t = mod.Tunnel(fake, "http://127.0.0.1:1")
    await t.start(timeout_s=5)
    await t.stop()
    assert kills == []
