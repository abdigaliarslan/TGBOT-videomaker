"""HTTP-сторона Mini App: авторизация, протокол загрузки кусками, ссылки на ролик.

Сюда вынесено то, что проверяется без базы и хранилища. Постановка задачи в
очередь (списание, S3, Redis) проверена прогоном на живом стенде.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

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


async def test_space_already_promised_to_other_uploads_is_not_promised_twice(client, monkeypatch):
    """Проверка «влезет ли файл» обязана считать и уже обещанное.

    Байты незавершённого приёма на диске ещё не лежат, поэтому свободного места
    по `disk_usage` сколько было, столько и есть. Лимит на пользователя тут не
    спасает: десять человек по два приёма — это двадцать обещанных гигабайт на
    диске, где их нет.
    """
    import shutil

    from src.webapp import server as mod

    free = 12 * 1024 * 1024
    monkeypatch.setattr(mod, "DISK_RESERVE_MB", 0)
    monkeypatch.setattr(
        mod.shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(0, 0, free)
    )
    # Два приёма по 5 МБ от разных людей — обещано 10 из 12 МБ.
    for user in (42, 43):
        r = await client.post("/api/uploads", json={"size": 5 * 1024 * 1024}, headers=auth(user))
        assert r.status == 200

    r = await client.post("/api/uploads", json={"size": 5 * 1024 * 1024}, headers=auth(44))
    assert r.status == 507, "пообещали место, которое уже обещано другим приёмам"
    # А то, что в остаток влезает, принимается.
    assert (await client.post("/api/uploads", json={"size": 1024}, headers=auth(44))).status == 200


async def test_a_reservation_nobody_feeds_stops_holding_space(client, monkeypatch):
    """Заявка на место — не бронь навечно.

    Объявить гигабайт и не прислать ни байта стоит ничего, а место держит до
    уборщика. Пяти аккаунтов хватило бы, чтобы на два часа закрыть загрузку
    всем. Поэтому заявка живёт коротким сроком, который продлевает каждый
    пришедший кусок: кто молчит — место отпускает, и своей же квоты не
    занимает.
    """
    import os
    import shutil
    import time as _time

    from src.webapp import server as mod

    free = 5 * 1024 * 1024
    monkeypatch.setattr(mod, "DISK_RESERVE_MB", 0)
    monkeypatch.setattr(
        mod.shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(0, 0, free)
    )
    cfg = client.server.app[mod.CFG_KEY]

    # Два приёма по 2 МБ — обещано 4 из 5, третий такой же не влезает.
    ids = []
    for user in (42, 43):
        r = await client.post("/api/uploads", json={"size": 2 * 1024 * 1024}, headers=auth(user))
        ids.append((await r.json())["upload_id"])
    assert (await client.post("/api/uploads", json={"size": 2 * 1024 * 1024}, headers=auth(44))).status == 507

    # Старим оба: кусков по ним не было дольше срока заявки.
    old = _time.time() - mod.UPLOAD_LEASE - 60
    for upload_id in ids:
        data = mod._upload_dir(cfg, upload_id) / "data"
        os.utime(data, (old, old))
        os.utime(data.parent, (old, old))

    r = await client.post("/api/uploads", json={"size": 2 * 1024 * 1024}, headers=auth(44))
    assert r.status == 200, "молчащая заявка продолжает держать место"
    # И своя же молчащая заявка не запирает квоту пользователя.
    assert (await client.post("/api/uploads", json={"size": 1024}, headers=auth(42))).status == 200


async def test_a_live_upload_keeps_its_reservation(client, monkeypatch):
    """Кусок пришёл — срок заявки пошёл заново, место остаётся за ней."""
    import shutil

    from src.webapp import server as mod

    monkeypatch.setattr(mod, "DISK_RESERVE_MB", 0)
    monkeypatch.setattr(
        mod.shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(0, 0, 5 * 1024 * 1024)
    )
    r = await client.post("/api/uploads", json={"size": 4 * 1024 * 1024}, headers=auth(42))
    upload_id = (await r.json())["upload_id"]
    await client.put(f"/api/uploads/{upload_id}?offset=0", data=b"x" * 1024, headers=auth(42))

    r = await client.post("/api/uploads", json={"size": 2 * 1024 * 1024}, headers=auth(43))
    assert r.status == 507, "живой приём лишился своего места"


async def test_only_one_finisher_can_claim_an_upload(client):
    """Сборку приёма забирает ровно один — переименованием папки.

    Два «Готово» разом (двойной тап, повтор по тайм-ауту) иначе создали бы две
    задачи за один файл, то есть списали бы два ролика. И папка уходит из-под
    уборки на время сборки: дальше ffprobe и заливка, кусков больше не будет, а
    чужой `upload_start` имеет право снести просроченную заявку.
    """
    import asyncio as _asyncio

    from src.webapp import server as mod

    cfg = client.server.app[mod.CFG_KEY]
    r = await client.post("/api/uploads", json={"size": 4}, headers=auth())
    folder = mod._upload_dir(cfg, (await r.json())["upload_id"])

    claims = await _asyncio.gather(
        _asyncio.to_thread(mod._claim_upload, folder),
        _asyncio.to_thread(mod._claim_upload, folder),
    )
    won = [c for c in claims if c is not None]
    assert len(won) == 1, f"приём забрали дважды: {claims}"
    assert won[0].exists() and not folder.exists()


async def test_a_claimed_upload_is_out_of_the_sweepers_reach(client, monkeypatch):
    """Забранную папку уборка приёмов не видит, даже когда срок заявки истёк."""
    import os
    import time as _time

    from src.webapp import server as mod

    cfg = client.server.app[mod.CFG_KEY]
    r = await client.post("/api/uploads", json={"size": 4}, headers=auth())
    claimed = mod._claim_upload(mod._upload_dir(cfg, (await r.json())["upload_id"]))
    assert claimed is not None
    old = _time.time() - mod.UPLOAD_LEASE - 60
    os.utime(claimed / "data", (old, old))
    os.utime(claimed, (old, old))

    assert mod._sweep_stale_uploads(cfg) == []
    assert claimed.exists(), "уборка снесла приём, который прямо сейчас собирают"


async def test_a_refused_finish_leaves_no_folder_behind(client):
    """Отказ на сборке обязан убирать за собой файл.

    Забранная папка не попадает ни в квоту пользователя, ни под уборку заявок —
    в этом и смысл, пока сборка идёт. Но если отказ случился ПОСЛЕ того, как её
    забрали (не та тема, нет баланса, очередь полна), папка оставалась лежать до
    почасовой уборки и места не занимала ни в одном счёте. Цикл «залил —
    получил отказ» тогда кладёт на диск сколько угодно гигабайт.
    """
    from src.webapp import server as mod

    cfg = client.server.app[mod.CFG_KEY]
    r = await client.post("/api/uploads", json={"size": 4}, headers=auth())
    upload_id = (await r.json())["upload_id"]
    await client.put(f"/api/uploads/{upload_id}?offset=0", data=b"abcd", headers=auth())

    r = await client.post(
        f"/api/uploads/{upload_id}/finish",
        json={"preset": "такой-темы-нет", "options": {}},
        headers=auth(),
    )
    assert r.status == 400

    left = list(Path(cfg.work_dir).glob("*"))
    assert left == [], f"после отказа на диске осталось: {[p.name for p in left]}"


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
    # Папку, забранную под сборку, тоже подбираем по короткому сроку: сюда она
    # попадает только если процесс умер на середине, и ждать сутки нечего.
    claimed_dir = tmp_path / "taking-deadbeef"
    claimed_dir.mkdir()
    (claimed_dir / "data").write_bytes(b"x")
    # Всем по три часа: папка задачи обязана дожить до суток, приёмы — нет.
    old = _time.time() - 3 * 3600
    for d in (job_dir, upload_dir, claimed_dir):
        os.utime(d, (old, old))

    assert await tasks.cleanup_old_files({}) == 2
    assert job_dir.exists(), "уборка снесла папку живой задачи"
    assert not upload_dir.exists(), "брошенный приём остался занимать диск"
    assert not claimed_dir.exists(), "брошенная сборка осталась занимать диск"


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


# ------------------------------------------------------------------ шрифты


@pytest.fixture
async def booted(tmp_path, monkeypatch):
    """Клиент с ботом-двойником: `/api/bootstrap` спрашивает у бота его имя.

    База тоже двойник, и это не экономия: `upsert_user` держит движок
    SQLAlchemy глобально, привязанным к тому циклу событий, в котором его
    создали. Второй тест с этой фикстурой получал «Future attached to a
    different loop» и 500 — то есть проверки бутстрапа работали, пока он был
    ровно один. Постгрес нужен денежным тестам, а не этим.
    """
    from types import SimpleNamespace

    from src.db import repo
    from src.webapp.server import create_app

    async def no_user(_user_id):
        return None

    async def upsert(user_id, **fields):
        return SimpleNamespace(
            id=user_id, balance=3, videos_done=1, preset="pulse", settings={}, lang=None, ref_credit=0,
            username=fields.get("username"), first_name=fields.get("first_name"),
            language_code=fields.get("language_code"), is_admin=fields.get("is_admin", False),
        )

    async def me():
        return SimpleNamespace(username="bot", first_name="Бот")

    monkeypatch.setattr(repo, "get_user", no_user)
    monkeypatch.setattr(repo, "upsert_user", upsert)
    cfg = Settings(_env_file=None, bot_token=TOKEN, work_dir=tmp_path, webapp_chunk_mb=1,
                   webapp_max_upload_mb=5)
    app = create_app(bot=SimpleNamespace(me=me), cfg=cfg)  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as c:
        yield c


async def test_bootstrap_offers_the_font_library(booted):
    """Приложение рисует чипы по тому, что пришло; без этого массива выбора нет."""
    from src.core.options import FONTS

    r = await booted.get("/api/bootstrap", headers=auth())
    assert r.status == 200
    fonts = (await r.json())["fonts"]
    assert [f["key"] for f in fonts] == list(FONTS)
    theme, *rest = fonts
    assert theme["key"] == "theme" and theme["file"] is None, "«как в теме» идёт первым"
    assert all(f["title"] and f["file"] and f["family"] for f in rest)


async def test_the_font_route_serves_the_whole_library(client):
    """Раньше список собирался только из пресетов. Шрифт, который не назвала ни
    одна тема, браузер бы не получил — и чип выбора показывал бы не тот шрифт."""
    from src.core.options import FONTS

    for key, (_, choice) in FONTS.items():
        if choice is None:
            continue
        r = await client.get(f"/fonts/{choice.file}")
        assert r.status == 200, f"{key}: шрифт не отдаётся приложению"
        assert (await r.read())[:4] in (b"\x00\x01\x00\x00", b"true", b"OTTO")


async def test_the_font_route_still_refuses_anything_else(client):
    for name in ("../../etc/passwd", "secret.env", "Unknown-Bold.ttf"):
        assert (await client.get(f"/fonts/{name}")).status == 404


async def test_fonts_travel_compressed(client):
    """Библиотека выбора — восемнадцать начертаний, и все рисуются на первом
    экране: плитка темы её шрифтом, чип образцом «Аа». В TTF это 3.1 МБ на
    открытие, в gzip — вдвое меньше. Распакованное обязано совпасть с файлом
    байт в байт: раскладку Python считает по ЭТОМУ файлу."""
    import gzip

    from src.core.config import REPO_ROOT

    name = "Forum-Regular.ttf"
    r = await client.get(f"/fonts/{name}", headers={"Accept-Encoding": "gzip"})
    assert r.status == 200
    # aiohttp распаковывает сам, поэтому сжатие видно по заголовку, а не по телу.
    assert r.headers.get("Content-Encoding") == "gzip"
    body = await r.read()
    raw = (REPO_ROOT / "assets" / "fonts" / name).read_bytes()
    assert body == raw
    assert len(gzip.compress(raw, 9)) < len(raw) * 0.75, "сжатие не окупается — проверить формат"

    plain = await client.get(f"/fonts/{name}", headers={"Accept-Encoding": "identity"})
    assert plain.status == 200 and await plain.read() == raw


# ------------------------------------------------------------------ темы


async def test_bootstrap_describes_every_theme_the_way_the_preview_draws_it(booted):
    """Превью рисует тему ЭТИМИ числами, а не своими.

    Пока их не было, картинка в приложении показывала одну композицию на все
    темы: один кегль, одно место в кадре, одно выделение. Тем стало
    семнадцать, и расхождение превью с роликом перестало быть мелочью —
    человек выбирает по картинке. Числа обязаны быть числами ТЕМЫ, иначе
    приложение обещает не то, что соберёт воркер.
    """
    from src.core.geometry import load_metrics
    from src.core.preset import available, load_preset

    data = await (await booted.get("/api/bootstrap", headers=auth())).json()
    assert {p["name"] for p in data["presets"]} == set(available())
    for item in data["presets"]:
        p = load_preset(item["name"])
        assert item["size"] == p.font.size, item["name"]
        assert item["uppercase"] == p.font.uppercase, item["name"]
        assert item["letter_spacing"] == p.font.letter_spacing, item["name"]
        assert item["line_height"] == p.caption.line_height, item["name"]
        assert item["position"] == p.caption.position, item["name"]
        assert item["margin"] == p.caption.margin, item["name"]
        assert item["accent_scale"] == p.animation.accent_scale, item["name"]
        assert item["panel"]["enabled"] == p.panel.enabled, item["name"]
        assert item["panel"]["opacity"] == p.panel.opacity, item["name"]
        # Высота прописных — из выгруженных метрик, а не из числа, переписанного
        # руками: по ней приложение пересчитывает кегль при смене шрифта ровно
        # так же, как это делает `options.apply_font` в пайплайне.
        m = load_metrics(p.font.file)
        assert item["cap_em"] == pytest.approx(m.cap_height / m.units_per_em, abs=1e-4)


async def test_themes_come_in_one_order_for_the_app_and_the_chat(booted):
    """Порядок тем — свойство данных (`order`), и он один на оба входа.

    В чате и в приложении список один и тот же; разойдись они, «третья слева»
    значило бы разное. Тема по умолчанию при этом всегда первая — с неё
    начинает тот, кто ничего не выбирал.
    """
    from src.bot.keyboards import _titles
    from src.core.preset import load_preset

    data = await (await booted.get("/api/bootstrap", headers=auth())).json()
    names = [p["name"] for p in data["presets"]]
    assert names[0] == Settings(_env_file=None, bot_token=TOKEN).default_preset
    assert names[1:] == [n for n, _ in _titles() if n != names[0]]
    assert [load_preset(n).order for n in names[1:]] == sorted(
        load_preset(n).order for n in names[1:]
    )


def test_today_starts_at_the_users_midnight_not_the_servers():
    """«+N сегодня» в приглашениях считается от полуночи человека."""
    from datetime import datetime, timezone

    from aiohttp.test_utils import make_mocked_request

    from src.webapp.server import _day_start

    now = datetime(2026, 11, 3, 20, 30, tzinfo=timezone.utc)
    # Владивосток (UTC+10): там уже 4 ноября, 06:30 — сутки начались в 14:00 UTC.
    vlad = make_mocked_request("GET", "/api/referrals?tz=-600")
    assert _day_start(vlad, now) == datetime(2026, 11, 3, 14, 0, tzinfo=timezone.utc)
    # Нью-Йорк (UTC−5): 15:30 3 ноября — сутки начались в 05:00 UTC.
    ny = make_mocked_request("GET", "/api/referrals?tz=300")
    assert _day_start(ny, now) == datetime(2026, 11, 3, 5, 0, tzinfo=timezone.utc)
    # Мусор и смещения больше суток — как UTC.
    for junk in ("abc", "99999"):
        bad = make_mocked_request("GET", f"/api/referrals?tz={junk}")
        assert _day_start(bad, now) == datetime(2026, 11, 3, 0, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ два видео


async def _uploaded(client, body: bytes, user_id: int = 42) -> str:
    r = await client.post("/api/uploads", json={"size": len(body)}, headers=auth(user_id))
    upload_id = (await r.json())["upload_id"]
    await client.put(f"/api/uploads/{upload_id}?offset=0", data=body, headers=auth(user_id))
    return upload_id


@pytest.fixture
def finish_env(client, monkeypatch):
    """Всё, что `finish` трогает до списания: пользователь, очередь, ffprobe.

    Нижний файл читается как видео со звуком, верхний (`TOP!`) — нет.
    """
    from types import SimpleNamespace

    from src.db import repo
    from src.pipeline import probe as probe_mod

    seen = {"created": 0}

    async def upsert_user(*a, **k):
        return SimpleNamespace(id=42, is_blocked=False, is_admin=False, balance=3, lang="ru",
                               language_code="ru", interface_lang=None)

    async def no_jobs(*a, **k):
        return 0

    async def create_job(*a, **k):
        seen["created"] += 1
        raise AssertionError("списали ролик за отказ")

    def fake_probe(path):
        if Path(path).read_bytes() == b"TOP!":
            raise RuntimeError("не видео")
        return SimpleNamespace(duration=20.0, has_audio=True)

    monkeypatch.setattr(repo, "upsert_user", upsert_user)
    monkeypatch.setattr(repo, "active_jobs_count", no_jobs)
    monkeypatch.setattr(repo, "create_job", create_job)
    monkeypatch.setattr(probe_mod, "probe", fake_probe)
    return seen


async def test_a_foreign_top_upload_is_not_found(client, finish_env):
    """Чужой верхний файл — 404, а свой нижний не забран и цел: его можно
    отправить снова с правильным верхним."""
    from src.webapp import server as mod

    cfg = client.server.app[mod.CFG_KEY]
    bottom = await _uploaded(client, b"abcd")
    foreign = await _uploaded(client, b"efgh", user_id=43)

    r = await client.post(
        f"/api/uploads/{bottom}/finish",
        json={"preset": "glass", "options": {}, "top_upload_id": foreign},
        headers=auth(),
    )
    assert r.status == 404
    assert (mod._upload_dir(cfg, bottom) / "data").read_bytes() == b"abcd"
    assert finish_env["created"] == 0


async def test_an_unreadable_top_is_refused_before_charging(client, finish_env):
    from src.webapp import server as mod

    cfg = client.server.app[mod.CFG_KEY]
    bottom = await _uploaded(client, b"abcd")
    top = await _uploaded(client, b"TOP!")

    r = await client.post(
        f"/api/uploads/{bottom}/finish",
        json={"preset": "glass", "options": {}, "top_upload_id": top},
        headers=auth(),
    )
    assert r.status == 415
    assert "верхнее" in (await r.json())["error"], "отказ не про верхнее видео — нижнее не прочиталось?"
    assert finish_env["created"] == 0
    left = list(Path(cfg.work_dir).glob("*"))
    assert left == [], f"после отказа на диске осталось: {[p.name for p in left]}"


async def test_a_user_can_open_two_uploads_for_one_reel(client):
    """Два файла одного ролика и один брошенный приём не упираются в квоту."""
    for _ in range(3):
        r = await client.post("/api/uploads", json={"size": 1024}, headers=auth())
        assert r.status == 200


async def test_a_picture_as_the_top_is_refused(client, finish_env, tmp_path):
    """Картинка проходит ffprobe, но верх из неё — ролик без верха."""
    import subprocess

    img = tmp_path / "a.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64",
                    "-frames:v", "1", str(img)], check=True)
    bottom = await _uploaded(client, b"abcd")
    top = await _uploaded(client, img.read_bytes())
    r = await client.post(
        f"/api/uploads/{bottom}/finish",
        json={"preset": "glass", "options": {}, "top_upload_id": top},
        headers=auth(),
    )
    assert r.status == 415
    assert finish_env["created"] == 0


async def test_a_finished_bottom_outlives_a_long_top_upload(client):
    """Нижнее догружено, верхнее идёт дольше срока заявки — нижнее обязано
    дожить до «Готово». Иначе пользователь залил два файла и получил 404."""
    import os
    import time as _time

    from src.webapp import server as mod

    cfg = client.server.app[mod.CFG_KEY]
    bottom = await _uploaded(client, b"abcd")
    r = await client.post("/api/uploads", json={"size": 8}, headers=auth())
    top = (await r.json())["upload_id"]

    # Нижний кусок пришёл давно, верхнее с тех пор всё грузится.
    old = _time.time() - mod.UPLOAD_LEASE - 300
    os.utime(mod._upload_dir(cfg, bottom) / "data", (old, old))
    await client.put(f"/api/uploads/{top}?offset=0", data=b"efgh", headers=auth())

    await client.post("/api/uploads", json={"size": 1}, headers=auth(43))  # чья-то загрузка запускает уборку
    assert (mod._upload_dir(cfg, bottom) / "data").exists(), "уборка снесла догруженное нижнее"


async def test_an_abandoned_finished_upload_still_expires(client):
    """Продление — только пока у хозяина идёт другая загрузка: брошенный
    догруженный файл по-прежнему уходит по сроку."""
    import os
    import time as _time

    from src.webapp import server as mod

    cfg = client.server.app[mod.CFG_KEY]
    bottom = await _uploaded(client, b"abcd")
    old = _time.time() - mod.UPLOAD_LEASE - 300
    os.utime(mod._upload_dir(cfg, bottom) / "data", (old, old))
    await client.post("/api/uploads", json={"size": 1}, headers=auth(43))
    assert not mod._upload_dir(cfg, bottom).exists()


# ------------------------------------------------------------------ токены


async def test_bootstrap_sells_token_packages(booted):
    """Тарифы как у Brandly: токены, «до N роликов», популярный, честная выгода."""
    body = await (await booted.get("/api/bootstrap", headers=auth())).json()
    assert body["video_cost"] == 23
    packs = body["packages"]
    assert [(p["tokens"], p["stars"], p["rub"], p["videos"]) for p in packs] == [
        (240, 749, 1099, 10), (590, 1899, 2799, 25), (1170, 3599, 5299, 50),
    ]
    assert [p["popular"] for p in packs] == [False, True, False]
    # Выгода — от цены токена в самом маленьком пакете и только если она есть:
    # 590 за 2799 ₽ токен ДОРОЖЕ, чем 240 за 1099 ₽, — «выгоды» там нет.
    assert [p["save"] for p in packs] == [0, 0, 1]


async def test_an_invoice_is_for_tokens_at_the_current_price(booted):
    from src.webapp import server as mod

    seen = {}

    async def create_invoice_link(**k):
        seen.update(k)
        return "https://t.me/$inv"

    booted.server.app[mod.BOT_KEY].create_invoice_link = create_invoice_link
    r = await booted.post("/api/invoice", json={"tokens": 590}, headers=auth())
    assert r.status == 200 and (await r.json())["link"] == "https://t.me/$inv"
    assert seen["payload"] == "tokens:590" and seen["currency"] == "XTR"
    assert seen["prices"][0].amount == 1899
    assert "590 токенов" in seen["title"]

    r = await booted.post("/api/invoice", json={"tokens": 300}, headers=auth())
    assert r.status == 400
