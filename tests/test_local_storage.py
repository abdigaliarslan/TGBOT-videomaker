"""Хранилище в папке: для выкладки на один сервер без S3."""

from __future__ import annotations

import pytest

from src.core.config import Settings
from src.storage.s3 import LocalStorage, Storage, make_storage


def store(tmp_path) -> LocalStorage:
    s = LocalStorage(Settings(_env_file=None, storage_backend="local", storage_dir=tmp_path / "st"))
    s.ensure_bucket()
    return s


async def test_round_trip_with_the_same_keys_as_s3(tmp_path):
    s = store(tmp_path)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"video")
    key = s.key_for("results", "job-1", "result.mp4")
    await s.upload(src, key, content_type="video/mp4")
    assert await s.exists(key)
    out = await s.download(key, tmp_path / "out" / "r.mp4")
    assert out.read_bytes() == b"video"
    await s.delete(key)
    assert not await s.exists(key)


async def test_a_missing_key_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        await store(tmp_path).download("results/nope.mp4", tmp_path / "x.mp4")


@pytest.mark.parametrize("key", ["../escape.mp4", "results/../../escape.mp4", "/etc/passwd"])
def test_keys_cannot_leave_the_storage_dir(tmp_path, key):
    with pytest.raises(ValueError):
        store(tmp_path).exists_sync(key)


def test_the_backend_is_chosen_by_settings(tmp_path):
    assert isinstance(make_storage(Settings(_env_file=None, storage_backend="local", storage_dir=tmp_path)), LocalStorage)
    assert type(make_storage(Settings(_env_file=None))) is Storage


def test_sweep_removes_only_old_files_and_empty_folders(tmp_path):
    import os
    from datetime import datetime, timedelta, timezone

    s = store(tmp_path)
    old = s._path("results/2026/09/01/job-old/result.mp4")
    fresh = s._path("results/2026/09/17/job-new/result.mp4")
    for p in (old, fresh):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    week_ago = (datetime.now(timezone.utc) - timedelta(days=9)).timestamp()
    os.utime(old, (week_ago, week_ago))

    removed = s.delete_older_than_sync(datetime.now(timezone.utc) - timedelta(days=8))
    assert removed == 1
    assert not old.exists() and fresh.exists()
    assert not (s.root / "results/2026/09/01").exists(), "пустые папки дат не копятся"
    assert s.root.is_dir()
