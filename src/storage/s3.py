"""Хранилище: MinIO (S3-совместимое) или папка на диске — исходники, результаты, кеш картинок.

boto3 синхронный, поэтому все вызовы уходят в поток: воркер не должен
блокировать loop на заливке гигабайтного файла.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from src.core.config import Settings, settings as default_settings

log = logging.getLogger(__name__)

SOURCES = "sources"
RESULTS = "results"
BROLL = "broll"


@lru_cache(maxsize=4)
def _client(endpoint: str, key: str, secret: str, region: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name=region,
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )


class Storage:
    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or default_settings
        self.bucket = self.cfg.s3_bucket

    @property
    def client(self):
        return _client(
            self.cfg.s3_endpoint, self.cfg.s3_access_key, self.cfg.s3_secret_key, self.cfg.s3_region
        )

    # ------------------------------------------------------------------ ключи
    @staticmethod
    def key_for(prefix: str, job_id: str, filename: str) -> str:
        day = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        return f"{prefix}/{day}/{job_id}/{filename}"

    # ------------------------------------------------------------------ синхронно
    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code not in ("404", "NoSuchBucket", "403"):
                raise
            log.info("создаю бакет %s", self.bucket)
            self.client.create_bucket(Bucket=self.bucket)

    def upload_sync(self, path: str | Path, key: str, *, content_type: str | None = None) -> str:
        path = Path(path)
        ctype = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": ctype})
        return key

    def download_sync(self, key: str, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(path))
        return path

    def exists_sync(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def delete_sync(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def delete_older_than_sync(self, cutoff: datetime) -> int:
        """Удаляет объекты, изменённые раньше `cutoff`. Возвращает, сколько удалено."""
        removed = 0
        pages = self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket)
        for page in pages:
            old = [{"Key": o["Key"]} for o in page.get("Contents", []) if o["LastModified"] < cutoff]
            for i in range(0, len(old), 1000):
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": old[i : i + 1000]})
                removed += len(old[i : i + 1000])
        return removed

    def presigned_url_sync(self, key: str, expires: int = 86_400) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires
        )

    # ------------------------------------------------------------------ асинхронно
    async def ensure_bucket_async(self) -> None:
        await asyncio.to_thread(self.ensure_bucket)

    async def upload(self, path: str | Path, key: str, *, content_type: str | None = None) -> str:
        return await asyncio.to_thread(self.upload_sync, path, key, content_type=content_type)

    async def download(self, key: str, path: str | Path) -> Path:
        return await asyncio.to_thread(self.download_sync, key, path)

    async def exists(self, key: str) -> bool:
        return await asyncio.to_thread(self.exists_sync, key)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self.delete_sync, key)

    async def delete_older_than(self, cutoff: datetime) -> int:
        return await asyncio.to_thread(self.delete_older_than_sync, cutoff)

    async def presigned_url(self, key: str, expires: int = 86_400) -> str:
        return await asyncio.to_thread(self.presigned_url_sync, key, expires)

    async def healthy(self) -> bool:
        try:
            await asyncio.to_thread(self.client.list_buckets)
            return True
        except Exception as e:  # noqa: BLE001 —проверка не должна ронять старт
            log.warning("хранилище недоступно: %s", e)
            return False


class LocalStorage(Storage):
    """То же хранилище, но в папке на диске — для выкладки на один сервер.

    Бот и воркер на одной машине видят один диск, и S3 между ними — лишний
    сервис. На сервере без root его вдобавок нечем поднять: сборки MinIO больше
    не раздаются, а Docker недоступен. Ключи те же, что у S3, поэтому БД и код
    не отличают один бэкенд от другого.
    """

    def __init__(self, cfg: Settings | None = None) -> None:
        super().__init__(cfg)
        self.root = Path(self.cfg.storage_dir).resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        # Ключ приходит из БД; «../» в нём не должен выводить за пределы хранилища.
        if self.root != path and self.root not in path.parents:
            raise ValueError(f"ключ выходит за пределы хранилища: {key!r}")
        return path

    def ensure_bucket(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def upload_sync(self, path: str | Path, key: str, *, content_type: str | None = None) -> str:
        import shutil

        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        shutil.copyfile(path, part)
        part.replace(dst)  # читатель никогда не увидит недописанный файл
        return key

    def download_sync(self, key: str, path: str | Path) -> Path:
        import shutil

        src = self._path(key)
        if not src.is_file():
            raise FileNotFoundError(f"в хранилище нет {key}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, path)
        return path

    def exists_sync(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete_sync(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def delete_older_than_sync(self, cutoff: datetime) -> int:
        """Файлы старше `cutoff` и опустевшие после них папки дат и задач."""
        if not self.root.is_dir():
            return 0
        border = cutoff.timestamp()
        removed = 0
        for path in list(self.root.rglob("*")):
            if path.is_file() and path.stat().st_mtime < border:
                path.unlink(missing_ok=True)
                removed += 1
        # Снизу вверх: сначала папки задач, потом дней, месяцев, лет.
        for folder in sorted((p for p in self.root.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
            try:
                folder.rmdir()  # только пустые
            except OSError:
                pass
        return removed

    def presigned_url_sync(self, key: str, expires: int = 86_400) -> str:
        raise NotImplementedError("у локального хранилища нет ссылок — ролик отдаёт сам бот")

    async def healthy(self) -> bool:
        return self.root.is_dir()


def make_storage(cfg: Settings | None = None) -> Storage:
    cfg = cfg or default_settings
    return LocalStorage(cfg) if cfg.storage_backend == "local" else Storage(cfg)


storage = make_storage()
