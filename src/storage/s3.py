"""Хранилище: MinIO (S3-совместимое) — исходники, результаты, кеш картинок.

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

    async def presigned_url(self, key: str, expires: int = 86_400) -> str:
        return await asyncio.to_thread(self.presigned_url_sync, key, expires)

    async def healthy(self) -> bool:
        try:
            await asyncio.to_thread(self.client.list_buckets)
            return True
        except Exception as e:  # noqa: BLE001 —проверка не должна ронять старт
            log.warning("хранилище недоступно: %s", e)
            return False


storage = Storage()
