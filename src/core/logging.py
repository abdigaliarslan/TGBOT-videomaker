"""Логирование + метрики этапов (§7: мерить с первого дня)."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

_configured = False


def setup_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    for noisy in ("botocore", "boto3", "urllib3", "aiosqlite", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


@contextmanager
def stage(name: str, log: logging.Logger | None = None, **extra):
    """Замер длительности этапа. Пишет в лог; вызывающий может сложить в БД."""
    lg = log or logging.getLogger("stage")
    t0 = time.perf_counter()
    meta: dict[str, object] = dict(extra)
    try:
        yield meta
    finally:
        dt = time.perf_counter() - t0
        meta["seconds"] = round(dt, 2)
        tail = " ".join(f"{k}={v}" for k, v in meta.items())
        lg.info("· %-18s %6.2fs  %s", name, dt, tail)
