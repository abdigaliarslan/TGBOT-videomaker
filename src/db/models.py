"""Схема БД: пользователи, задачи, вызовы модели.

Отдельная таблица под вызовы LLM — не украшение: себестоимость ролика надо
видеть с первого дня (§4), иначе экономика вскроется поздно (§7).
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class User(Base):
    __tablename__ = "users"

    # id пользователя в телеграме, он же первичный ключ: свой суррогат не нужен.
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    username: Mapped[str | None] = mapped_column(String(64))
    first_name: Mapped[str | None] = mapped_column(String(128))
    language_code: Mapped[str | None] = mapped_column(String(8))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_blocked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    preset: Mapped[str] = mapped_column(String(32), default="glass", nullable=False)
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    videos_done: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    jobs: Mapped[list["Job"]] = relationship(back_populates="user", lazy="noload")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", native_enum=False, length=16),
        default=JobStatus.QUEUED,
        nullable=False,
        index=True,
    )
    stage: Mapped[str | None] = mapped_column(String(32))
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # источник
    source_file_id: Mapped[str | None] = mapped_column(String(256))
    source_key: Mapped[str | None] = mapped_column(String(512))  # ключ в S3
    # Путь, который отдал локальный Bot API server (M0 ⚠): файл лежит на общем
    # томе, скачивать его через API не нужно и нельзя.
    source_local_path: Mapped[str | None] = mapped_column(String(1024))
    source_duration: Mapped[float | None] = mapped_column(Float)
    source_size: Mapped[int | None] = mapped_column(BigInteger)

    # результат
    result_key: Mapped[str | None] = mapped_column(String(512))
    result_duration: Mapped[float | None] = mapped_column(Float)
    preset: Mapped[str] = mapped_column(String(32), default="glass", nullable=False)

    # EDL хранится целиком: сохранённый EDL + другой пресет = новое видео
    # без повторного ASR и LLM (§0.4).
    edl: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    words: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    timings: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    edl_from_llm: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    error: Mapped[str | None] = mapped_column(Text)
    chat_id: Mapped[int | None] = mapped_column(BigInteger)
    status_message_id: Mapped[int | None] = mapped_column(BigInteger)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="jobs", lazy="noload")
    llm_calls: Mapped[list["LLMCall"]] = relationship(
        back_populates="job", lazy="noload", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_jobs_user_created", "user_id", "created_at"),)

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


class LLMCall(Base):
    """Один вызов модели. Себестоимость ролика — сумма по job_id."""

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    prompt: Mapped[str] = mapped_column(String(64), nullable=False)
    tier: Mapped[str] = mapped_column(String(16), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)

    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cache_creation_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    job: Mapped[Job | None] = relationship(back_populates="llm_calls", lazy="noload")
