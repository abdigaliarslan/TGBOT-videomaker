"""Единственный источник конфигурации. Секреты только отсюда, никогда в коде (§6)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ telegram
    bot_token: str = ""
    telegram_api_id: str = ""
    telegram_api_hash: str = ""

    # Пусто → облачный Bot API (лимит файла 20 МБ).
    # http://localhost:8081 → локальный сервер из docker-compose (лимит 2 ГБ).
    bot_api_url: str = ""
    # Один и тот же том, смонтированный в двух местах. Локальный Bot API отдаёт
    # ПУТЬ внутри контейнера; чтобы прочитать файл, его надо перевести в host-путь.
    bot_api_data_dir_host: Path = REPO_ROOT / "data" / "telegram-bot-api"
    bot_api_data_dir_container: str = "/var/lib/telegram-bot-api"

    admin_ids: list[int] = Field(default_factory=list)

    # ------------------------------------------------------------------ инфраструктура
    postgres_dsn: str = "postgresql+asyncpg://reelbot:reelbot@localhost:5432/reelbot"
    redis_url: str = "redis://localhost:6379/0"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "reelbot"
    s3_secret_key: str = "reelbot-secret"
    s3_bucket: str = "reelbot"
    s3_region: str = "us-east-1"

    work_dir: Path = REPO_ROOT / "data" / "work"

    # ------------------------------------------------------------------ LLM (§4)
    anthropic_api_key: str = ""
    # Тиры, а не имена моделей, в вызывающем коде. Меняются здесь, не в логике.
    llm_tier_cheap: str = "claude-haiku-4-5-20251001"
    llm_tier_smart: str = "claude-opus-5"
    llm_tier_vision: str = "claude-sonnet-5"
    llm_max_retries: int = 1  # §4: ретрай ровно один
    llm_timeout_s: float = 180.0
    # Нет ключа → детерминированный режиссёр (§3, «фолбэк»). Ролик всё равно соберётся.
    llm_enabled: bool = True

    # ------------------------------------------------------------------ ASR (M1)
    # whisperx — форсированное выравнивание, нужен torch+CUDA.
    # faster_whisper — CTranslate2, работает на CPU, тайминги слов из самой модели.
    asr_backend: Literal["auto", "whisperx", "faster_whisper"] = "auto"
    whisper_model: str = "small"
    whisper_language: str = "ru"  # пусто = автоопределение
    whisper_device: str = "auto"  # auto|cpu|cuda
    whisper_compute_type: str = "auto"  # auto|int8|float16|float32
    whisper_cpu_threads: int = 8
    whisper_beam_size: int = 5
    # Сегменты со средней вероятностью слов ниже порога — галлюцинации (музыка/шум).
    whisper_min_confidence: float = 0.55

    # ------------------------------------------------------------------ лимиты
    min_video_seconds: float = 3.0
    max_video_minutes: float = 3.0
    max_concurrent_jobs: int = 1
    queue_max_size: int = 20

    # ------------------------------------------------------------------ нарезка пауз (M6)
    cut_max_gap: float = 0.8  # пауза длиннее — вырезается
    cut_pad_before: float = 0.15
    cut_pad_after: float = 0.25
    cut_min_segment: float = 0.3
    min_speech_ratio: float = 0.05  # меньше — считаем, что речи нет, не режем

    # ------------------------------------------------------------------ субтитры (M2/M3)
    sub_max_words: int = 4  # §3: 3-5 слов в блоке
    sub_max_chars: int = 22
    sub_gap_split: float = 0.6  # пауза внутри блока длиннее — рвём блок
    sub_tail: float = 0.15  # сколько блок висит после последнего слова

    # ------------------------------------------------------------------ рендер
    render_fps: int = 30
    render_width: int = 1080
    render_height: int = 1920
    render_backend: Literal["ass", "remotion"] = "remotion"
    render_concurrency: int = 2
    default_preset: str = "glass"
    # Шрифты ставятся в контейнер рендера (§5 M5 ⚠), тут — путь для локального прогона.
    fonts_dir: Path = REPO_ROOT / "assets" / "fonts"

    # ------------------------------------------------------------------ b-roll (M7)
    pexels_api_key: str = ""
    wavespeed_api_key: str = ""
    wavespeed_model: str = "wavespeed-ai/z-image/turbo"
    wavespeed_size: str = "768*1024"
    broll_max_per_8s: float = 1.0
    broll_enabled: bool = True

    # ------------------------------------------------------------------ прочее
    debug: bool = False
    log_level: str = "INFO"

    @field_validator("admin_ids", mode="before")
    @classmethod
    def _parse_admin_ids(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("["):
                return json.loads(v)
            return [int(x) for x in v.replace(",", " ").split()]
        return v

    @field_validator("whisper_language", mode="before")
    @classmethod
    def _blank_language_is_auto(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return ""
        return v

    # -------------------------------------------------------------- производные
    @property
    def max_video_seconds(self) -> float:
        return self.max_video_minutes * 60.0

    @property
    def llm_ready(self) -> bool:
        """LLM реально доступна: есть ключ и не выключено флагом."""
        return bool(self.anthropic_api_key) and self.llm_enabled

    def model_for_tier(self, tier: str) -> str:
        try:
            return {
                "cheap": self.llm_tier_cheap,
                "smart": self.llm_tier_smart,
                "vision": self.llm_tier_vision,
            }[tier]
        except KeyError:
            raise ValueError(f"unknown model tier: {tier!r}") from None

    def bot_api_path_to_local(self, path: str) -> Path:
        """Локальный Bot API отдаёт путь внутри своего контейнера (M0 ⚠).

        Переводим его в путь на хосте по смонтированному тому. Если путь уже
        локальный (облачный Bot API или сервер на том же хосте) — возвращаем как есть.
        """
        p = Path(path)
        container_root = Path(self.bot_api_data_dir_container)
        try:
            rel = p.relative_to(container_root)
        except ValueError:
            return p
        return (self.bot_api_data_dir_host / rel).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.work_dir.mkdir(parents=True, exist_ok=True)
    return s


settings = get_settings()
