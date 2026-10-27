"""Единственный источник конфигурации. Секреты только отсюда, никогда в коде (§6)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

# Модели по умолчанию для каждого провайдера (§4).
#
# `smart` — главный вызов, от него зависит качество продукта, поэтому там
# самая сильная модель линейки. `cheap` — правки индексами и метаданные.
# `vision` — план b-roll по кадрам.
PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "anthropic": {
        "cheap": "claude-haiku-4-5",
        "smart": "claude-opus-5",
        "vision": "claude-sonnet-5",
    },
    "deepseek": {
        "cheap": "deepseek-v4-flash",
        # Замерено на задаче `direct` (одинаковый транскрипт, один и тот же план
        # на выходе): v4-pro — 135.6 с и 8545 выходных токенов ($0.034),
        # v4-flash — 8.6 с и 1186 токенов ($0.0016). В 16 раз быстрее и в 21 раз
        # дешевле при том же результате. Кто из них лучше НА САМОМ ДЕЛЕ, решит
        # только эвал-сет из §7; до него v4-pro включается одной строкой:
        # LLM_TIER_SMART=deepseek-v4-pro
        "smart": "deepseek-v4-flash",
        "vision": "deepseek-v4-flash-vision-exp",
    },
    # ⚠ На бесплатном тире Gemini модели Pro недоступны совсем: ключ отвечает
    # 429 с `limit: 0` на квоте `generate_content_free_tier_requests`. Это не
    # исчерпанный лимит, а его отсутствие. Flash-модели при этом работают, в
    # том числе с картинками.
    #
    # Доступность flash-моделей на бесплатном ключе тоже разная. Замерено, по
    # 6 одинаковых запросов на модель:
    #   gemini-3.5-flash       6/6, 4.9 с   ← самая надёжная
    #   gemini-3.1-flash-lite  6/6, 3.3 с
    #   gemini-3.8-flash       3/6 (503)
    #   gemini-3.7-flash       2/6 (503)
    #   gemini-3.6-flash       0/6 (400 на thinkingConfig, см. providers.py)
    # «503 high demand» — это загрузка, а не наша ошибка: на платном ключе
    # картина другая. Поэтому в defaults остаются сильнейшие модели линейки, а
    # бесплатный ключ переводится на надёжные двумя строками в .env:
    #   LLM_TIER_SMART=gemini-3.5-flash
    #   LLM_TIER_VISION=gemini-3.5-flash
    "gemini": {
        "cheap": "gemini-3.1-flash-lite",
        "smart": "gemini-3.1-pro-preview",
        "vision": "gemini-3.7-flash",
    },
}


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

    # Заливка готового ролика в Telegram — это десятки мегабайт. Дефолтных
    # 60 секунд не хватает, и ролик не доходит при успешном монтаже.
    bot_upload_timeout_s: float = 900.0

    admin_ids: list[int] = Field(default_factory=list)

    # ------------------------------------------------------------------ инфраструктура
    postgres_dsn: str = "postgresql+asyncpg://reelbot:reelbot@localhost:5432/reelbot"
    redis_url: str = "redis://localhost:6379/0"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "reelbot"
    s3_secret_key: str = "reelbot-secret"
    s3_bucket: str = "reelbot"
    s3_region: str = "us-east-1"

    # `s3` — MinIO/S3 (docker compose, локальная разработка). `local` — папка на
    # диске: для выкладки на один сервер, где бот и воркер видят одну файловую
    # систему и держать ради них отдельный S3 незачем.
    storage_backend: Literal["s3", "local"] = "s3"
    storage_dir: Path = REPO_ROOT / "data" / "storage"
    # Сколько дней хранить исходники и готовые ролики. Потом уборщик их удаляет:
    # диск сервера не резиновый, а ролик у пользователя уже лежит в чате. 0 —
    # не удалять никогда.
    storage_retention_days: int = 7

    work_dir: Path = REPO_ROOT / "data" / "work"

    # ------------------------------------------------------------------ LLM (§4)
    #
    # §4 требует ОДИН абстрактный клиент и тиры вместо имён моделей в логике.
    # Провайдер — такая же деталь реализации, как имя модели: меняется здесь.
    llm_provider: Literal["anthropic", "deepseek", "gemini"] = "anthropic"

    anthropic_api_key: str = ""
    deepseek_api_key: str = ""
    gemini_api_key: str = ""

    anthropic_base_url: str = "https://api.anthropic.com"
    deepseek_base_url: str = "https://api.deepseek.com"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # Пусто → модель по умолчанию для выбранного провайдера (см. PROVIDER_MODELS).
    llm_tier_cheap: str = ""
    llm_tier_smart: str = ""
    llm_tier_vision: str = ""

    # Размышления reasoning-моделей: считаются в max_tokens и оплачиваются как
    # выход. На структурных задачах не нужны — см. комментарий в providers.py.
    llm_thinking: bool = False
    llm_max_retries: int = 1  # §4: ретрай ровно один
    llm_timeout_s: float = 180.0
    # Нет ключа → детерминированный режиссёр (§3, «фолбэк»). Ролик всё равно соберётся.
    llm_enabled: bool = True

    # ------------------------------------------------------------------ ASR (M1)
    # auto → whisperx, если установлен: границы слов считает wav2vec2 по звуку.
    # На поставляемых настройках (обрезка включена у обоих) — 46 мс средней
    # ошибки против 216 мс у faster_whisper на фикстуре и 25 мс против 88 мс на
    # русской речи; числа, цена и разбор faster_whisper/medium — в комментарии к
    # pick_backend в pipeline/asr.py. faster_whisper остаётся фолбэком: втрое
    # легче по памяти и не требует модели выравнивания.
    asr_backend: Literal["auto", "whisperx", "faster_whisper"] = "auto"
    # small, а не medium: границы слов считает wav2vec2, и на ОДИНАКОВОМ тексте
    # размер модели Whisper их не двигает (на фикстуре списки слов совпали
    # побуквенно; на русском medium меняет сам текст, 38 слов против 39, и там
    # где слова совпали — тайминги те же). Точность та же, цена вдвое выше.
    whisper_model: str = "small"
    whisper_language: str = "ru"  # пусто = автоопределение
    whisper_device: str = "auto"  # auto|cpu|cuda
    whisper_compute_type: str = "auto"  # auto|int8|float16|float32
    # 4, а не «все ядра»: пиковое потребление важнее пары секунд. См. ffmpeg_threads.
    whisper_cpu_threads: int = 4
    whisper_beam_size: int = 5
    # Сегменты со средней вероятностью слов ниже порога — галлюцинации (музыка/шум).
    whisper_min_confidence: float = 0.55
    # Оба бэкенда растягивают последнее слово перед паузой на саму паузу (замер —
    # в комментарии к trim_word_silence в pipeline/asr.py). Пауза, спрятанная
    # внутрь слова, не видна нарезке: она считает паузы как зазор между словами.
    # Обрезка возвращает границы к фактической речи по энергии сигнала: на
    # эталонном ролике нарезка находит 3.44 с пауз вместо 0.98 с.
    asr_trim_word_silence: bool = True

    # ------------------------------------------------------------------ лимиты
    min_video_seconds: float = 3.0
    max_video_minutes: float = 3.0
    max_concurrent_jobs: int = 1
    queue_max_size: int = 20

    # ------------------------------------------------------------------ нарезка пауз (M6)
    # Пауза длиннее — вырезается. 0.8 с резало обычный вдох посреди фразы, и
    # ролик выходил рваным: каждая такая пауза — видимый скачок в кадре.
    # 1.2 с — это уже настоящая остановка, а не дыхание.
    cut_max_gap: float = 1.2
    # Заминка короче — НЕ режется. Вырезание «эм» длиной 0.15 с стоит зрителю
    # целого скачка, а экономит полкадра: цена не сходится. Режем только те,
    # что слышны как затяжка.
    cut_min_filler: float = 0.35
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
    # Каждый поток рендера — отдельная вкладка Chromium плюс кодирование.
    # На ноутбуке два потока дают тот самый скачок тока, после которого машина
    # гаснет без записи в журнал. На машине с нормальным питанием ставится обратно.
    render_concurrency: int = 1
    # Пресет x264 при сборке кадров в mp4. На сервере рендер упирается в
    # отрисовку кадров, а не в кодек: замер на 10 с видео — medium 98 с,
    # veryfast 92 с при том же размере файла. Дефолт прежний.
    render_x264_preset: str = "medium"
    # Режим GL в Chromium рендера. Пусто — прежний "swangle" (программная
    # эмуляция WebGL). Замер на сервере без GPU, 3 потока: swangle 92 с,
    # angle 65 с, swiftshader 64 с; картинка та же (PSNR 52 дБ).
    render_gl: str = ""

    # Потолок потоков ffmpeg. 0 = отдать решение ffmpeg (все ядра) — так можно
    # на сервере, но не на ноутбуке.
    ffmpeg_threads: int = 4

    # Качество финального файла. Замерено: при crf=18 двадцатисекундный ролик
    # весит 22 МБ (8.9 Мбит/с) и заливается в Telegram дольше, чем монтируется.
    # Телеграм всё равно пережимает — отдавать визуально-безупречный поток
    # некуда. 22 даёт примерно вдвое меньший файл без заметной разницы.
    render_crf: int = 22
    default_preset: str = "glass"
    # Шрифты ставятся в контейнер рендера (§5 M5 ⚠), тут — путь для локального прогона.
    fonts_dir: Path = REPO_ROOT / "assets" / "fonts"

    # ------------------------------------------------------------------ звук (Ф2)
    # Этап audio_post между нарезкой и рендером: loudnorm, подложка, дакинг, эффекты.
    audio_post_enabled: bool = True
    # Свой трек кладут сюда, имя файла включают в секции audio пресета. Репозиторий
    # музыки не везёт: чужая лицензия в чужом продукте — это не «мелочь потом».
    music_dir: Path = REPO_ROOT / "assets" / "music"
    # Эффекты синтезируются процедурно и лежат готовыми; папка пустая — соберутся сами.
    sfx_dir: Path = REPO_ROOT / "assets" / "sfx"

    # ------------------------------------------------------------------ b-roll (M7)
    pexels_api_key: str = ""
    wavespeed_api_key: str = ""
    wavespeed_model: str = "wavespeed-ai/z-image/turbo"
    # Ровно 9:16, как кадр: тогда полноэкранная вставка не нуждается в размытой
    # подложке, а `cover` ничего не режет. Прежние 768*1024 — это 3:4, и такая
    # картинка приходила в кадр с размытыми полями сверху и снизу.
    #
    # ⚠ Модель принимает не всякий размер: замерено на живом ключе, 864*1536 и
    # 768*1365 проходят, 900*1600 и 1080*1920 отвергаются с «size does not meet
    # the constraints». Здесь самый большой из прошедших.
    wavespeed_size: str = "864*1536"
    fal_api_key: str = ""
    # Тот же 9:16, что у WaveSpeed. fal берёт любой размер, но платят за
    # мегапиксели: полный 1080*1920 вышел бы вдвое дороже, а Ken Burns и
    # пережатие в Telegram разницу съедают.
    fal_size: str = "864*1536"
    broll_max_per_8s: float = 1.0
    broll_enabled: bool = True
    # Откуда берётся картинка: рисуем или ищем.
    #
    # `generate` — сначала генерация, сток запасным путём. Так вставка попадает
    # в смысл фразы, а не в ближайшее к ней слово: сток по запросу «камера»
    # отдаёт чужую камеру на чужом столе, и это видно. Эталонный ролик
    # (`STYLE_PLAN.md §5`) тем и отличается, что графика в нём нарисована под
    # сценарий; там это сочли невоспроизводимым, здесь — воспроизводим.
    #
    # `stock` возвращает прежний порядок: видео → фото → генерация.
    broll_source: Literal["generate", "stock"] = "generate"
    # Чем рисуем.
    #
    # `gemini` умеет принимать кадры ролика референсом палитры и света — это
    # лучший способ вписать вставку в ролик, и весь код для него написан. Но
    # умолчанием он быть не может: ⚠ на БЕСПЛАТНОМ тарифе у всех его моделей
    # картинок квота ровно нулевая. Замерено на живом ключе, ответ один и тот
    # же для `gemini-3.1-flash-image`, `gemini-3.1-flash-lite-image`,
    # `gemini-2.5-flash-image` и `gemini-3-pro-image`:
    #   «Rate limit exceeded … (limit: 0 requests per day on Free Tier)»
    # Это не исчерпанный лимит, а его отсутствие. Появится платный ключ —
    # переключается одной строкой IMAGE_PROVIDER=gemini.
    #
    # `wavespeed` референсов не принимает, зато работает: z-image/turbo стоит
    # $0.005 за картинку. Связь с роликом у него держится только на словах
    # стилевой памятки (`ReelStyle` в pipeline/broll.py).
    #
    # `fal` — та же z-image/turbo без опроса: один синхронный запрос, ответ за
    # 1-3 с. Цена $0.005 за мегапиксель, кадр `fal_size` стоит ~$0.0066.
    image_provider: Literal["gemini", "wavespeed", "fal"] = "wavespeed"
    # Пусто → модель по умолчанию для провайдера (PROVIDER_IMAGE_MODELS).
    image_model: str = ""
    # Разрешение у Gemini: 1K хватает на вертикаль 1080×1920, 2K стоит дороже.
    image_size: str = "1K"
    # Спрашивать ли у vision-модели, то ли нарисовано и нет ли на картинке
    # надписей. Одна дешёвая проверка на вставку и одна повторная попытка;
    # не помогло — уходим на сток. Ноль отключает проверку вместе с её ценой.
    broll_verify: bool = True
    # Сколько дней держать скачанные и сгенерированные картинки. Кеш общий для
    # всех роликов: один и тот же запрос не должен стоить дважды, а генерация,
    # в отличие от стока, стоит денег.
    broll_cache_days: int = 14
    # Брать ли видео вместо фото (М4). Двухсекундный клип читается как монтаж
    # мгновенно, но стоит дороже: единицы мегабайт на скачивании и второй
    # видеопоток на композитинге. Выключение возвращает поведение до М4 —
    # только фото; откат на фото при отсутствии подходящего клипа работает
    # всегда и без этой ручки.
    broll_video: bool = True

    # ------------------------------------------------------------------ Mini App
    # Публичный HTTPS-адрес приложения. Telegram открывает Mini App только по
    # https, поэтому локальный порт наружу выводит туннель: пустой URL при
    # `webapp_tunnel` значит «поднять туннель и взять адрес у него».
    webapp_url: str = ""
    webapp_tunnel: bool = False
    webapp_port: int = 8080
    webapp_host: str = "127.0.0.1"
    # Бинарник cloudflared. Нет в PATH — скачивается сюда при первом старте.
    cloudflared_bin: Path = REPO_ROOT / "data" / "bin" / "cloudflared"
    # Сколько живёт подпись `initData`. Telegram выдаёт её при открытии приложения,
    # и приложение держат открытым часами — сутки не дают украденной строке жить
    # вечно и не выкидывают честного пользователя посреди загрузки.
    webapp_auth_max_age_s: int = 24 * 3600
    # Загрузка идёт кусками: бесплатный туннель Cloudflare режет тело запроса на
    # 100 МБ, а трёхминутное видео с телефона бывает больше.
    webapp_chunk_mb: int = 8
    webapp_max_upload_mb: int = 1024

    # ------------------------------------------------------------------ оплата
    # Пакеты «роликов → звёзд». Ключ — сколько роликов, значение — цена в XTR.
    # Владельцы из ADMIN_IDS монтируют бесплатно и баланс не тратят.
    star_packages: dict[int, int] = Field(default_factory=lambda: {1: 50, 5: 200, 20: 600})

    # ------------------------------------------------------------------ прочее
    debug: bool = False
    log_level: str = "INFO"

    @field_validator("star_packages", mode="before")
    @classmethod
    def _parse_star_packages(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = json.loads(v) if v.strip() else {}
        if isinstance(v, dict):
            packages = {int(k): int(price) for k, price in v.items()}
            if any(k <= 0 or price <= 0 for k, price in packages.items()):
                raise ValueError("STAR_PACKAGES: и число роликов, и цена обязаны быть > 0")
            return dict(sorted(packages.items()))
        return v

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
    def llm_api_key(self) -> str:
        return {
            "anthropic": self.anthropic_api_key,
            "deepseek": self.deepseek_api_key,
            "gemini": self.gemini_api_key,
        }.get(self.llm_provider, "")

    @property
    def llm_base_url(self) -> str:
        return {
            "anthropic": self.anthropic_base_url,
            "deepseek": self.deepseek_base_url,
            "gemini": self.gemini_base_url,
        }[self.llm_provider]

    @property
    def llm_ready(self) -> bool:
        """LLM реально доступна: есть ключ выбранного провайдера и не выключено."""
        return bool(self.llm_api_key) and self.llm_enabled

    def model_for_tier(self, tier: str) -> str:
        """Тир → имя модели. Пустая настройка = значение по умолчанию провайдера."""
        if tier not in ("cheap", "smart", "vision"):
            raise ValueError(f"неизвестный тир модели: {tier!r}")
        override = {
            "cheap": self.llm_tier_cheap,
            "smart": self.llm_tier_smart,
            "vision": self.llm_tier_vision,
        }[tier]
        if override:
            return override
        return PROVIDER_MODELS[self.llm_provider][tier]

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
