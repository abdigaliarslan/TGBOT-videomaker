# Образ бота и воркера. Один образ на оба процесса: код и зависимости общие,
# различается только команда запуска (см. deploy/compose.yml).
#
# Распознавание — только faster-whisper (CTranslate2). Стек whisperx с torch
# сюда не едет: он весит гигабайты, а продукт на сервере без GPU работает на
# faster-whisper (ASR_BACKEND=faster_whisper).

FROM node:20-bookworm-slim AS node

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models/hf

# ffmpeg — нарезка, кроп, звук. Остальное — библиотеки, без которых не стартует
# headless Chromium рендера (список из документации Remotion для Debian).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates curl \
        libnss3 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
        libasound2 libpango-1.0-0 libcairo2 libx11-6 libxext6 libxcb1 libexpat1 \
        fonts-liberation fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# Туннель для Mini App: бот ищет cloudflared в PATH и не качает его на старте.
RUN curl -fsSL -o /usr/local/bin/cloudflared \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    && chmod +x /usr/local/bin/cloudflared

WORKDIR /app

# Зависимости отдельным слоем: правка кода не должна перекачивать сотни мегабайт.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY render/package.json render/package-lock.json render/
RUN cd render && npm ci --no-audit --no-fund

# Chromium и бандл — при сборке, а не на первом ролике.
COPY render/ render/
RUN node render/render.mjs --ensure-browser && node render/render.mjs --bundle

COPY alembic.ini ./
COPY alembic/ alembic/
COPY assets/ assets/
COPY src/ src/

CMD ["python", "-m", "src.bot.main"]
