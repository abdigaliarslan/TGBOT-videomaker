#!/usr/bin/env bash
# Выкладка на сервер: код → сборка → запуск. Запускать из корня репозитория.
#
#   deploy/deploy.sh                 # хост по умолчанию — `redstar` из ~/.ssh/config
#   DEPLOY_HOST=user@host deploy/deploy.sh
#
# Что делает:
#   1. rsync кода в ~/reelbot на сервере (без .venv, данных, видео и секретов);
#   2. при первом запуске собирает deploy/.env из локального .env и генерирует
#      пароли базы и хранилища — дальше файл на сервере не перезаписывается;
#   3. docker compose up -d --build.
#
# ⚠ Один токен — один запущенный бот. Остановите локального бота ДО запуска на
# сервере, иначе Telegram будет рвать long polling то одному, то другому.
set -euo pipefail

HOST="${DEPLOY_HOST:-redstar}"
REMOTE_DIR="${REMOTE_DIR:-reelbot}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "→ код в $HOST:~/$REMOTE_DIR"
rsync -az --delete \
  --exclude '.git/' --exclude '.venv/' --exclude 'data/' --exclude 'logs/' \
  --exclude 'samples/' --exclude '.kilo/' --exclude '.claude/' --exclude '.playwright-mcp/' \
  --exclude '.pytest_cache/' --exclude '.ruff_cache/' --exclude '.remotion/' \
  --exclude '__pycache__/' --exclude 'render/node_modules/' --exclude 'render/.cache/' \
  --exclude '*.mp4' --exclude '*.mov' --exclude '*.wav' \
  --exclude '.env' --exclude '.env.docker' --exclude 'deploy/.env' \
  ./ "$HOST:$REMOTE_DIR/"

if ! ssh "$HOST" "test -f $REMOTE_DIR/deploy/.env"; then
  echo "→ первый запуск: собираю deploy/.env на сервере"
  pg_pass="$(openssl rand -hex 24)"
  s3_pass="$(openssl rand -hex 24)"
  {
    # Локальные адреса инфраструктуры не нужны: compose задаёт свои.
    grep -vE '^(POSTGRES_DSN|REDIS_URL|S3_ENDPOINT|S3_ACCESS_KEY|S3_SECRET_KEY|BOT_API_URL|WEBAPP_URL|WEBAPP_HOST|POSTGRES_PORT|REDIS_PORT)=' .env
    echo
    echo "# --- сгенерировано deploy.sh $(date -u +%F)"
    echo "POSTGRES_PASSWORD=$pg_pass"
    echo "S3_SECRET_KEY=$s3_pass"
  } | ssh "$HOST" "umask 077 && cat > $REMOTE_DIR/deploy/.env"
fi

echo "→ сборка и запуск"
ssh "$HOST" "cd $REMOTE_DIR && docker compose -f deploy/compose.yml up -d --build --remove-orphans && docker compose -f deploy/compose.yml ps"
