#!/usr/bin/env bash
# Обновление бота на сервере без sudo и Docker (systemd --user).
#
#   deploy/update.sh               # хост `redstar` из ~/.ssh/config
#   DEPLOY_HOST=user@host deploy/update.sh
#
# Код → миграции → перезапуск. Postgres и Redis не трогаются.
# Новые Python- или npm-зависимости ставятся сами, если поменялись
# pyproject.toml или render/package-lock.json.
set -euo pipefail
HOST="${DEPLOY_HOST:-redstar}"
cd "$(dirname "$0")/.."

# Отметка о том, ЧТО именно выложено. Сегодня (3 ноября 2026) полдня
# ушло на выяснение, доехал ли код: службы живы, а на сервере лежал
# прошлый коммит. Проверять приходилось по содержимому файлов. Теперь
# ответ лежит рядом: `ssh <хост> cat reelbot/.deployed`.
#
# Грязное дерево помечается отдельно: rsync везёт рабочие файлы, а не
# коммит, и «выложен 9cb0c6c» без этой пометки было бы неправдой.
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "без-git")
[ -n "$(git status --porcelain 2>/dev/null)" ] && SHA="$SHA+несохранённые-правки"
STAMP="$SHA $(date -Is) $(whoami)@$(hostname)"

rsync -az --delete --itemize-changes \
  --exclude '.git/' --exclude '.venv/' --exclude 'data/' --exclude 'logs/' \
  --exclude 'samples/' --exclude '.kilo/' --exclude '.claude/' --exclude '.playwright-mcp/' \
  --exclude '.pytest_cache/' --exclude '.ruff_cache/' --exclude '.remotion/' \
  --exclude '__pycache__/' --exclude 'render/node_modules/' --exclude 'render/.cache/' \
  --exclude '*.mp4' --exclude '*.mov' --exclude '*.wav' \
  --exclude '.env' --exclude '.env.docker' --exclude 'deploy/.env' \
  ./ "$HOST:reelbot/" > /tmp/reelbot-rsync.log
# Отдельно от rsync: в одном конвейере с `|| true` его провал терялся, и службы
# перезапускались на старом коде под видом успешной выкладки.
grep -E 'pyproject|package-lock|\.py$|\.tsx?$|\.mjs$|\.css$|\.js$|\.html$' /tmp/reelbot-rsync.log | head -30 || true

ssh "$HOST" "STAMP='$STAMP' bash -s" <<'REMOTE'
set -euo pipefail
R=~/reelbot-runtime
cd ~/reelbot
if [ pyproject.toml -nt "$R/requirements.txt" ]; then
  echo "→ Python-зависимости"
  python3 -c "import tomllib; print(chr(10).join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > "$R/requirements.txt"
  UV_HTTP_TIMEOUT=300 "$R/bin/uv" pip install --python "$R/venv/bin/python" -r "$R/requirements.txt"
fi
if [ render/package-lock.json -nt render/node_modules/.package-lock.json ]; then
  echo "→ npm-зависимости"
  source ~/.nvm/nvm.sh >/dev/null && nvm use 20 >/dev/null
  (cd render && npm ci --no-audit --no-fund \
    && npm install --no-save --no-audit --no-fund "@rspack/binding-linux-x64-gnu@$(node -p "require('./node_modules/@rspack/binding/package.json').version")" \
    && node render.mjs --ensure-browser)
fi
cp deploy/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user restart reelbot-migrate
systemctl --user restart reelbot-worker reelbot-janitor reelbot-bot
sleep 20
systemctl --user is-active reelbot-postgres reelbot-redis reelbot-worker reelbot-janitor reelbot-bot
# Пишем ПОСЛЕ перезапуска: отметка значит «этот код запущен», а не «этот код
# скопирован». Сам файл живёт только на сервере — следующий rsync его снесёт
# (--delete) и этот же шаг напишет заново.
printf '%s\n' "$STAMP" > ~/reelbot/.deployed
echo "выложено: $STAMP"
journalctl --user -u reelbot-bot -n 30 --no-pager | grep -E "Mini App:|Run polling|Traceback|ERROR" || true
REMOTE
