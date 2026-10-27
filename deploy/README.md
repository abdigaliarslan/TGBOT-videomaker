# Выкладка на сервер

Сейчас бот крутится на `dev@109.199.115.142` (хост `redstar` в `~/.ssh/config`)
**без sudo и без Docker** — службами `systemd --user`. `Linger=yes` у
пользователя `dev`, поэтому всё поднимается само после перезагрузки сервера, а
упавший процесс перезапускается через 5 секунд.

## Обновить

Само: push в `main` → GitHub Actions (`.github/workflows/deploy.yml`) гоняет
pytest и проверку типов рендера, затем запускает `deploy/update.sh`. Красные
тесты до сервера не доезжают. Две выкладки подряд встают в очередь, а не
обрывают друг друга. Запустить руками без коммита — Actions → deploy → Run workflow.

Для этого в репозитории нужен секрет `DEPLOY_SSH_KEY` — закрытый ключ, чья
открытая часть (`github-actions-reelbot-deploy`) лежит в `authorized_keys` у
`dev`. Ключ хоста сервера закреплён в `deploy/known_hosts`; сменится ключ хоста —
обновить этот файл, иначе выкладка честно упадёт на проверке.

Вручную, с машины разработчика:

```
deploy/update.sh
```

Код → зависимости (если менялись) → миграции → перезапуск → проверка, что все
службы живы (иначе скрипт выходит с ошибкой). Локальный бот на том же токене в
это время должен быть выключен.

⚠ rsync идёт с `--delete`: файл на сервере в `~/reelbot`, которого нет в git,
будет удалён. Свои бэкапы и данные держать в `~/reelbot-runtime`.

## Что где на сервере

| что | где |
|---|---|
| код | `~/reelbot` (rsync, без `.env`) |
| настройки | `~/reelbot/.env` (права 600), исходник паролей — `~/reelbot/deploy/.env` |
| Python, ffmpeg, cloudflared, uv | `~/reelbot-runtime/{venv,bin}` |
| Postgres 16 и Redis 7 | распакованы из пакетов Ubuntu в `~/reelbot-runtime/sys`, данные — `pgdata`, `redis` |
| ролики и исходники | `~/reelbot-runtime/storage` (`STORAGE_BACKEND=local`, MinIO не нужен) |
| модель распознавания | `~/reelbot-runtime/hf` |
| службы | `~/.config/systemd/user/reelbot-*.service` (копии из `deploy/systemd/`) |

Порты только на `127.0.0.1`: Postgres 5439, Redis 6390, Mini App 8099. Наружу
приложение выходит через cloudflared-туннель; его адрес новый при каждом
перезапуске бота, кнопку меню бот перевешивает сам.

Лимиты, чтобы не мешать соседним проектам: воркер — 3 ядра и 4 ГБ, бот — 1 ядро
и 768 МБ, Postgres 512 МБ, Redis 256 МБ.

## Посмотреть

```
ssh redstar systemctl --user status 'reelbot-*'
ssh redstar journalctl --user -u reelbot-bot -f
ssh redstar journalctl --user -u reelbot-worker -f
```

## Docker

`Dockerfile` и `deploy/compose.yml` — тот же стек в контейнерах (образ собран и
проверен). Нужен доступ к Docker: `sudo usermod -aG docker dev`.
