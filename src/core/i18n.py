"""Язык интерфейса: русский и английский. Все тексты, которые видит пользователь, — здесь.

Переводится только ИНТЕРФЕЙС: сообщения бота, приложение, стадии и ошибки
монтажа. Язык речи в роликах от этого выбора не зависит (`WHISPER_LANGUAGE`).

Пайплайн пользовательских фраз не пишет: он поднимает `JobError` с кодом, а
фразу по коду и языку собирает тот, кто показывает её человеку. Иначе русский
текст ошибки, записанный в Redis воркером, англоязычный пользователь увидел бы
как есть.
"""

from __future__ import annotations

from typing import Any

LANGS = ("ru", "en")
DEFAULT_LANG = "ru"

# Кому по умолчанию русский, пока язык не выбран руками. Telegram отдаёт язык
# клиента; из СНГ-клиентов русский интерфейс понятнее английского.
_RU_CLIENTS = {"ru", "uk", "be", "kk", "uz", "ky", "tg", "hy", "az"}

LANG_NAMES = {"ru": "🇷🇺 Русский", "en": "🇬🇧 English"}


def guess_lang(language_code: str | None) -> str:
    code = (language_code or "").lower().split("-")[0]
    if not code:
        return DEFAULT_LANG
    return "ru" if code in _RU_CLIENTS else "en"


def lang_of(user: Any) -> str:
    """Выбранный язык пользователя, а если не выбран — угаданный по клиенту."""
    if user is None:
        return DEFAULT_LANG
    chosen = getattr(user, "lang", None)
    if chosen in LANGS:
        return chosen
    return guess_lang(getattr(user, "language_code", None))


def normalize(lang: str | None) -> str:
    return lang if lang in LANGS else DEFAULT_LANG


def t(key: str, lang: str | None = None, **params: Any) -> str:
    """Текст по ключу. Нет перевода — русский; нет ключа — сам ключ, а не падение."""
    entry = TEXTS.get(key)
    if entry is None:
        return key
    text = entry.get(normalize(lang)) or entry[DEFAULT_LANG]
    return text.format(**params) if params else text


def plural(n: int, lang: str | None, forms_ru: tuple[str, str, str], forms_en: tuple[str, str]) -> str:
    if normalize(lang) == "en":
        return forms_en[0] if abs(n) == 1 else forms_en[1]
    a = abs(n) % 100
    if 11 <= a <= 14:
        return forms_ru[2]
    return {1: forms_ru[0], 2: forms_ru[1], 3: forms_ru[1], 4: forms_ru[1]}.get(a % 10, forms_ru[2])


def videos_word(n: int, lang: str | None) -> str:
    return plural(n, lang, ("ролик", "ролика", "роликов"), ("video", "videos"))


# Коды ошибок монтажа. Пайплайн знает только код, фразу подставляет интерфейс.
ERROR_CODES = (
    "source_lost", "edl_missing", "too_short", "too_long", "no_audio",
    "no_speech", "edl_invalid", "cut_broken", "crop_broken", "worker_lost", "failed",
)

STAGE_KEYS = ("probe", "asr", "clean", "direct", "broll", "cut", "audio", "crop", "render", "done")


TEXTS: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------ язык
    "lang.pick": {
        "ru": "Выберите язык интерфейса · Choose your language",
        "en": "Choose your language · Выберите язык интерфейса",
    },
    "lang.set": {"ru": "Язык интерфейса: русский", "en": "Interface language: English"},

    # ------------------------------------------------------------ старт и помощь
    "start.welcome": {
        "ru": (
            "Привет! Я — Reelmaker, бот-монтажёр 🎬\n\n"
            "Снимаете видео, где говорите на камеру о работе, экспертизе или своём дне? "
            "Присылайте сырой файл — дальше я сам:\n\n"
            "✂️ вырежу паузы, «эээ» и неловкие заминки\n"
            "🎥 оживлю картинку: смена планов, кадр 9:16, лицо в центре\n"
            "💬 поставлю трендовые субтитры с акцентом на главном слове\n"
            "🖼 подберу картинки по смыслу и добавлю звуковые акценты\n\n"
            "Через пару минут — готовый Reels, который сразу можно выкладывать. "
            "Без монтажёра, без CapCut, без вечера за таймлайном.\n\n"
            "📎 Видео от {min_s} с до {max_m} мин, подойдёт и горизонтальное\n"
            "🎨 Стиль и цвет субтитров — в приложении, кнопка ниже"
        ),
        "en": (
            "Hi! I'm Reelmaker, your video editing bot 🎬\n\n"
            "Recorded yourself talking about your work, expertise or day? "
            "Send me the raw file — I'll take it from there:\n\n"
            "✂️ cut the pauses, “umm”s and awkward stumbles\n"
            "🎥 make it dynamic: shot changes, 9:16 frame, face centered\n"
            "💬 add trending captions with the key word highlighted\n"
            "🖼 pick matching images and add sound accents\n\n"
            "A few minutes later you get a Reel that's ready to post. "
            "No editor, no CapCut, no evening lost on a timeline.\n\n"
            "📎 Videos from {min_s} s to {max_m} min, landscape works too\n"
            "🎨 Caption style and colour are in the app — button below"
        ),
    },
    "btn.open_app": {"ru": "Открыть приложение", "en": "Open the app"},
    "btn.menu": {"ru": "Монтаж", "en": "Editor"},
    "help.text": {
        "ru": (
            "Как это работает:\n\n"
            "1. Распознаю речь пословно.\n"
            "2. Решаю, что вырезать и что выделить.\n"
            "3. Режу паузы, кадрирую 9:16, рисую субтитры.\n\n"
            "Команды:\n"
            "/start — начало и выбор языка\n"
            "/language — сменить язык\n"
            "/preset — стиль субтитров\n"
            "/stats — что уже сделано\n\n"
            "Если ролик не получился — пришлите его ещё раз, "
            "а лучше напишите, что именно не так."
        ),
        "en": (
            "How it works:\n\n"
            "1. I transcribe the speech word by word.\n"
            "2. I decide what to cut and what to highlight.\n"
            "3. I cut the pauses, reframe to 9:16 and draw the captions.\n\n"
            "Commands:\n"
            "/start — welcome and language\n"
            "/language — change language\n"
            "/preset — caption style\n"
            "/stats — what's been done\n\n"
            "If a reel didn't turn out right, send it again — "
            "or better, tell us what went wrong."
        ),
    },
    "preset.current": {
        "ru": "Стиль субтитров. Сейчас: «{current}».\nДоступны: {names}",
        "en": "Caption style. Current: “{current}”.\nAvailable: {names}",
    },
    "preset.unknown": {"ru": "Такого стиля нет", "en": "No such style"},
    "preset.picked": {"ru": "Стиль: {title}", "en": "Style: {title}"},
    "stats.empty": {"ru": "Пока ничего не смонтировано. Пришлите видео.", "en": "Nothing edited yet. Send me a video."},
    "stats.done": {"ru": "Готовых роликов: {n}", "en": "Reels done: {n}"},
    "stats.style": {"ru": "Стиль: {preset}", "en": "Style: {preset}"},
    "stats.balance": {"ru": "На балансе: {n}", "en": "Balance: {n}"},
    "stats.cost_head": {"ru": "\nРасход на модель за 7 дней:", "en": "\nModel spend, last 7 days:"},
    "stats.cost_total": {"ru": "  Итого: ${total:.2f}", "en": "  Total: ${total:.2f}"},
    "stats.cost_none": {"ru": "\nВызовов модели пока не было.", "en": "\nNo model calls yet."},

    "fallback.hint": {
        "ru": "Пришлите видео, где вы говорите на камеру, — смонтирую ролик. Стиль и настройки — в приложении.",
        "en": "Send me a video of you talking on camera and I'll edit it into a reel. Style and settings are in the app.",
    },
    "fallback.keyboard_removed": {
        "ru": "Кнопки обновились — теперь всё в меню «Монтаж» и в приложении.",
        "en": "The buttons have changed — everything is now in the “Editor” menu and the app.",
    },

    # ------------------------------------------------------------ приём видео
    "video.not_video": {"ru": "Это не видео. Пришлите видеофайл.", "en": "That's not a video. Please send a video file."},
    "video.one_in_flight": {
        "ru": "Один ролик уже в работе. Дождитесь его — и присылайте следующий.",
        "en": "One reel is already in progress. Wait for it, then send the next one.",
    },
    "video.queue_full": {"ru": "Очередь забита. Попробуйте через несколько минут.", "en": "The queue is full. Try again in a few minutes."},
    "video.too_long": {"ru": "Видео длиннее {max_m} мин. Пришлите кусок покороче.", "en": "The video is longer than {max_m} min. Send a shorter clip."},
    "video.no_balance": {
        "ru": "На балансе нет роликов. Пополнить можно в приложении — кнопка ниже.",
        "en": "Your balance is empty. You can top it up in the app — button below.",
    },
    "video.accepted": {"ru": "Принял. Ставлю в очередь…", "en": "Got it. Adding to the queue…"},
    "video.accepted_app": {"ru": "Принял видео из приложения. Ставлю в очередь…", "en": "Got your video from the app. Adding to the queue…"},
    "video.too_big": {
        "ru": "Файл больше {mb} МБ — такие Telegram боту не отдаёт. Загрузите видео через приложение (кнопка ниже) или пришлите его как обычное видео, со сжатием.",
        "en": "The file is over {mb} MB, and Telegram won't hand files that big to a bot. Upload it in the app (button below) or send it as a regular, compressed video.",
    },
    "video.download_failed": {
        "ru": "Не смог скачать файл из Telegram. Попробуйте прислать ещё раз.",
        "en": "Couldn't download the file from Telegram. Please try sending it again.",
    },
    "video.done": {"ru": "Готово.", "en": "Done."},
    "video.result_lost": {"ru": "Ролик готов, но файл потерялся. Напишите нам.", "en": "The reel is ready, but the file got lost. Please contact us."},
    "video.not_sent": {"ru": "Ролик готов, но не отправился: {error}", "en": "The reel is ready but couldn't be sent: {error}"},
    "video.caption_durations": {"ru": "{src} с → {res} с", "en": "{src} s → {res} s"},
    "video.caption_saved": {"ru": "вырезано {n} с пауз", "en": "{n} s of pauses cut"},
    "video.caption_style": {"ru": "стиль: {preset}", "en": "style: {preset}"},
    "video.rebuild_button": {"ru": "Пересобрать в «{title}»", "en": "Rebuild in “{title}”"},
    "rerender.no_edl": {"ru": "Исходный монтаж не сохранился", "en": "The original edit wasn't saved"},
    "rerender.purged": {
        "ru": "Исходник удалён по сроку хранения — пересобрать уже нельзя. Пришлите видео заново.",
        "en": "The source was removed after the storage period, so it can't be rebuilt. Please send the video again.",
    },
    "rerender.toast": {"ru": "Пересобираю — без повторного распознавания", "en": "Rebuilding — no need to transcribe again"},
    "rerender.status": {"ru": "Пересобираю в «{preset}»…", "en": "Rebuilding in “{preset}”…"},

    # ------------------------------------------------------------ стадии монтажа
    "stage.probe": {"ru": "Читаю файл", "en": "Reading the file"},
    "stage.asr": {"ru": "Распознаю речь", "en": "Transcribing speech"},
    "stage.clean": {"ru": "Проверяю расшифровку", "en": "Checking the transcript"},
    "stage.direct": {"ru": "Собираю монтаж", "en": "Planning the edit"},
    "stage.broll": {"ru": "Подбираю картинки", "en": "Picking images"},
    "stage.cut": {"ru": "Режу паузы", "en": "Cutting pauses"},
    "stage.audio": {"ru": "Свожу звук", "en": "Mixing audio"},
    "stage.crop": {"ru": "Кадрирую 9:16", "en": "Reframing to 9:16"},
    "stage.render": {"ru": "Рисую субтитры", "en": "Drawing captions"},
    "stage.done": {"ru": "Готово", "en": "Done"},
    "stage.download": {"ru": "Скачиваю файл", "en": "Downloading the file"},
    "stage.upload": {"ru": "Загружаю результат", "en": "Uploading the result"},
    "stage.queued": {"ru": "ждёт своей очереди", "en": "waiting in the queue"},

    # ------------------------------------------------------------ ошибки монтажа
    "err.source_lost": {"ru": "Исходный файл потерялся. Пришлите видео ещё раз.", "en": "The source file got lost. Please send the video again."},
    "err.edl_missing": {"ru": "Монтажный лист не сохранился — пересобрать нечем.", "en": "The edit wasn't saved, so there's nothing to rebuild."},
    "err.too_short": {"ru": "Видео короче {min_s} с — монтировать нечего.", "en": "The video is shorter than {min_s} s — nothing to edit."},
    "err.too_long": {"ru": "Видео длиннее {max_m} мин. Пришлите кусок покороче.", "en": "The video is longer than {max_m} min. Send a shorter clip."},
    "err.no_audio": {"ru": "В видео нет звуковой дорожки — субтитры делать не из чего.", "en": "The video has no audio track, so there's nothing to caption."},
    "err.no_speech": {"ru": "Речь не распознана. Возможно, в ролике только музыка или шум.", "en": "No speech detected. The video may contain only music or noise."},
    "err.edl_invalid": {"ru": "Не удалось собрать монтаж. Попробуйте прислать видео ещё раз.", "en": "Couldn't put the edit together. Please try sending the video again."},
    "err.cut_broken": {"ru": "Нарезка оборвалась — файл повреждён. Пришлите видео ещё раз.", "en": "Cutting was interrupted and the file is damaged. Please send the video again."},
    "err.crop_broken": {"ru": "Кадрирование оборвалось — файл повреждён. Пришлите видео ещё раз.", "en": "Reframing was interrupted and the file is damaged. Please send the video again."},
    "err.worker_lost": {"ru": "Монтаж прервался на нашей стороне — ролик вернулся на баланс. Пришлите видео ещё раз.", "en": "The edit was interrupted on our side — the reel is back on your balance. Please send the video again."},
    "err.failed": {"ru": "Не получилось смонтировать этот ролик. Мы записали ошибку.", "en": "We couldn't edit this video. The error has been logged."},

    # ------------------------------------------------------------ оплата
    "pay.bad_payload": {"ru": "Счёт не распознан. Откройте приложение и выберите пакет заново.", "en": "Invoice not recognised. Open the app and pick a package again."},
    "pay.stars_only": {"ru": "Оплата принимается только звёздами.", "en": "Only Telegram Stars are accepted."},
    "pay.price_changed": {"ru": "Цена изменилась. Откройте приложение и выберите пакет заново.", "en": "The price has changed. Open the app and pick a package again."},
    "pay.unrecognized": {"ru": "Оплата прошла, но пакет не распознан. Напишите нам — зачислим руками.", "en": "Payment went through but the package wasn't recognised. Contact us and we'll credit it manually."},
    "pay.credited": {"ru": "Зачислено: {videos} {word}. На балансе: {balance}.", "en": "Added {videos} {word}. Balance: {balance}."},
    "pay.invoice_title": {"ru": "{videos} {word}", "en": "{videos} {word}"},
    "pay.invoice_description": {
        "ru": "Монтаж вертикальных роликов: субтитры, нарезка пауз, кадр 9:16.",
        "en": "Vertical reel editing: captions, pause cutting, 9:16 framing.",
    },
    "pay.invoice_label": {"ru": "{videos} шт.", "en": "{videos} pcs"},

    # ------------------------------------------------------------ приложение (ответы API)
    "api.empty_file": {"ru": "Пустой файл.", "en": "The file is empty."},
    "api.too_big": {"ru": "Файл больше {mb} МБ.", "en": "The file is larger than {mb} MB."},
    "api.upload_not_found": {"ru": "Загрузка не найдена — начните заново.", "en": "Upload not found — please start again."},
    "api.bad_offset": {"ru": "Некорректный сдвиг загрузки.", "en": "Invalid upload offset."},
    "api.chunk_too_far": {"ru": "Кусок выходит за размер файла.", "en": "The chunk goes past the end of the file."},
    "api.offset_mismatch": {"ru": "Сдвиг не совпал.", "en": "Offset mismatch."},
    "api.incomplete": {"ru": "Файл загружен не полностью.", "en": "The file hasn't been fully uploaded."},
    "api.unknown_style": {"ru": "Такого стиля нет.", "en": "No such style."},
    "api.blocked": {"ru": "Доступ закрыт.", "en": "Access denied."},
    "api.unreadable": {"ru": "Не получилось прочитать видео. Пришлите mp4 или mov.", "en": "Couldn't read the video. Please upload an mp4 or mov."},
    "api.no_balance": {"ru": "На балансе нет роликов.", "en": "Your balance is empty."},
    "api.enqueue_failed": {"ru": "Не получилось поставить ролик в очередь. Баланс не списан.", "en": "Couldn't queue the video. Your balance was not charged."},
    "api.video_not_found": {"ru": "Ролик не найден.", "en": "Video not found."},
    "api.not_ready": {"ru": "Ролик ещё не готов.", "en": "The video isn't ready yet."},
    "api.purged": {
        "ru": "Ролик удалён с сервера по сроку хранения. В чате он остался.",
        "en": "The video was removed from the server after the storage period. It's still in your chat.",
    },
    "api.resend_failed": {
        "ru": "Не получилось отправить в чат. Напишите боту /start и попробуйте ещё раз.",
        "en": "Couldn't send it to the chat. Send /start to the bot and try again.",
    },
    "api.no_package": {"ru": "Не выбран пакет.", "en": "No package selected."},
    "api.unknown_package": {"ru": "Такого пакета нет.", "en": "No such package."},
    "api.link_expired": {"ru": "Ссылка устарела.", "en": "The link has expired."},
    "api.font_not_found": {"ru": "Шрифт не найден.", "en": "Font not found."},
}


def stage_title(key: str | None, lang: str | None) -> str | None:
    if not key:
        return None
    return t(f"stage.{key}", lang) if f"stage.{key}" in TEXTS else None


def error_text(code: str | None, lang: str | None, *, min_s: float, max_m: float) -> str:
    code = code if code in ERROR_CODES else "failed"
    return t(f"err.{code}", lang, min_s=f"{min_s:.0f}", max_m=f"{max_m:.0f}")
