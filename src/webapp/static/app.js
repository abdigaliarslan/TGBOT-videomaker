/* Mini App. Без сборки и фреймворка: шесть экранов, история переходов, API на том же хосте. */
"use strict";

const tg = window.Telegram && window.Telegram.WebApp;
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// Тексты интерфейса. Язык приходит с сервера (`user.lang`) и меняется в шапке и в профиле.
const I18N = {
  ru: {
    theme: "Сменить тему", back: "Назад", forward: "Вперёд", close: "Закрыть",
    "nav.home": "Главная", "nav.invite": "Приглашения", "nav.edit": "Монтаж", "nav.videos": "Результаты",
    "nav.tariffs": "Тарифы", "nav.profile": "Профиль", "nav.collapse": "Свернуть панель",
    "title.invite": "Приглашайте друзей", "step": "Шаг {n} из 2",
    "home.new": "Новый ролик", "home.new_sub": "Загрузите видео — смонтируем за пару минут",
    "home.recent": "Последнее", "home.all": "Все", "home.invite": "Пригласите друга",
    "home.invite_sub": "{p}% с каждой его покупки — вам на баланс",
    "home.empty": "Роликов пока нет", "home.empty_sub": "Первый ролик появится здесь",
    "home.offer_trial": "Первый ролик — бесплатно. Дальше {single} ⭐ за ролик",
    "home.offer_paid": "{single} ⭐ за ролик, пакетом — от {best} ⭐",
    "home.offer_free": "Сейчас монтаж бесплатный",
    "preview.pause": "Пауза", "preview.play": "Воспроизвести",
    "preview.s1": ["это", "пример", "субтитров", "в", "стиле"],
    "preview.s2": ["паузы", "уже", "вырезаны"],
    "preview.s3": ["и", "кадр", "по центру", "лица"],
    "edit.s1_title": "Загрузите видео для ролика",
    "edit.s1_lead": "Видео, где человек говорит в камеру. Соберём из него вертикальный ролик с субтитрами.",
    "edit.limits": "до {mb} МБ · от {min} с до {max} мин",
    "edit.style": "Стиль субтитров", "edit.font": "Шрифт", "edit.color": "Цвет выделения", "edit.effects": "Эффекты",
    "edit.requirements": "Требования к видео", "edit.in_chat": "Отправить в чат с ботом",
    "edit.untitled": "Видео", "edit.remove": "Убрать видео",
    "edit.next": "Далее: оформление", "edit.next_idle": "Сначала добавьте видео", "edit.next_ready": "стиль, шрифт и цвет субтитров",
    "edit.back_to_video": "← Вернуться к видео",
    "opt.cut": "Нарезка", "cut.on": "Вырезать паузы", "cut.off": "Без нарезки",
    "cut.on_hint": "Уберём затяжные паузы и долгие «эээ» — ролик станет плотнее.",
    "cut.off_hint": "Видео пойдёт целиком, как снято: только кадр 9:16 и субтитры.",
    "opt.sfx": "Звуковые эффекты", "opt.broll": "Медиа-карточки", "opt.camera": "Движение камеры",
    "help.cut": ["Нарезка", "Вырезаем затяжные паузы дольше 1.2 секунды и долгие «эээ». Обычные вдохи и слова-паразиты остаются — так речь звучит естественно."],
    "help.sfx": ["Звуковые эффекты", "Тихий щелчок на каждом субтитре, «whoosh» на появлении медиа-карточки и при смене крупности. Громкость голоса выравнивается в любом случае."],
    "help.broll": ["Медиа-карточки", "Картинки и короткие клипы по смыслу сказанного — не чаще одной на 8 секунд и никогда поверх субтитров."],
    "help.camera": ["Движение камеры", "На резах кадр чередует общий и крупный план, чтобы склейки читались как монтаж. Выключите — весь ролик пойдёт одним планом."],
    "req.title": "Требования к видео",
    "req.duration": "Длительность от {min} с до {max} мин.", "req.size": "Размер до {mb} МБ.",
    "req.speaker": "В кадре — говорящий человек, звук с речью.",
    "req.orientation": "Подойдёт и вертикальное, и горизонтальное: кадр 9:16 соберём сами.",
    "file.too_long": "Видео длиннее {max} мин — сервер его не примет.",
    "file.too_short": "Видео короче {min} с — монтировать нечего.",
    "size.mb": "МБ", "size.gb": "ГБ", "dur.s": "с",
    "cta.edit": "Смонтировать", "cta.choose": "Выбрать видео",
    "layout.single": "Одно видео", "layout.split": "Два видео",
    "split.top_title": "Сверху — что показываем", "split.top_sub": "без звука",
    "split.bottom_title": "Снизу — вы в кадре", "split.bottom_sub": "звук и субтитры отсюда",
    "split.need_both": "добавьте оба видео", "split.top_role": "Сверху", "split.bottom_role": "Снизу",
    "edit.s1_lead_split": "Сверху — то, что показываете, снизу — вы в кадре. Соберём из двух видео ролик с субтитрами.",
    "cta.topup": "Пополнить баланс", "cta.missing": "не хватает 1 ролика",
    "cta.free": "бесплатно для владельца", "cta.free_all": "сейчас монтаж бесплатный", "cta.pick": "сначала выберите видео", "cta.charge": "спишется 1 ролик из {n}",
    "cta.uploading": "Загрузка {p}%", "cta.checking": "Проверяю видео…", "cta.keep_open": "не закрывайте приложение до конца загрузки",
    "upload.lost": "Связь оборвалась во время загрузки. Попробуйте ещё раз.",
    "toast.accepted": "Видео принято. Готовый ролик придёт в чат, а прогресс — в «Результатах».",
    "err.title": "Не получилось",
    "balance.unlimited": "без ограничений",
    "balance.note": "Один ролик — один смонтированный видеофайл. Если монтаж не получится, ролик вернётся на баланс.",
    "balance.note_admin": "Вы владелец бота: монтаж бесплатный, баланс не списывается.",
    "balance.note_free": "Сейчас монтаж бесплатный: ролики с баланса не списываются.",
    "status.queued": "в очереди", "status.running": "монтируется", "status.done": "готов", "status.failed": "не получилось", "status.cancelled": "отменён",
    "jobs.all": "Все", "jobs.done": "Готовые", "jobs.active": "В работе",
    "jobs.empty": "Роликов пока нет", "jobs.empty_hint": "Загрузите видео — готовый ролик придёт в чат и появится здесь.",
    "jobs.empty_filter": "Здесь пока пусто",
    "jobs.to_edit": "К монтажу", "jobs.reel": "Ролик", "jobs.refunded": "Ролик вернулся на баланс.",
    "jobs.purged": "Удалён с сервера по сроку хранения. В чате ролик остался.",
    "jobs.retention": "Готовые ролики хранятся на сервере {n} дн., в чате — всегда.",
    "jobs.send": "Прислать в чат", "jobs.sent": "Отправил в чат.", "jobs.not_sent": "Не отправилось",
    "tariffs.balance": "Баланс", "tariffs.history": "История", "tariffs.package": "Пакет", "tariffs.method": "Способ оплаты",
    "pkg.per": "{n} ⭐ за ролик", "pkg.try": "попробовать", "pkg.popular": "популярный", "pkg.save": "выгода −{n}%",
    "pay.button": "Оплатить {n} ⭐", "pay.sub": "{n} {w} на баланс",
    "pay.title": "Оплата", "pay.update": "Обновите Telegram — эта версия не умеет оплату звёздами.",
    "pay.failed": "Оплата не прошла", "pay.failed_text": "Звёзды не списаны. Попробуйте ещё раз.",
    "pay.crediting": "Оплачено. Зачисляю…", "pay.credited": "Зачислено: {n} {w}.", "pay.later": "Оплата прошла, баланс обновится через минуту.",
    "history.title": "История", "history.empty": "Оплат пока не было",
    "history.payment": "Покупка пакета", "history.bonus": "Бонус за друга",
    "invite.lead": "За каждую оплату друга вам возвращается {p}% от купленных им роликов — прямо на баланс.",
    "invite.your_link": "Ваша ссылка", "invite.copy": "Скопировать", "invite.share": "Поделиться",
    "invite.copied": "Ссылка скопирована",
    "invite.step1": "Друг открывает бота по вашей ссылке", "invite.step2": "Покупает пакет роликов",
    "invite.step3": "Вам на баланс — {p}% от купленного",
    "invite.all_time": "За всё время", "invite.came": "Пришли", "invite.paid": "Оплатили", "invite.earned": "Роликов",
    "invite.today": "+{n} сегодня",
    "invite.credit": "Накоплено {n} ролика — на баланс уйдёт целым роликом, как только наберётся.",
    "invite.friends": "Приглашённые", "invite.summary": "{a} пришли · {b} оплатили · +{c} роликов",
    "invite.none": "Пока никого", "invite.none_sub": "Поделитесь ссылкой — приглашённые появятся здесь.",
    "invite.friend_paid": "оплатил", "invite.friend_waiting": "ещё не покупал", "invite.noname": "Без имени",
    "period.7": "7 дней", "period.30": "30 дней", "period.0": "Всё время",
    "invite.text": "Монтирую вертикальные ролики за пару минут: субтитры, паузы, кадр 9:16.",
    "profile.noname": "Без имени", "profile.done": "смонтировано", "profile.balance": "на балансе",
    "profile.language": "Язык", "profile.theme": "Тема", "profile.chat": "Чат с ботом",
    "profile.hint_admin": "Вы в списке владельцев: монтаж без списаний. Чтобы проверить оплату как обычный пользователь, уберите свой id из ADMIN_IDS.",
    "profile.hint": "Видео можно присылать и прямо в чат с ботом — стиль возьмётся последний выбранный здесь.",
    "blocker.title": "Откройте из Telegram", "blocker.text": "Приложение работает внутри бота: нажмите кнопку «Монтаж» в чате.",
    "blocker.failed": "Не получилось открыть",
  },
  en: {
    theme: "Toggle theme", back: "Back", forward: "Forward", close: "Close",
    "nav.home": "Home", "nav.invite": "Invites", "nav.edit": "Editor", "nav.videos": "Results",
    "nav.tariffs": "Pricing", "nav.profile": "Profile", "nav.collapse": "Collapse sidebar",
    "title.invite": "Invite friends", "step": "Step {n} of 2",
    "home.new": "New reel", "home.new_sub": "Upload a video — we'll edit it in a couple of minutes",
    "home.recent": "Recent", "home.all": "All", "home.invite": "Invite a friend",
    "home.invite_sub": "{p}% of every purchase they make goes to your balance",
    "home.empty": "No videos yet", "home.empty_sub": "Your first reel will show up here",
    "home.offer_trial": "Your first reel is free. After that it's {single} ⭐ per reel",
    "home.offer_paid": "{single} ⭐ per reel, or from {best} ⭐ in a pack",
    "home.offer_free": "Editing is free right now",
    "preview.pause": "Pause", "preview.play": "Play",
    "preview.s1": ["this", "is", "captions", "in", "style"],
    "preview.s2": ["pauses", "are", "gone"],
    "preview.s3": ["and", "the frame", "follows", "the face"],
    "edit.s1_title": "Upload a video for your reel",
    "edit.s1_lead": "A video of someone talking to the camera. We'll turn it into a vertical reel with captions.",
    "edit.limits": "up to {mb} MB · {min} s to {max} min",
    "edit.style": "Caption style", "edit.font": "Typeface", "edit.color": "Highlight colour", "edit.effects": "Effects",
    "edit.requirements": "Video requirements", "edit.in_chat": "Send it in the bot chat",
    "edit.untitled": "Video", "edit.remove": "Remove video",
    "edit.next": "Next: design", "edit.next_idle": "Add a video first", "edit.next_ready": "caption style, typeface and colour",
    "edit.back_to_video": "← Back to the video",
    "opt.cut": "Cutting", "cut.on": "Cut pauses", "cut.off": "No cutting",
    "cut.on_hint": "We'll remove long pauses and drawn-out “umm”s — the reel gets tighter.",
    "cut.off_hint": "The video stays whole, as shot: just the 9:16 frame and captions.",
    "opt.sfx": "Sound effects", "opt.broll": "Media cards", "opt.camera": "Camera movement",
    "help.cut": ["Cutting", "We remove pauses longer than 1.2 seconds and long “umm”s. Normal breaths and filler words stay, so the speech sounds natural."],
    "help.sfx": ["Sound effects", "A quiet tick on every caption, a “whoosh” when a media card appears and when the shot size changes. Voice loudness is levelled either way."],
    "help.broll": ["Media cards", "Images and short clips that match what's being said — at most one per 8 seconds and never over the captions."],
    "help.camera": ["Camera movement", "At cuts the frame alternates between wide and close-up so the edits read as editing. Turn it off to keep one shot throughout."],
    "req.title": "Video requirements",
    "req.duration": "Length from {min} s to {max} min.", "req.size": "Up to {mb} MB.",
    "req.speaker": "A person talking on camera, with audible speech.",
    "req.orientation": "Portrait or landscape — we'll build the 9:16 frame ourselves.",
    "file.too_long": "The video is longer than {max} min — the server won't accept it.",
    "file.too_short": "The video is shorter than {min} s — nothing to edit.",
    "size.mb": "MB", "size.gb": "GB", "dur.s": "s",
    "cta.edit": "Create reel", "cta.choose": "Choose a video",
    "layout.single": "One video", "layout.split": "Two videos",
    "split.top_title": "Top — what you show", "split.top_sub": "muted",
    "split.bottom_title": "Bottom — you on camera", "split.bottom_sub": "sound and captions come from here",
    "split.need_both": "add both videos", "split.top_role": "Top", "split.bottom_role": "Bottom",
    "edit.s1_lead_split": "On top — what you show, below — you on camera. We'll turn the two videos into a reel with captions.",
    "cta.topup": "Top up balance", "cta.missing": "you need 1 more video",
    "cta.free": "free for the owner", "cta.free_all": "editing is free right now", "cta.pick": "pick a video first", "cta.charge": "uses 1 of your {n} videos",
    "cta.uploading": "Uploading {p}%", "cta.checking": "Checking the video…", "cta.keep_open": "keep the app open until the upload finishes",
    "upload.lost": "The connection dropped during upload. Please try again.",
    "toast.accepted": "Video received. The finished reel will arrive in the chat; track progress in “Results”.",
    "err.title": "Something went wrong",
    "balance.unlimited": "unlimited",
    "balance.note": "One video — one edited reel. If editing fails, the video goes back to your balance.",
    "balance.note_admin": "You own this bot: editing is free and your balance isn't used.",
    "balance.note_free": "Editing is free right now: your balance isn't used.",
    "status.queued": "queued", "status.running": "editing", "status.done": "ready", "status.failed": "failed", "status.cancelled": "cancelled",
    "jobs.all": "All", "jobs.done": "Ready", "jobs.active": "In progress",
    "jobs.empty": "No videos yet", "jobs.empty_hint": "Upload a video — the finished reel will arrive in the chat and show up here.",
    "jobs.empty_filter": "Nothing here yet",
    "jobs.to_edit": "Go to editor", "jobs.reel": "Reel", "jobs.refunded": "The video was returned to your balance.",
    "jobs.purged": "Removed from the server after the storage period. It's still in your chat.",
    "jobs.retention": "Finished videos stay on the server for {n} days and in your chat forever.",
    "jobs.send": "Send to chat", "jobs.sent": "Sent to the chat.", "jobs.not_sent": "Couldn't send",
    "tariffs.balance": "Balance", "tariffs.history": "History", "tariffs.package": "Package", "tariffs.method": "Payment method",
    "pkg.per": "{n} ⭐ per video", "pkg.try": "try it out", "pkg.popular": "popular", "pkg.save": "save {n}%",
    "pay.button": "Pay {n} ⭐", "pay.sub": "{n} {w} to your balance",
    "pay.title": "Payment", "pay.update": "Please update Telegram — this version can't pay with Stars.",
    "pay.failed": "Payment failed", "pay.failed_text": "No Stars were charged. Please try again.",
    "pay.crediting": "Paid. Adding to your balance…", "pay.credited": "Added: {n} {w}.", "pay.later": "Payment received; your balance will update in a minute.",
    "history.title": "History", "history.empty": "No payments yet",
    "history.payment": "Package purchase", "history.bonus": "Friend bonus",
    "invite.lead": "Every time a friend pays, {p}% of the videos they bought comes back to you — straight to your balance.",
    "invite.your_link": "Your link", "invite.copy": "Copy", "invite.share": "Share",
    "invite.copied": "Link copied",
    "invite.step1": "A friend opens the bot with your link", "invite.step2": "They buy a video package",
    "invite.step3": "You get {p}% of it on your balance",
    "invite.all_time": "All time", "invite.came": "Joined", "invite.paid": "Paid", "invite.earned": "Videos",
    "invite.today": "+{n} today",
    "invite.credit": "{n} of a video saved up — it moves to your balance as a whole video once it adds up.",
    "invite.friends": "Invited", "invite.summary": "{a} joined · {b} paid · +{c} videos",
    "invite.none": "Nobody yet", "invite.none_sub": "Share your link — invited friends will show up here.",
    "invite.friend_paid": "paid", "invite.friend_waiting": "hasn't bought yet", "invite.noname": "No name",
    "period.7": "7 days", "period.30": "30 days", "period.0": "All time",
    "invite.text": "I turn talking-head videos into vertical reels in a couple of minutes: captions, pauses cut, 9:16 frame.",
    "profile.noname": "No name", "profile.done": "edited", "profile.balance": "balance",
    "profile.language": "Language", "profile.theme": "Theme", "profile.chat": "Bot chat",
    "profile.hint_admin": "You're on the owners list: editing is free. To test payments as a regular user, remove your id from ADMIN_IDS.",
    "profile.hint": "You can also send videos straight to the bot chat — it will use the style you last picked here.",
    "blocker.title": "Open from Telegram", "blocker.text": "This app runs inside the bot: tap the “Editor” button in the chat.",
    "blocker.failed": "Couldn't open the app",
  },
};

const SCREENS = ["home", "invite", "edit", "videos", "tariffs", "profile"];
// Старые имена вкладок: бот присылает кнопки с `?screen=balance`.
const SCREEN_ALIASES = { balance: "tariffs" };

function currentLang() {
  if (state.boot && state.boot.user && I18N[state.boot.user.lang]) return state.boot.user.lang;
  const code = ((tg && tg.initDataUnsafe && tg.initDataUnsafe.user && tg.initDataUnsafe.user.language_code) || navigator.language || "").toLowerCase();
  return ["ru", "uk", "be", "kk", "uz", "ky", "tg", "hy", "az"].includes(code.split("-")[0]) ? "ru" : "en";
}

function tr(key, params) {
  const lang = currentLang();
  let text = (I18N[lang] && I18N[lang][key]) ?? I18N.ru[key] ?? key;
  if (typeof text === "string" && params) {
    for (const [k, v] of Object.entries(params)) text = text.replaceAll(`{${k}}`, v);
  }
  return text;
}

function applyStaticTexts() {
  document.documentElement.lang = currentLang();
  for (const node of $$("[data-i18n]")) node.textContent = tr(node.dataset.i18n);
  for (const node of $$("[data-i18n-aria]")) node.setAttribute("aria-label", tr(node.dataset.i18nAria));
}

const state = {
  boot: null,
  preset: null,
  accent: "theme",
  font: "theme",
  options: {},
  file: null,
  // «Два видео» (`pipeline.split`): `file` — нижнее, говорящий; `topFile` —
  // верхнее, то, что показывают.
  layout: "single",
  topFile: null,
  busy: false,
  jobs: [],
  jobsLoaded: false,
  jobsFilter: "all",
  pollTimer: null,
  screen: null,
  history: [],
  step: 1,
  pkg: null,
  refDays: 30,
  refs: null,
};

// ------------------------------------------------------------------ сеть

async function api(path, { method = "GET", body, raw } = {}) {
  const headers = { Authorization: `tma ${tg ? tg.initData : ""}` };
  let payload = body;
  if (body !== undefined && !raw) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(path, { method, headers, body: payload });
  let data = {};
  try { data = await res.json(); } catch (_) { /* пустой ответ */ }
  if (!res.ok) {
    const err = new Error(data.error || `HTTP ${res.status}`);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

// ------------------------------------------------------------------ мелочи

function plural(n, one, few, many) {
  const a = Math.abs(n) % 100;
  if (a >= 11 && a <= 14) return many;
  return { 1: one, 2: few, 3: few, 4: few }[a % 10] || many;
}
const videosWord = (n) => (currentLang() === "en"
  ? (Math.abs(n) === 1 ? "video" : "videos")
  : plural(n, "ролик", "ролика", "роликов"));

// Сотые ролика → «0,5» / «0.5». Бонус за друга копится дробью: 10 % от пакета
// в 5 роликов — половина ролика.
function fmtShare(hundredths) {
  const n = (hundredths || 0) / 100;
  const text = Number.isInteger(n) ? String(n) : n.toFixed(2).replace(/0+$/, "");
  return currentLang() === "ru" ? text.replace(".", ",") : text;
}

function locale() { return currentLang() === "en" ? "en-GB" : "ru-RU"; }
function fmtWhen(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  const today = new Date();
  if (d.toDateString() === today.toDateString()) {
    return d.toLocaleTimeString(locale(), { hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleString(locale(), { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}
function fmtDate(iso) {
  return iso ? new Date(iso).toLocaleDateString(locale(), { day: "numeric", month: "short" }) : "";
}

function haptic(kind = "selection") {
  if (!tg || !tg.HapticFeedback) return;
  try {
    if (kind === "selection") tg.HapticFeedback.selectionChanged();
    else tg.HapticFeedback.notificationOccurred(kind);
  } catch (_) { /* старый клиент */ }
}

let toastTimer = null;
function toast(text) {
  const node = $("#toast");
  node.textContent = text;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 3200);
}

function popup(title, message) {
  if (tg && tg.showPopup) {
    try { tg.showPopup({ title, message, buttons: [{ type: "close" }] }); return; } catch (_) { /* ниже */ }
  }
  alert(`${title}\n\n${message}`);
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of [].concat(children)) if (c) node.append(c);
  return node;
}

// Значок из спрайта в index.html.
function icon(name, size = 20, cls) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("width", size);
  svg.setAttribute("height", size);
  if (cls) svg.setAttribute("class", cls);
  const use = document.createElementNS(ns, "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

// «#0B0F14» → «11, 15, 20»: плашка превью красится той же непрозрачностью, что
// в кадре, а значит цвет нужен разобранный.
function hexToRgb(hex) {
  let v = String(hex || "#000000").replace("#", "");
  if (v.length === 3) v = [...v].map((c) => c + c).join("");
  const n = parseInt(v, 16) || 0;
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255].join(", ");
}

function fmtSize(bytes) {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / 1024 ** 3).toFixed(1)} ${tr("size.gb")}`;
  return `${Math.max(1, Math.round(bytes / 1024 / 1024))} ${tr("size.mb")}`;
}
function fmtDuration(sec) {
  if (!isFinite(sec)) return "";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m ? `${m}:${String(s).padStart(2, "0")}` : `${s} ${tr("dur.s")}`;
}

function openTelegram(url) {
  if (tg && tg.openTelegramLink) tg.openTelegramLink(url);
  else window.open(url, "_blank");
}

// ------------------------------------------------------------------ тема

const THEME_KEY = "reel-theme";
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const bg = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim();
  if (tg) {
    try { tg.setHeaderColor(bg); tg.setBackgroundColor(bg); tg.setBottomBarColor && tg.setBottomBarColor(bg); } catch (_) { /* старый клиент */ }
  }
  renderThemeSeg();
}
function setTheme(theme) {
  try { localStorage.setItem(THEME_KEY, theme); } catch (_) { /* не критично */ }
  applyTheme(theme);
  haptic();
}
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch (_) { /* приватный режим */ }
  // Тёмная — основа: приложение про видео, и так его рисует образец. Светлая —
  // если человек выбрал её сам или так настроен его Telegram.
  const initial = saved || (tg && tg.colorScheme === "light" ? "light" : "dark");
  applyTheme(initial);
  for (const btn of $$("[data-theme-toggle]")) {
    btn.addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  }
}
function renderThemeSeg() {
  const box = $("#theme-seg");
  if (!box) return;
  const now = document.documentElement.dataset.theme;
  box.replaceChildren(...[["light", "sun"], ["dark", "moon"]].map(([key, ico]) =>
    el("button", {
      type: "button", class: key === now ? "is-active" : "", "aria-label": key,
      "aria-pressed": key === now ? "true" : "false",
      onclick: () => setTheme(key),
    }, [icon(ico, 16)])));
}

// ------------------------------------------------------------------ превью субтитров

// Шрифты превью объявляются по списку из /api/bootstrap: их несколько, и
// зашитый в css один означал бы, что чип выбора показывает чужой шрифт.
// Диапазон весов — тот же, что в рендере и в линейке замеров: один файл
// закрывает весь диапазон, и браузер не синтезирует жирность.
function declareFonts() {
  const faces = new Map();
  for (const f of state.boot.fonts || []) if (f.file && f.family) faces.set(f.file, f.family);
  for (const p of state.boot.presets || []) if (p.font_file && p.font) faces.set(p.font_file, p.font);
  // На каждое семейство ДВА объявления с одним именем и разными наборами
  // символов. Браузер выбирает файл по тому, что реально на экране:
  //
  //   лента («Аа» и латинские названия тем) → урезанный файл, 8-19 КБ;
  //   превью (русская фраза)                → полный, и только для выбранного.
  //
  // Это снимает главную причину, по которой лента выглядела одинаковой:
  // восемнадцать полных файлов — 1406 КБ в gzip и 25 секунд по туннелю
  // (замер), и всё это время образцы рисуются одним системным шрифтом.
  // Восемнадцать урезанных весят 280 КБ на всех.
  //
  // Урезанный объявлен ВТОРЫМ: при пересечении наборов побеждает последний,
  // и буквы ленты берутся из маленького файла, а не из полного.
  const range = state.boot.font_ui_range;
  const css = [...faces].flatMap(([file, family]) => {
    const face = (src, extra) => `@font-face{font-family:"${family}";` +
      `src:url("${src}") format("truetype");font-weight:1 1000;font-display:swap;${extra}}`;
    const full = face(`/fonts/${file}`, "");
    return range ? [full, face(`/fonts/ui/${file}`, `unicode-range:${range};`)] : [full];
  }).join("\n");
  const tag = document.getElementById("font-faces") || el("style", { id: "font-faces" });
  tag.textContent = css;
  if (!tag.parentNode) document.head.appendChild(tag);

  // Просим файлы СРАЗУ, не дожидаясь, пока их потребует текст на экране.
  //
  // `@font-face` грузится лениво: пока лента выбора не нарисована, ни один
  // файл не запрошен. А рисуется она на шаге 2 — и первые секунды все
  // восемнадцать образцов показываются одним системным шрифтом, то есть
  // выглядят одинаково (замерено: ширина «Аа» у всех 25.2 px, через две
  // секунды — от 20.7 до 37.4). Здесь же загрузка начинается на первом
  // экране и идёт, пока человек выбирает видео.
  //
  if (!document.fonts) return;
  if (range) {
    // Просим ПОИМЕННО урезанные начертания, а не семейства.
    //
    // `document.fonts.load("21px \"X\"")` грузит ВСЕ подходящие начертания
    // семейства, а полное подходит к любому тексту — замерено: скачивались и
    // 17 урезанных, и 17 полных, то есть 3.2 МБ вместо 280 КБ, вдвое хуже,
    // чем было. Поэтому идём по самим объявлениям и будим только те, у
    // которых задан диапазон: полные браузер дотянет сам, когда превью
    // наберёт русскую фразу, и только для выбранного шрифта.
    for (const face of document.fonts) {
      if (face.unicodeRange && face.unicodeRange !== "U+0-10FFFF" && face.status === "unloaded") {
        face.load().catch(() => { /* нет файла — останется запасной */ });
      }
    }
    return;
  }
  // Сервер урезанных начертаний не отдаёт (старее приложения — так бывает,
  // когда статика уехала, а Python ещё нет). Тогда будим полные: ждать их
  // по-прежнему долго, но лента хотя бы не стоит одинаковой, пока человек
  // выбирает видео.
  //
  // Эта ветка не запасная вежливость, а починка: пока условие стояло на
  // `range`, при старом сервере не грузилось НИЧЕГО, и ранняя загрузка —
  // уже выложенная — в бою не работала.
  for (const family of new Set(faces.values())) {
    document.fonts.load(`21px "${family}"`).catch(() => { /* нет файла — останется запасной */ });
  }
}

function activePreset() {
  return (state.boot.presets || []).find((x) => x.name === state.preset) || null;
}

function activeThemeFamily() {
  const p = activePreset();
  return (p && p.font) || "Montserrat ExtraBold";
}

// Шрифт, которым рисуется превью: выбранный, а иначе — шрифт темы.
function activeFont() {
  return (state.boot.fonts || []).find((f) => f.key === state.font && f.family) || null;
}
function activeFamily() {
  const chosen = activeFont();
  return (chosen && chosen.family) || activeThemeFamily();
}

// Цвет выделения: выбранный, а иначе — задуманный темой.
function activeAccent() {
  const chosen = (state.boot.accents || []).find((a) => a.key === state.accent);
  const p = activePreset();
  return (chosen && chosen.color) || (p ? p.colors.accent : "#F3D28F");
}

const preview = {
  timer: null,
  paused: false,
  step: 0,
  script() {
    const title = (activePreset() || {}).title || "Glass";
    const [a1, a2, big1, a3, a4] = tr("preview.s1");
    const [b1, b2, big2] = tr("preview.s2");
    const [c1, c2, big3, c3] = tr("preview.s3");
    return [
      { small: [a1, a2], big: big1, after: [a3, a4, title] },
      { small: [b1, b2], big: big2, after: [] },
      { small: [c1, c2], big: big3, after: [c3] },
    ];
  },
  // Кегль превью — не вкус, а пересчёт: пиксели кадра 1920 в пиксели карточки.
  // Поэтому мелкая тема (Widescreen, 46 px) здесь и выглядит мелкой, а крупная
  // (Heavyweight, 109) занимает полкадра — как и будет в ролике.
  //
  // Смена шрифта двигает кегль ровно так же, как `core.options.apply_font`:
  // по высоте прописных. Иначе превью обещало бы один размер, а ролик дал
  // другой — у Cormorant прописная 0.625 em против 0.800 у Prata.
  size(p) {
    const k = ($("#preview").clientHeight || 620) / 1920;
    const font = activeFont();
    const ratio = font && font.cap_em && p.cap_em ? p.cap_em / font.cap_em : 1;
    return { k, base: p.size * ratio * k };
  },
  place(box, p, k) {
    box.style.top = "auto";
    box.style.bottom = "auto";
    box.style.transform = "none";
    if (p.position === "bottom") box.style.bottom = `${Math.round(p.margin * k)}px`;
    else if (p.position === "top") box.style.top = `${Math.round(p.margin * k)}px`;
    else { box.style.top = "50%"; box.style.transform = "translateY(-50%)"; }
  },
  render() {
    const box = $("#preview-caption");
    const p = activePreset();
    if (!p) return;
    const accent = activeAccent();
    const tx = (w) => (p.uppercase ? w.toUpperCase() : w);
    const scenes = this.script();
    const scene = scenes[Math.floor(this.step / 10) % scenes.length];
    const shown = this.step % 10; // 0..9: слова загораются по одному, потом блок гаснет
    const { k, base } = this.size(p);

    box.replaceChildren();
    box.style.color = p.colors.text;
    box.style.fontSize = `${base}px`;
    box.style.lineHeight = p.line_height;
    box.style.letterSpacing = `${p.letter_spacing}em`;
    box.style.setProperty("--cap-font", `"${activeFamily()}"`);
    this.place(box, p, k);
    // Движение берётся у темы, а не зашито в стили: иначе превью обещает одно,
    // а рендер делает другое — ровно так и было, пока слово в ролике включалось
    // за кадр, а здесь плавно проявлялось.
    const motion = p.motion || {};
    box.style.setProperty("--cap-in", `${motion.word_in_ms || 0}ms`);
    box.style.setProperty("--cap-rise", `${motion.word_rise_em || 0}em`);

    let i = 0;
    const word = (w, size, color) => {
      const on = shown >= 1 && i < shown && shown < 9;
      i += 1;
      return el("span", {
        class: `cap-word${on ? " on" : ""}`,
        text: tx(w),
        style: `font-size:${size}px${color ? `;color:${color}` : ""}`,
      });
    };
    // Плашка — та же, что у блока в кадре: цвет, непрозрачность, радиус и
    // паддинги темы. У тем, где она приходит только над светлым фоном
    // (`opacity: 0`), в превью её и нет — фон карточки тёмный.
    const plate = el("div", { class: "cap-plate" });
    const panel = p.panel || {};
    let plateX = 0;
    if (panel.enabled && panel.opacity > 0) {
      plateX = Math.round(panel.padding_x * k);
      plate.style.background = `rgba(${hexToRgb(panel.color)}, ${panel.opacity})`;
      plate.style.padding = `${Math.round(panel.padding_y * k)}px ${plateX}px`;
      plate.style.borderRadius = `${Math.round(panel.radius * k)}px`;
    }
    plate.append(el("div", { class: "cap-line" }, scene.small.map((w) => word(w, base))));
    plate.append(el("div", { class: "cap-line" }, [word(scene.big, base * (p.accent_scale || 1), accent)]));
    if (scene.after.length) {
      plate.append(el("div", { class: "cap-line" }, scene.after.map((w) => word(w, base))));
    }
    box.append(plate);

    // Строка ужимается под ширину превью, как в рендере (С1): слово «СУБТИТРОВ»
    // капсом в Bold иначе вылезает за края карточки.
    //
    // Паддинг плашки вычитается ЯВНО. Раньше тут стояло
    // `plate.offsetWidth - plate.clientWidth`, но clientWidth паддинг уже
    // включает: разность даёт рамку, а её у плашки нет — то есть вычиталось
    // всегда ноль. На единственной теме с видимой плашкой (Notebook) строка,
    // которая шире её внутренней коробки, проверку проходила и вылезала
    // словами за подложку.
    const room = box.clientWidth - 2 * plateX;
    for (const line of plate.children) {
      const over = line.scrollWidth / room;
      if (room > 0 && over > 1) {
        for (const w of line.children) {
          const size = parseFloat(getComputedStyle(w).fontSize);
          w.style.fontSize = `${Math.floor(size / over)}px`;
        }
      }
    }
    const font = activeFont();
    $("#preview-tag").textContent = `${p.label || p.title} · ${font ? font.title : p.font}`;
  },
  tick() {
    this.step += 1;
    this.render();
  },
  start() {
    clearInterval(this.timer);
    // При сокращённом движении кадр собирается один раз и стоит: мигание слов
    // раз в 280 мс — ровно то, от чего эта настройка защищает.
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      this.step = 5;
      this.render();
      return;
    }
    this.timer = setInterval(() => { if (!this.paused) this.tick(); }, 280);
    this.render();
  },
  stop() {
    clearInterval(this.timer);
    this.timer = null;
  },
  toggle() {
    this.paused = !this.paused;
    $("#preview").classList.toggle("is-paused", this.paused);
    $("#preview-play").setAttribute("aria-label", tr(this.paused ? "preview.play" : "preview.pause"));
  },
};

// Превью живёт только на втором шаге монтажа. У спрятанного `display:none`
// высота равна нулю, масштаб уходил на запасные 620, и на возврате кадр на
// четверть секунды показывал субтитр в полтора раза крупнее, с неужатыми
// строками. Заодно это 3.6 перерисовки в секунду, которые шли бы вечно.
function syncPreview() {
  if (state.screen === "edit" && state.step === 2 && !document.hidden) preview.start();
  else preview.stop();
}

// ------------------------------------------------------------------ экран «Монтаж», шаг 2

// Лента перерисовывается целиком, поэтому к активному элементу прокручиваемся
// уже ПОСЛЕ замены: старый узел к этому моменту не в документе.
//
// Двигаем САМУ ленту, а не зовём scrollIntoView: тот бы потянул и страницу по
// вертикали, а на открытии приложения это прыжок экрана ещё до первого
// касания.
function scrollToActive(box, behavior = "smooth") {
  const active = box.querySelector(".is-active");
  if (!active || box.scrollWidth <= box.clientWidth) return;
  const a = active.getBoundingClientRect();
  const b = box.getBoundingClientRect();
  // `behavior` — перечисление WebIDL: клиент, который не знает значения
  // «instant», бросает TypeError прямо на входе. Здесь это стоило бы всего
  // остального в `main()` — обработчиков загрузки, кнопок, опроса задач.
  try {
    box.scrollBy({ left: a.left - b.left - (b.width - a.width) / 2, behavior });
  } catch (_) {
    box.scrollLeft += a.left - b.left - (b.width - a.width) / 2;
  }
}

// Стрелки ленты видны, только когда в их сторону есть куда листать: стрелка
// «назад» на самом начале ленты обещала бы то, чего нет. Пересчитываются на
// прокрутке, на смене размера и после каждой перерисовки ленты — ширина
// таблеток меняется с языком и с загрузкой шрифтов тем.
function syncArrows(strip) {
  const track = strip.querySelector(".strip__track");
  const max = track.scrollWidth - track.clientWidth;
  const prev = max > 2 && track.scrollLeft > 2;
  const next = max > 2 && track.scrollLeft < max - 2;
  strip.querySelector(".strip__arrow--prev").hidden = !prev;
  strip.querySelector(".strip__arrow--next").hidden = !next;
  strip.classList.toggle("has-prev", prev);
  strip.classList.toggle("has-next", next);
}
function syncAllArrows() {
  for (const strip of $$(".strip")) syncArrows(strip);
}
function initStrips() {
  for (const strip of $$(".strip")) {
    const track = strip.querySelector(".strip__track");
    track.addEventListener("scroll", () => syncArrows(strip), { passive: true });
    for (const btn of strip.querySelectorAll(".strip__arrow")) {
      btn.addEventListener("click", () => {
        haptic();
        const step = Math.max(160, track.clientWidth * 0.7) * Number(btn.dataset.dir);
        try { track.scrollBy({ left: step, behavior: "smooth" }); } catch (_) { track.scrollLeft += step; }
      });
    }
    if (window.ResizeObserver) new ResizeObserver(() => syncArrows(strip)).observe(track);
  }
}

function renderPresets() {
  const box = $("#preset-tabs");
  $("#style-count").textContent = state.boot.presets.length;
  box.replaceChildren(...state.boot.presets.map((p) =>
    el("button", {
      class: `style-pill${p.name === state.preset ? " is-active" : ""}`,
      type: "button",
      // Не «вкладка»: панели у неё нет, и скринридер обещал бы то, чего нет.
      // Это выбор одного из многих — радиогруппа.
      role: "radio",
      title: `${p.label || p.title} · ${p.title}`,
      "aria-checked": p.name === state.preset ? "true" : "false",
      onclick: () => {
        state.preset = p.name;
        haptic();
        renderPresets();
        renderAccents();
        renderFonts();
        preview.render();
        scrollToActive(box);
      },
    }, [
      // Таблетка — это образец: подпись набрана шрифтом темы и её регистром.
      // Семнадцать одинаково набранных подписей ничего не говорили бы о том,
      // чем темы отличаются. Текст — понятная подпись («Крупные субтитры»), а
      // не название темы: «Chisel» тому, кто видит её впервые, не говорит ничего.
      el("span", {
        class: `style-pill__name${p.uppercase ? " is-caps" : ""}`,
        text: p.uppercase ? (p.label || p.title).toUpperCase() : (p.label || p.title),
        style: `font-family:"${p.font}", system-ui, sans-serif;letter-spacing:${p.letter_spacing}em`,
      }),
    ])));
  syncArrows(box.parentNode);
}

function renderAccents() {
  const p = activePreset();
  const box = $("#accent-chips");
  $("#accent-count").textContent = state.boot.accents.length;
  box.replaceChildren(...state.boot.accents.map((a) =>
    el("button", {
      class: `swatch${a.key === state.accent ? " is-active" : ""}`,
      type: "button",
      role: "radio",
      "aria-checked": a.key === state.accent ? "true" : "false",
      title: a.title,
      onclick: () => {
        state.accent = a.key;
        haptic();
        renderAccents();
        preview.render();
        scrollToActive(box);
      },
    }, [
      el("i", { class: "swatch__dot", style: `background:${a.color || (p ? p.colors.accent : "#F3D28F")}` }),
      el("span", { class: "swatch__name", text: a.title }),
    ])));
  syncArrows($("#accent-chips").parentNode);
}

function renderFonts() {
  const box = $("#font-chips");
  $("#font-count").textContent = (state.boot.fonts || []).length;
  box.replaceChildren(...(state.boot.fonts || []).map((f) =>
    el("button", {
      class: `chip${f.key === state.font ? " is-active" : ""}`,
      type: "button",
      role: "radio",
      "aria-checked": f.key === state.font ? "true" : "false",
      title: f.family || activeThemeFamily(),
      // Образец рисуется САМИМ шрифтом: название «Антиква» ничего не говорит,
      // пока не видно букв. Подпись при этом остаётся интерфейсным шрифтом —
      // иначе «Моноширинный», набранный Cormorant, читается как ещё один образец.
      style: `--chip-font:"${f.family || activeThemeFamily()}"`,
      onclick: () => {
        state.font = f.key;
        haptic();
        renderFonts();
        preview.render();
        scrollToActive(box);
      },
    }, [
      el("span", { class: "chip__aa", text: "Аа" }),
      el("span", { class: "chip__name", text: f.title }),
    ])));
  syncArrows($("#font-chips").parentNode);
}

function renderOptions() {
  for (const input of $$("[data-option]")) input.checked = !!state.options[input.dataset.option];
  renderCut();
  renderLayout();
}

// Раскладка кадра — выбор из двух, как нарезка. Помнится между заходами: кто
// снимает ролики «сверху товар, снизу я», снимает их подряд.
const LAYOUT_KEY = "reelbot:layout";

function loadLayout() {
  try { return localStorage.getItem(LAYOUT_KEY) === "split" ? "split" : "single"; } catch (_) { return "single"; }
}

function renderLayout() {
  const split = state.layout === "split";
  $("#layout-seg").replaceChildren(...[["single", "layout.single"], ["split", "layout.split"]].map(([value, key]) =>
    el("button", {
      type: "button", role: "radio", class: value === state.layout ? "is-active" : "",
      "aria-checked": value === state.layout ? "true" : "false", text: tr(key),
      onclick: () => {
        if (state.busy || state.layout === value) return;
        state.layout = value;
        try { localStorage.setItem(LAYOUT_KEY, value); } catch (_) { /* не критично */ }
        haptic();
        renderLayout();
      },
    })));
  $("#split-frame").classList.toggle("is-split", split);
  $("#top-dropzone").hidden = !split;
  $("#drop-empty-title").textContent = tr(split ? "split.bottom_title" : "cta.choose");
  $("#drop-role").textContent = tr("split.bottom_sub");
  $("#drop-role").hidden = !split;
  $("#drop-role-chip").hidden = !split;
  $("#s1-lead").textContent = tr(split ? "edit.s1_lead_split" : "edit.s1_lead");
  // Два файла в чат с ботом не отправить, а картинок по смыслу в этом режиме
  // нет вовсе: верхнюю половину пользователь показывает сам.
  $("#send-in-chat").hidden = split;
  $('[data-option="broll"]').closest(".toggle-row").hidden = split;
  renderNext();
  renderCta();
}

// Готово ли к монтажу: в «Двух видео» нужны оба файла.
function ready() {
  return !!state.file && (state.layout !== "split" || !!state.topFile);
}

// Какое поле открыть, если файла не хватает.
function missingInput() {
  return $(!state.file ? "#file-input" : "#top-input");
}

// ------------------------------------------------------------------ экран «Монтаж», шаг 1

// Нарезка — выбор из двух, а не тумблер: на образце это режимы, и «без
// нарезки» читается как осознанный выбор, а не забытая галочка. Третьего
// режима нет, пока пайплайн его не умеет (`core.options`).
function renderCut() {
  const box = $("#cut-seg");
  const on = !!state.options.cut;
  box.replaceChildren(...[[true, "cut.on"], [false, "cut.off"]].map(([value, key]) =>
    el("button", {
      type: "button", role: "radio", class: value === on ? "is-active" : "",
      "aria-checked": value === on ? "true" : "false", text: tr(key),
      onclick: () => { state.options.cut = value; haptic(); renderCut(); },
    })));
  $("#cut-hint").textContent = tr(on ? "cut.on_hint" : "cut.off_hint");
}

function renderLimits() {
  const l = state.boot.limits;
  $("#drop-limits").textContent = tr("edit.limits", { mb: l.max_upload_mb, min: l.min_seconds, max: l.max_minutes });
}

// Поля загрузки: нижнее (говорящий, оно же единственное в обычном режиме) и
// верхнее режима «Два видео».
const SLOTS = {
  bottom: { key: "file", zone: "#dropzone", thumb: "#drop-thumb", title: "#drop-title", sub: "#drop-sub", input: "#file-input", remove: "#drop-remove" },
  top: { key: "topFile", zone: "#top-dropzone", thumb: "#top-thumb", title: "#top-title", sub: "#top-sub", input: "#top-input", remove: "#top-remove" },
};

function setFile(file, which = "bottom") {
  const slot = SLOTS[which];
  state[slot.key] = file;
  const zone = $(slot.zone);
  const thumb = $(slot.thumb);
  zone.classList.toggle("has-file", !!file);
  thumb.style.backgroundImage = "";
  thumb.classList.remove("has-image");
  renderNext();
  if (!file) {
    $(slot.input).value = "";
    renderCta();
    return;
  }
  $(slot.title).textContent = file.name || tr("edit.untitled");
  $(slot.sub).textContent = fmtSize(file.size);
  // Длительность читаем браузером до загрузки: слишком длинное видео лучше
  // отбить сразу, а не после минуты заливки по мобильной сети. Заодно снимаем
  // кадр на обложку: имя файла вроде `IMG_4821.mov` не говорит, то ли это видео.
  const url = URL.createObjectURL(file);
  const probe = document.createElement("video");
  probe.preload = "metadata";
  probe.muted = true;
  probe.playsInline = true;
  // Освобождение — один раз и в любом случае. Перемотка может завершиться
  // вообще без события (`seekable` пуст — так бывает у .mkv и .webm без
  // индекса): тогда ни `seeked`, ни `error` не придут, и видео с буфером
  // осталось бы жить до перезагрузки страницы. Сторож снимает это через три
  // секунды — обложки не будет, остальное работает.
  let freed = false;
  const done = () => {
    if (freed) return;
    freed = true;
    clearTimeout(guard);
    probe.removeAttribute("src");
    URL.revokeObjectURL(url);
  };
  const guard = setTimeout(done, 3000);
  probe.onloadedmetadata = () => {
    const d = probe.duration;
    if (state[slot.key] !== file) { done(); return; }
    $(slot.sub).textContent = `${fmtSize(file.size)} · ${fmtDuration(d)}`;
    // Пределы длины — у речи. Верхнему они не нужны: короткое просто раньше
    // раскроет нижнее на весь экран, длинное обрежется.
    const lim = state.boot.limits;
    if (which === "bottom" && isFinite(d) && d > lim.max_minutes * 60 + 1) toast(tr("file.too_long", { max: lim.max_minutes }));
    else if (which === "bottom" && isFinite(d) && d < lim.min_seconds) toast(tr("file.too_short", { min: lim.min_seconds }));
    try { probe.currentTime = Math.min(0.5, (isFinite(d) ? d : 1) / 2); } catch (_) { done(); }
  };
  probe.onseeked = () => {
    try {
      const side = Math.min(probe.videoWidth, probe.videoHeight);
      if (side && state[slot.key] === file) {
        const c = document.createElement("canvas");
        c.width = c.height = 128;
        c.getContext("2d").drawImage(
          probe, (probe.videoWidth - side) / 2, (probe.videoHeight - side) / 2, side, side, 0, 0, 128, 128,
        );
        thumb.style.backgroundImage = `url(${c.toDataURL("image/jpeg", 0.72)})`;
        thumb.classList.add("has-image");
      }
    } catch (_) { /* кодек не отдаёт кадр в canvas — остаётся иконка */ }
    done();
  };
  probe.onerror = done;
  probe.src = url;
  renderCta();
}

function renderNext() {
  const btn = $("#next");
  btn.classList.toggle("is-idle", !ready());
  $("#next-sub").textContent = tr(
    ready() ? "edit.next_ready" : (state.layout === "split" && (state.file || state.topFile) ? "split.need_both" : "edit.next_idle"),
  );
}

function setStep(step) {
  state.step = step;
  for (const s of $$("[data-step]")) s.hidden = Number(s.dataset.step) !== step;
  window.scrollTo({ top: 0 });
  renderHeader();
  syncBackButton();
  syncPreview();
  if (step === 2) {
    // Ленты меряются только видимыми: на скрытом шаге ширина нулевая. А шрифты
    // плиток браузер начинает тянуть, только когда шаг стал видимым, — первый
    // замер идёт по подставному шрифту, поэтому второй ждёт их загрузки.
    const fit = () => {
      syncAllArrows();
      for (const id of ["#preset-tabs", "#font-chips", "#accent-chips"]) scrollToActive($(id), "instant");
      preview.render();
    };
    fit();
    if (document.fonts && document.fonts.ready) requestAnimationFrame(() => document.fonts.ready.then(fit));
  }
}

function initEdit() {
  $("#preview-play").addEventListener("click", () => { preview.toggle(); haptic(); });
  $("#requirements").addEventListener("click", () => {
    const l = state.boot.limits;
    popup(tr("req.title"), [
      tr("req.duration", { min: l.min_seconds, max: l.max_minutes }),
      tr("req.size", { mb: l.max_upload_mb }),
      tr("req.speaker"),
      tr("req.orientation"),
    ].join("\n"));
  });
  $("#send-in-chat").addEventListener("click", () => {
    haptic();
    openTelegram(`https://t.me/${state.boot.bot_username}`);
  });

  for (const [which, slot] of Object.entries(SLOTS)) {
    const input = $(slot.input);
    input.addEventListener("change", () => setFile(input.files[0] || null, which));
    const zone = $(slot.zone);
    zone.addEventListener("click", (ev) => {
      if (state[slot.key] && ev.target.closest(slot.remove)) {
        ev.preventDefault();
        if (!state.busy) { setFile(null, which); haptic(); }
      } else if (state.busy) {
        ev.preventDefault();
      }
    });
    zone.addEventListener("dragover", (ev) => { ev.preventDefault(); zone.classList.add("is-over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("is-over"));
    zone.addEventListener("drop", (ev) => {
      ev.preventDefault();
      zone.classList.remove("is-over");
      const f = ev.dataTransfer.files[0];
      if (f && !state.busy) setFile(f, which);
    });
  }

  for (const box of $$("[data-option]")) {
    box.addEventListener("change", () => { state.options[box.dataset.option] = box.checked; haptic(); });
  }
  for (const btn of $$("[data-help]")) {
    btn.addEventListener("click", () => popup(...tr(`help.${btn.dataset.help}`)));
  }
  $("#next").addEventListener("click", () => {
    if (!ready()) { haptic("warning"); missingInput().click(); return; }
    haptic();
    setStep(2);
  });
  $("#to-step1").addEventListener("click", () => { haptic(); setStep(1); });
  $("#cta").addEventListener("click", onCta);
}

// Платит ли пользователь за ролик. Владельцы не платят никогда, остальные —
// пока не включён бесплатный режим (`free` в bootstrap, FREE_VIDEOS на сервере).
function pays(u) {
  return !u.is_admin && !(state.boot && state.boot.free);
}

function renderCta(progressText, share) {
  const btn = $("#cta");
  const u = state.boot && state.boot.user;
  const title = $("#cta-title");
  const sub = $("#cta-sub");
  $("#cta-fill").style.width = share ? `${Math.round(share * 100)}%` : "0";
  if (!u) return;
  if (progressText) {
    title.textContent = progressText;
    sub.textContent = tr("cta.keep_open");
    btn.disabled = true;
    return;
  }
  btn.disabled = false;
  if (pays(u) && u.balance < 1) {
    title.textContent = tr("cta.topup");
    sub.textContent = tr("cta.missing");
  } else {
    // Пока файла нет, нажатие открывает выбор файла, а не запускает монтаж —
    // так кнопка и должна называться.
    title.textContent = tr(ready() ? "cta.edit" : "cta.choose");
    const free = u.is_admin ? "cta.free" : "cta.free_all";
    sub.textContent = !ready()
      ? tr(state.layout === "split" ? "split.need_both" : "cta.pick")
      : (pays(u) ? tr("cta.charge", { n: u.balance }) : tr(free));
  }
}

async function onCta() {
  if (state.busy) return;
  const u = state.boot.user;
  if (pays(u) && u.balance < 1) { go("tariffs"); return; }
  if (!ready()) { haptic("warning"); missingInput().click(); return; }

  state.busy = true;
  try {
    const job = await uploadAndStart();
    haptic("success");
    setFile(null);
    setFile(null, "top");
    state.step = 1;
    for (const s of $$("[data-step]")) s.hidden = Number(s.dataset.step) !== 1;
    await refreshBoot();
    toast(tr("toast.accepted"));
    state.jobs = [job, ...state.jobs.filter((j) => j.id !== job.id)];
    state.jobsLoaded = true;
    go("videos", { replace: true });
  } catch (e) {
    haptic("error");
    if (e.status === 402) { await refreshBoot(); go("tariffs"); }
    popup(tr("err.title"), e.message);
  } finally {
    state.busy = false;
    renderCta();
  }
}

// Один файл кусками; `onBytes(сколько принято)` — для общей полосы прогресса.
async function uploadOne(file, onBytes) {
  const start = await api("/api/uploads", { method: "POST", body: { size: file.size, name: file.name } });
  const id = start.upload_id;
  const size = file.size;
  let offset = start.received || 0;
  let failures = 0;
  while (offset < size) {
    const end = Math.min(size, offset + start.chunk_bytes);
    onBytes(offset);
    try {
      const res = await api(`/api/uploads/${id}?offset=${offset}`, { method: "PUT", body: file.slice(offset, end), raw: true });
      offset = res.received;
      failures = 0;
    } catch (e) {
      // 409 — сервер принял больше или меньше, чем мы думаем: продолжаем с его числа.
      if (e.status === 409 && e.data && typeof e.data.received === "number") { offset = e.data.received; continue; }
      if (e.status && e.status < 500) throw e;
      failures += 1;
      if (failures > 5) throw new Error(tr("upload.lost"));
      await new Promise((r) => setTimeout(r, 1500 * failures));
    }
  }
  return id;
}

// Нижнее видео, затем верхнее («Два видео»), и одна задача на оба. Полоса —
// общая по байтам: два прогресса подряд читались бы как «загрузилось и
// началось заново».
async function uploadAndStart() {
  const files = state.layout === "split" ? [state.file, state.topFile] : [state.file];
  const total = files.reduce((n, f) => n + f.size, 0) || 1;
  let before = 0;
  const show = (sent) => {
    const share = (before + sent) / total;
    renderCta(tr("cta.uploading", { p: Math.floor(share * 100) }), share);
  };
  const ids = [];
  for (const f of files) {
    ids.push(await uploadOne(f, show));
    before += f.size;
  }
  renderCta(tr("cta.checking"), 1);
  const body = {
    preset: state.preset,
    options: { ...state.options, accent: state.accent, font: state.font },
  };
  if (ids[1]) body.top_upload_id = ids[1];
  const res = await api(`/api/uploads/${ids[0]}/finish`, { method: "POST", body });
  return res.job;
}

// ------------------------------------------------------------------ «Главная»

function jobTitle(j) {
  return j.result_duration ? `${tr("jobs.reel")} · ${fmtDuration(j.result_duration)}` : tr("jobs.reel");
}

function renderHome() {
  const box = $("#home-recent");
  if (!state.jobsLoaded) {
    box.replaceChildren(el("div", { class: "skeleton" }));
    return;
  }
  const recent = state.jobs.slice(0, 3);
  if (!recent.length) {
    box.replaceChildren(el("button", { class: "card list-row", type: "button", onclick: () => go("edit") }, [
      el("span", { class: "thumb" }, [icon("film", 20)]),
      el("span", { class: "list-row__text" }, [el("b", { text: tr("home.empty") }), el("span", { class: "muted", text: tr("home.empty_sub") })]),
      icon("chevron", 18, "chev"),
    ]));
    return;
  }
  box.replaceChildren(...recent.map((j) => {
    const active = j.status === "queued" || j.status === "running";
    const sub = active
      ? `${tr(`status.${j.status}`)} · ${Math.round((j.progress || 0) * 100)}%`
      : [j.status === "done" ? null : tr(`status.${j.status}`), fmtWhen(j.created_at)].filter(Boolean).join(" · ");
    return el("button", { class: "card list-row", type: "button", onclick: () => go("videos") }, [
      el("span", { class: "thumb" }, [icon(active ? "clock" : "film", 20)]),
      el("span", { class: "list-row__text" }, [el("b", { text: jobTitle(j) }), el("span", { class: "muted", text: sub })]),
      icon("chevron", 18, "chev"),
    ]);
  }));
}

// ------------------------------------------------------------------ «Результаты»

const JOB_FILTERS = {
  all: () => true,
  done: (j) => j.status === "done",
  active: (j) => j.status === "queued" || j.status === "running",
};

function renderJobFilter() {
  $("#jobs-filter").replaceChildren(...Object.keys(JOB_FILTERS).map((key) => {
    const n = state.jobs.filter(JOB_FILTERS[key]).length;
    return el("button", {
      type: "button", role: "radio", class: key === state.jobsFilter ? "is-active" : "",
      "aria-checked": key === state.jobsFilter ? "true" : "false",
      onclick: () => { state.jobsFilter = key; haptic(); renderJobs(); },
    }, [tr(`jobs.${key}`), state.jobsLoaded && n ? el("b", { text: String(n) }) : null]);
  }));
}

function renderJobs() {
  const box = $("#jobs");
  renderJobFilter();
  const retention = $("#jobs-retention");
  retention.hidden = !(state.boot.retention_days > 0 && state.jobs.length);
  retention.textContent = tr("jobs.retention", { n: state.boot.retention_days });
  if (!state.jobsLoaded) {
    box.replaceChildren(...[0, 1, 2].map(() => el("div", { class: "skeleton" })));
    return;
  }
  const jobs = state.jobs.filter(JOB_FILTERS[state.jobsFilter]);
  if (!jobs.length) {
    const none = !state.jobs.length;
    box.replaceChildren(el("div", { class: "card empty" }, [
      el("b", { text: tr(none ? "jobs.empty" : "jobs.empty_filter") }),
      none ? el("p", { class: "muted", text: tr("jobs.empty_hint") }) : null,
      el("button", { class: "btn-soft", type: "button", text: tr("jobs.to_edit"), onclick: () => go("edit") }),
    ]));
    return;
  }
  const presets = Object.fromEntries(state.boot.presets.map((p) => [p.name, p.label || p.title]));
  box.replaceChildren(...jobs.map((j) => {
    const accent = state.boot.accents.find((a) => a.key === (j.options && j.options.accent));
    const meta = [presets[j.preset] || j.preset, accent && accent.key !== "theme" ? accent.title : null, fmtWhen(j.created_at)].filter(Boolean).join(" · ");
    const parts = [
      el("div", { class: "job__head" }, [
        el("div", {}, [el("div", { class: "job__title", text: jobTitle(j) }), el("div", { class: "muted", text: meta })]),
        el("span", { class: `status status--${j.status}`, text: tr(`status.${j.status}`) }),
      ]),
    ];
    if (j.status === "queued" || j.status === "running") {
      parts.push(el("div", { class: "muted", text: j.stage || "" }));
      parts.push(el("div", { class: "bar" }, [el("i", { style: `width:${Math.round((j.progress || 0) * 100)}%` })]));
    } else if (j.status === "failed") {
      parts.push(el("div", { class: "muted", text: j.error || "" }));
      if (pays(state.boot.user)) parts.push(el("div", { class: "muted", text: tr("jobs.refunded") }));
    } else if (j.purged) {
      parts.push(el("div", { class: "muted", text: tr("jobs.purged") }));
    } else if (j.status === "done" && j.video_url) {
      // `#t=0.5` — браузер покажет кадр ролика обложкой, а не чёрный прямоугольник.
      parts.push(el("video", { src: `${j.video_url}#t=0.5`, controls: true, playsinline: true, preload: "metadata" }));
      parts.push(el("div", { class: "job__actions" }, [
        el("button", {
          class: "btn-soft", type: "button", text: tr("jobs.send"),
          onclick: async (ev) => {
            const b = ev.currentTarget;
            b.disabled = true;
            try { await api(`/api/jobs/${j.id}/send`, { method: "POST" }); haptic("success"); toast(tr("jobs.sent")); }
            catch (e) { haptic("error"); popup(tr("jobs.not_sent"), e.message); }
            finally { b.disabled = false; }
          },
        }),
      ]));
    }
    return el("div", { class: "card job" }, parts);
  }));
}

function renderActivity() {
  const active = state.jobs.some((j) => j.status === "queued" || j.status === "running");
  for (const dot of $$("[data-active-dot]")) dot.hidden = !active;
}

async function loadJobs() {
  try {
    const data = await api("/api/jobs");
    const sig = (list) => JSON.stringify(list.map((j) => [j.id, j.status, j.progress, j.stage]));
    const changed = !state.jobsLoaded || sig(data.jobs) !== sig(state.jobs);
    state.jobs = data.jobs;
    state.jobsLoaded = true;
    state.boot.user.balance = data.balance;
    renderBalance();
    renderActivity();
    // Перерисовываем только при изменениях: иначе плеер готового ролика
    // сбрасывался бы каждые три секунды посреди просмотра.
    if (changed) {
      if (state.screen === "videos") renderJobs();
      if (state.screen === "home") renderHome();
    }
  } catch (e) {
    if (state.screen === "videos") toast(e.message);
  }
  schedulePoll();
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  const active = state.jobs.some((j) => j.status === "queued" || j.status === "running");
  if (active) state.pollTimer = setTimeout(loadJobs, 3000);
}

// ------------------------------------------------------------------ «Тарифы»

function packageFacts() {
  const list = state.boot.packages || [];
  const base = list.length ? list[0].stars / list[0].videos : 0;
  // «Популярный» — средний пакет, когда их хотя бы три: самый частый выбор
  // между «попробовать» и «оптом». Метка — подсказка, а не продажа: выгода
  // пишется честным процентом от цены одиночного ролика.
  const popular = list.length >= 3 ? Math.floor((list.length - 1) / 2) : -1;
  return list.map((p, i) => ({
    ...p,
    popular: i === popular,
    per: Math.round(p.stars / p.videos),
    save: base ? Math.round((1 - p.stars / p.videos / base) * 100) : 0,
  }));
}

function renderTariffs() {
  const facts = packageFacts();
  if (state.pkg === null || !facts.some((p) => p.videos === state.pkg)) {
    const pop = facts.find((p) => p.popular) || facts[0];
    state.pkg = pop ? pop.videos : null;
  }
  $("#packages").replaceChildren(...facts.map((p) => {
    const on = p.videos === state.pkg;
    return el("button", {
      class: "package", type: "button", role: "radio", "aria-checked": on ? "true" : "false",
      onclick: () => { state.pkg = p.videos; haptic(); renderTariffs(); },
    }, [
      el("span", { class: `radio${on ? " is-on" : ""}` }),
      el("span", { class: "package__main" }, [
        p.popular ? el("span", { class: "badge", text: tr("pkg.popular") }) : null,
        el("span", { class: "package__amount" }, [el("b", { text: String(p.videos) }), el("span", { class: "muted", text: videosWord(p.videos) })]),
        el("span", { class: "muted", text: p.videos > 1 ? tr("pkg.per", { n: p.per }) : tr("pkg.try") }),
      ]),
      el("span", { class: "package__price" }, [
        el("b", { text: `${p.stars} ⭐` }),
        p.save > 0 ? el("span", { class: "package__save", text: tr("pkg.save", { n: p.save }) }) : null,
      ]),
    ]);
  }));
  const chosen = facts.find((p) => p.videos === state.pkg);
  $("#pay-method-price").textContent = chosen ? `${chosen.stars} ⭐` : "";
  $("#pay-title").textContent = chosen ? tr("pay.button", { n: chosen.stars }) : "";
  $("#pay-sub").textContent = chosen ? tr("pay.sub", { n: chosen.videos, w: videosWord(chosen.videos) }) : "";
  $("#pay").hidden = !chosen;
}

async function buy(videos) {
  haptic();
  if (!tg || !tg.openInvoice) { popup(tr("pay.title"), tr("pay.update")); return; }
  let link;
  try {
    link = (await api("/api/invoice", { method: "POST", body: { videos } })).link;
  } catch (e) {
    popup(tr("err.title"), e.message);
    return;
  }
  const before = state.boot.user.balance;
  tg.openInvoice(link, async (status) => {
    if (status !== "paid") {
      if (status === "failed") popup(tr("pay.failed"), tr("pay.failed_text"));
      return;
    }
    haptic("success");
    toast(tr("pay.crediting"));
    // Зачисляет бот по сообщению Telegram об оплате, а не этот ответ клиента,
    // поэтому ждём, пока баланс действительно вырастет.
    for (let i = 0; i < 15; i += 1) {
      await refreshBoot();
      if (state.boot.user.balance > before) { toast(tr("pay.credited", { n: videos, w: videosWord(videos) })); return; }
      await new Promise((r) => setTimeout(r, 1000));
    }
    toast(tr("pay.later"));
  });
}

// ------------------------------------------------------------------ лист снизу

let sheetReturnFocus = null;
function openSheet(title, body) {
  sheetReturnFocus = document.activeElement;
  $("#sheet-title").textContent = title;
  $("#sheet-body").replaceChildren(...[].concat(body));
  $("#sheet").hidden = false;
  syncBackButton();
  const close = $("#sheet [data-sheet-close].icon-btn");
  if (close) close.focus();
}
function closeSheet() {
  if ($("#sheet").hidden) return false;
  $("#sheet").hidden = true;
  syncBackButton();
  if (sheetReturnFocus && sheetReturnFocus.focus) sheetReturnFocus.focus();
  return true;
}

async function openHistory() {
  haptic();
  openSheet(tr("history.title"), el("div", { class: "skeleton" }));
  let items;
  try {
    items = (await api("/api/payments")).items;
  } catch (e) {
    $("#sheet-body").replaceChildren(el("p", { class: "muted", text: e.message }));
    return;
  }
  if (!items.length) {
    $("#sheet-body").replaceChildren(el("p", { class: "muted empty-line", text: tr("history.empty") }));
    return;
  }
  $("#sheet-body").replaceChildren(...items.map((it) => {
    const bonus = it.kind === "bonus";
    return el("div", { class: "history-row" }, [
      el("span", { class: "tile" }, [icon(bonus ? "invite" : "star", 18, bonus ? null : "star")]),
      el("span", { class: "list-row__text" }, [
        el("b", { text: tr(bonus ? "history.bonus" : "history.payment") }),
        el("span", { class: "muted", text: fmtWhen(it.at) }),
      ]),
      el("span", { class: "list-row__end" }, [
        el("b", { class: "pos", text: bonus ? `+${fmtShare(it.share)}` : `+${it.videos}` }),
        el("span", { class: "muted", text: bonus ? videosWord(2) : `${it.stars} ⭐` }),
      ]),
    ]);
  }));
}

// ------------------------------------------------------------------ «Приглашения»

function referralLink() {
  const r = state.boot.referral;
  return r ? `https://t.me/${state.boot.bot_username}?start=${r.code}` : "";
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (_) {
    // WebView Telegram на части клиентов не пускает в буфер обмена через
    // Clipboard API — старый путь через выделение работает почти везде.
    const area = el("textarea", { style: "position:fixed;opacity:0" });
    area.value = text;
    document.body.append(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch (_) { ok = false; }
    area.remove();
    return ok;
  }
}

function shareInvite() {
  const share = `https://t.me/share/url?url=${encodeURIComponent(referralLink())}&text=${encodeURIComponent(tr("invite.text"))}`;
  haptic();
  openTelegram(share);
}

function renderInviteStatic() {
  const r = state.boot.referral;
  if (!r) return;
  $("#invite-lead").textContent = tr("invite.lead", { p: r.percent });
  $("#invite-step3").textContent = tr("invite.step3", { p: r.percent });
  $("#invite-link").textContent = referralLink();
  const credit = $("#ref-credit");
  credit.hidden = !(r.credit > 0);
  credit.textContent = tr("invite.credit", { n: fmtShare(r.credit) });
  $("#ref-period").replaceChildren(...[7, 30, 0].map((d) =>
    el("button", {
      type: "button", role: "radio", class: d === state.refDays ? "is-active" : "",
      "aria-checked": d === state.refDays ? "true" : "false",
      onclick: () => { if (state.refDays !== d) { state.refDays = d; haptic(); renderInviteStatic(); loadReferrals(); } },
    }, d ? [icon("calendar", 15), tr(`period.${d}`)] : [tr("period.0")])));
}

function renderReferrals() {
  const data = state.refs;
  if (!data) {
    $("#ref-friends").replaceChildren(el("div", { class: "skeleton" }));
    return;
  }
  const s = data.stats;
  $("#ref-invited").textContent = s.invited;
  $("#ref-paid").textContent = s.paid;
  $("#ref-earned").textContent = `+${fmtShare(s.earned)}`;
  $("#ref-invited-today").textContent = tr("invite.today", { n: s.invited_today });
  $("#ref-paid-today").textContent = tr("invite.today", { n: s.paid_today });
  $("#ref-earned-today").textContent = tr("invite.today", { n: fmtShare(s.earned_today) });

  const friends = data.friends;
  const paid = friends.filter((f) => f.paid).length;
  const earned = friends.reduce((sum, f) => sum + f.earned, 0);
  $("#ref-summary").textContent = tr("invite.summary", { a: friends.length, b: paid, c: fmtShare(earned) });
  if (!friends.length) {
    $("#ref-friends").replaceChildren(el("div", { class: "empty-line" }, [
      el("b", { text: tr("invite.none") }), el("span", { class: "muted small", text: tr("invite.none_sub") }),
    ]));
    return;
  }
  $("#ref-friends").replaceChildren(...friends.map((f) => {
    const name = f.name || tr("invite.noname");
    return el("div", { class: "list-row" }, [
      el("span", { class: "avatar", text: name.slice(0, 1).toUpperCase() }),
      el("span", { class: "list-row__text" }, [el("b", { text: name }), el("span", { class: "muted", text: fmtDate(f.joined_at) })]),
      el("span", { class: "list-row__end" }, f.paid
        ? [el("b", { class: "pos", text: `+${fmtShare(f.earned)}` }), el("span", { class: "muted", text: tr("invite.friend_paid") })]
        : [el("span", { class: "muted", text: tr("invite.friend_waiting") })]),
    ]);
  }));
}

async function loadReferrals() {
  if (!state.boot.referral) return;
  const days = state.refDays;
  try {
    const data = await api(`/api/referrals?days=${days}&tz=${new Date().getTimezoneOffset()}`);
    if (days !== state.refDays) return; // пока шёл ответ, выбрали другой период
    state.refs = data;
    renderReferrals();
  } catch (e) {
    if (state.screen === "invite") toast(e.message);
  }
}

// ------------------------------------------------------------------ «Профиль» и общее

function renderUser() {
  const u = state.boot.user;
  const name = u.first_name || u.username || tr("profile.noname");
  for (const n of $$("[data-user-name]")) n.textContent = name;
  for (const n of $$("[data-user-handle]")) n.textContent = u.username ? `@${u.username}` : "";
  for (const n of $$("[data-avatar]")) n.textContent = name.slice(0, 1).toUpperCase();
  const brand = $("#brand");
  brand.replaceChildren(state.boot.bot_name || "Reels", el("i", { class: "grad-dot", text: "." }));
  for (const n of $$("[data-needs='referral']")) n.hidden = !state.boot.referral;
  const p = state.boot.referral ? state.boot.referral.percent : 0;
  $("#home-invite-sub").textContent = tr("home.invite_sub", { p });
  $("#profile-invite-sub").textContent = tr("home.invite_sub", { p });
  const offer = offerText();
  const offerNode = $("#home-offer");
  if (offerNode) {
    offerNode.textContent = offer;
    offerNode.hidden = !offer;
  }
}

// Условия одной строкой на первом экране: что бесплатно и сколько стоит
// следующий ролик. Раньше о цене человек узнавал отказом после загрузки видео.
// Владельцу строка не нужна — с него ролики не списываются.
function offerText() {
  const boot = state.boot;
  if (boot.user.is_admin) return "";
  if (boot.free) return tr("home.offer_free");
  const packs = (boot.packages || []).filter((x) => x.videos > 0);
  if (!packs.length) return "";
  const one = packs.find((x) => x.videos === 1);
  const single = one ? one.stars : Math.min(...packs.map((x) => x.stars));
  const best = Math.round(Math.min(...packs.map((x) => x.stars / x.videos)));
  // Проба ещё не потрачена: ни одного готового ролика, а на балансе что-то есть.
  const trial = boot.trial > 0 && boot.user.videos_done === 0 && boot.user.balance > 0;
  return tr(trial ? "home.offer_trial" : "home.offer_paid", { single, best });
}

function renderProfile() {
  const u = state.boot.user;
  $("#stat-done").textContent = u.videos_done;
  $("#profile-topup").hidden = u.is_admin;
  $("#profile-hint").textContent = tr(u.is_admin ? "profile.hint_admin" : "profile.hint");
  $("#profile-bot").textContent = state.boot.bot_username ? `@${state.boot.bot_username}` : "";
}

function renderBalance() {
  const u = state.boot.user;
  const n = u.balance;
  for (const node of $$("[data-balance-value]")) node.textContent = u.is_admin ? "∞" : n;
  for (const node of $$("[data-balance-unit]")) node.textContent = u.is_admin ? tr("balance.unlimited") : videosWord(n);
  for (const node of $$("[data-balance-count]")) node.textContent = u.is_admin ? "∞" : n;
  $("#balance-note").textContent = tr(
    u.is_admin ? "balance.note_admin" : (pays(u) ? "balance.note" : "balance.note_free"),
  );
  renderCta();
}

function renderLanguages() {
  const current = currentLang();
  for (const box of $$("[data-lang-seg]")) {
    box.replaceChildren(...["ru", "en"].map((code) =>
      el("button", {
        type: "button", class: code === current ? "is-active" : "", text: code.toUpperCase(),
        "aria-pressed": code === current ? "true" : "false",
        onclick: () => setLanguage(code),
      })));
  }
}

async function setLanguage(code) {
  if (code === currentLang()) return;
  haptic();
  const previous = state.boot.user.lang;
  state.boot.user.lang = code;
  rerenderAll();
  try {
    await api("/api/language", { method: "POST", body: { lang: code } });
    // Названия цветов, шрифтов и стадий приходят с сервера на языке
    // пользователя — перечитываем.
    const fresh = await api("/api/bootstrap");
    state.boot.accents = fresh.accents;
    state.boot.fonts = fresh.fonts;
    state.boot.user = fresh.user;
    rerenderAll();
    loadJobs();
  } catch (e) {
    state.boot.user.lang = previous;
    rerenderAll();
    popup(tr("err.title"), e.message);
  }
}

function rerenderAll() {
  applyStaticTexts();
  sidebarTitles();
  renderUser();
  renderLanguages();
  renderThemeSeg();
  renderPresets();
  renderAccents();
  renderFonts();
  renderOptions();
  renderLimits();
  renderNext();
  renderBalance();
  renderHeader();
  preview.render();
  if (state.file) setFile(state.file);
  if (state.topFile) setFile(state.topFile, "top");
  renderScreen(state.screen);
}

// ------------------------------------------------------------------ навигация

const TITLES = { home: "nav.home", invite: "title.invite", edit: "nav.edit", videos: "nav.videos", tariffs: "nav.tariffs", profile: "nav.profile" };

function renderHeader() {
  $("#page-title").textContent = tr(TITLES[state.screen] || "nav.home");
  const step = $("#page-step");
  step.hidden = state.screen !== "edit";
  step.textContent = tr("step", { n: state.step });
  document.title = `${tr(TITLES[state.screen] || "nav.home")} · ${(state.boot && state.boot.bot_name) || "Reels"}`;
}

function canGoBack() {
  return !$("#sheet").hidden || (state.screen === "edit" && state.step === 2) || state.history.length > 0;
}

function goBack() {
  if (closeSheet()) return;
  if (state.screen === "edit" && state.step === 2) { setStep(1); return; }
  const prev = state.history.pop();
  go(prev || "home", { push: false });
}

// Две кнопки «назад» — Telegram и своя в шапке — живут от одного условия:
// иначе открытый лист «Истории» закрывался одной и не закрывался другой.
function syncBackButton() {
  const back = canGoBack();
  $("#back").hidden = !back;
  if (!tg || !tg.BackButton) return;
  if (back) tg.BackButton.show(); else tg.BackButton.hide();
}

function renderScreen(screen) {
  if (screen === "home") renderHome();
  if (screen === "videos") renderJobs();
  if (screen === "tariffs") renderTariffs();
  if (screen === "profile") renderProfile();
  if (screen === "invite") { renderInviteStatic(); renderReferrals(); }
}

function go(screen, { push = true, replace = false } = {}) {
  screen = SCREEN_ALIASES[screen] || screen;
  if (!SCREENS.includes(screen)) screen = "home";
  if (screen === "invite" && !state.boot.referral) { shareInvite(); return; }
  closeSheet();
  if (screen !== state.screen) {
    // Главная — корень: дойдя до неё, историю дальше не копим, иначе «назад»
    // водил бы кругами по всем когда-то открытым вкладкам.
    if (screen === "home") state.history = [];
    else if (push && !replace && state.screen) state.history.push(state.screen);
    state.screen = screen;
  }
  for (const s of $$("[data-screen]")) s.hidden = s.dataset.screen !== screen;
  for (const b of $$("[data-nav]")) b.classList.toggle("is-active", b.dataset.nav === screen);
  window.scrollTo({ top: 0 });
  renderHeader();
  syncBackButton();
  syncPreview();
  renderScreen(screen);
  if (screen === "videos" || screen === "home") loadJobs();
  if (screen === "invite") loadReferrals();
  if (screen === "edit" && state.step === 2) setStep(2);
}

async function refreshBoot() {
  const fresh = await api("/api/bootstrap");
  state.boot.user = fresh.user;
  state.boot.referral = fresh.referral;
  renderBalance();
  renderUser();
}

const COLLAPSE_KEY = "reel-sidebar";
function initSidebar() {
  const shell = $("#app");
  let saved = null;
  try { saved = localStorage.getItem(COLLAPSE_KEY); } catch (_) { /* приватный режим */ }
  shell.classList.toggle("is-collapsed", saved === "1");
  $("#collapse").addEventListener("click", () => {
    const next = !shell.classList.contains("is-collapsed");
    shell.classList.toggle("is-collapsed", next);
    try { localStorage.setItem(COLLAPSE_KEY, next ? "1" : "0"); } catch (_) { /* не критично */ }
    haptic();
  });
  sidebarTitles();
}

// В свёрнутой панели подписи спрятаны — пусть живут во всплывающей подсказке.
// Отдельно, потому что язык меняется на ходу, и подсказки должны идти за ним.
function sidebarTitles() {
  for (const b of $$(".side-nav__item")) {
    const label = b.querySelector("[data-i18n]");
    if (label) b.title = label.textContent;
  }
}

async function main() {
  initTheme();
  applyStaticTexts();
  if (!tg || !tg.initData) {
    $("#blocker").hidden = false;
    return;
  }
  tg.ready();
  tg.expand();
  try { tg.disableVerticalSwipes && tg.disableVerticalSwipes(); } catch (_) { /* старый клиент */ }
  if (tg.BackButton) tg.BackButton.onClick(goBack);

  try {
    state.boot = await api("/api/bootstrap");
  } catch (e) {
    $("#blocker").hidden = false;
    $("#blocker h2").textContent = tr("blocker.failed");
    // 401 — подпись Telegram не прошла: причина с сервера на русском и пользователю
    // ничего не скажет, а «откройте из бота» — единственное, что он может сделать.
    $("#blocker p").textContent = e.status === 401 ? tr("blocker.text") : e.message;
    return;
  }
  const b = state.boot;
  applyStaticTexts();
  state.preset = b.presets.some((p) => p.name === b.user.preset) ? b.user.preset : (b.presets[0] || {}).name;
  state.options = { ...b.defaults };
  state.layout = loadLayout();
  // Запомненный выбор важнее умолчания: тема уже так и работает (`user.preset`),
  // а цвет со шрифтом до сих пор сбрасывались на каждом открытии приложения.
  const saved = (b.user && b.user.settings) || {};
  state.accent = saved.accent || b.defaults.accent || "theme";
  state.font = saved.font || b.defaults.font || "theme";

  declareFonts();
  renderUser();
  renderLanguages();
  renderPresets();
  renderAccents();
  renderFonts();
  renderOptions();
  renderLimits();
  renderNext();
  renderBalance();
  initEdit();
  initStrips();
  initSidebar();
  // Ширина строки и подписи плиток меряются шрифтами тем — до их загрузки
  // замер идёт по подставному системному и врёт в меньшую сторону: названия
  // приезжали обрезанными многоточием ровно на первом, холодном открытии.
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => {
      syncAllArrows();
      scrollToActive($("#preset-tabs"), "instant");
      preview.render();
    });
  }
  // Превью прилипает под шапкой и ужимается, когда человек ушёл листать
  // ленты: иначе выбор цвета и шрифта меняет то, чего в этот момент не видно.
  const wrap = $("#preview-wrap");
  addEventListener("scroll", () => {
    wrap.classList.toggle("is-compact", window.scrollY > 120);
  }, { passive: true });
  // Вкладка в фоне — таймер не нужен; возврат к ней пересчитывает кадр.
  document.addEventListener("visibilitychange", syncPreview);

  for (const btn of $$("[data-nav]")) btn.addEventListener("click", () => { haptic(); go(btn.dataset.nav); });
  $("#back").addEventListener("click", () => { haptic(); goBack(); });
  for (const n of $$("[data-sheet-close]")) n.addEventListener("click", () => closeSheet());
  addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeSheet(); });
  $("#history-open").addEventListener("click", openHistory);
  $("#pay").addEventListener("click", () => { if (state.pkg) buy(state.pkg); });
  $("#invite-copy").addEventListener("click", async () => {
    const ok = await copyText(referralLink());
    haptic(ok ? "success" : "error");
    toast(ok ? tr("invite.copied") : referralLink());
  });
  $("#invite-share").addEventListener("click", shareInvite);
  $("#open-chat").addEventListener("click", () => { haptic(); openTelegram(`https://t.me/${state.boot.bot_username}`); });

  // Вкладка приходит параметром `?screen=`, а не фрагментом: фрагмент занят
  // Telegram — в нём едут параметры запуска (`#tgWebAppData=…`).
  go(new URLSearchParams(location.search).get("screen") || "home", { push: false });
  if (state.screen !== "home" && state.screen !== "videos") loadJobs();
}

main();
