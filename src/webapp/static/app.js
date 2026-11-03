/* Mini App. Без сборки и фреймворка: один экран, несколько вкладок, API на том же хосте. */
"use strict";

const tg = window.Telegram && window.Telegram.WebApp;
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// Тексты интерфейса. Язык приходит с сервера (`user.lang`) и меняется в профиле.
const I18N = {
  ru: {
    theme: "Сменить тему", back: "Назад", forward: "Вперёд",
    "preview.pause": "Пауза", "preview.play": "Воспроизвести",
    "preview.s1": ["это", "пример", "субтитров", "в", "стиле"],
    "preview.s2": ["паузы", "уже", "вырезаны"],
    "preview.s3": ["и", "кадр", "по центру", "лица"],
    "edit.style": "Стиль субтитров", "edit.font": "Шрифт", "edit.color": "Цвет выделения",
    "edit.video": "Видео", "edit.requirements": "Требования к видео", "edit.upload": "Загрузите видео",
    "edit.params": "Параметры", "edit.untitled": "Видео",
    "opt.cut": "Нарезать видео", "opt.sfx": "Звуковые эффекты", "opt.broll": "Медиа-карточки", "opt.camera": "Движение камеры",
    "help.cut": ["Нарезать видео", "Вырезаем затяжные паузы дольше 1.2 секунды и долгие «эээ». Обычные вдохи и слова-паразиты остаются — так речь звучит естественно."],
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
    "cta.edit": "Смонтировать", "cta.topup": "Пополнить баланс", "cta.missing": "не хватает 1 ролика",
    "cta.free": "бесплатно для владельца", "cta.pick": "сначала выберите видео", "cta.charge": "спишется 1 ролик из {n}",
    "cta.uploading": "Загрузка {p}%", "cta.checking": "Проверяю видео…", "cta.keep_open": "не закрывайте приложение до конца загрузки",
    "upload.lost": "Связь оборвалась во время загрузки. Попробуйте ещё раз.",
    "toast.accepted": "Видео принято. Готовый ролик придёт в чат, а прогресс — во вкладке «Ролики».",
    "err.title": "Не получилось",
    "nav.invite": "Пригласить", "nav.videos": "Ролики", "nav.edit": "Монтаж", "nav.balance": "Баланс", "nav.profile": "Профиль",
    "balance.unlimited": "без ограничений", "balance.on": "На балансе", "balance.topup": "Пополнить звёздами",
    "balance.note": "Один ролик — один смонтированный видеофайл. Если монтаж не получится, ролик вернётся на баланс.",
    "balance.note_admin": "Вы владелец бота: монтаж бесплатный, баланс не списывается.",
    "status.queued": "в очереди", "status.running": "монтируется", "status.done": "готов", "status.failed": "не получилось", "status.cancelled": "отменён",
    "jobs.empty": "Роликов пока нет", "jobs.empty_hint": "Загрузите видео — готовый ролик придёт в чат и появится здесь.",
    "jobs.to_edit": "К монтажу", "jobs.reel": "Ролик", "jobs.refunded": "Ролик вернулся на баланс.",
    "jobs.purged": "Удалён с сервера по сроку хранения. В чате ролик остался.",
    "jobs.retention": "Готовые ролики хранятся на сервере {n} дн., в чате — всегда.",
    "jobs.send": "Прислать в чат", "jobs.sent": "Отправил в чат.", "jobs.not_sent": "Не отправилось",
    "pkg.per": "{n} ⭐ за ролик", "pkg.try": "попробовать",
    "pay.title": "Оплата", "pay.update": "Обновите Telegram — эта версия не умеет оплату звёздами.",
    "pay.failed": "Оплата не прошла", "pay.failed_text": "Звёзды не списаны. Попробуйте ещё раз.",
    "pay.crediting": "Оплачено. Зачисляю…", "pay.credited": "Зачислено: {n} {w}.", "pay.later": "Оплата прошла, баланс обновится через минуту.",
    "profile.noname": "Без имени", "profile.done": "смонтировано", "profile.balance": "на балансе", "profile.language": "Язык интерфейса",
    "profile.hint_admin": "Вы в списке владельцев: монтаж без списаний. Чтобы проверить оплату как обычный пользователь, уберите свой id из ADMIN_IDS.",
    "profile.hint": "Видео можно присылать и прямо в чат с ботом — стиль возьмётся последний выбранный здесь.",
    "invite.text": "Монтирую вертикальные ролики за пару минут: субтитры, паузы, кадр 9:16.",
    "blocker.title": "Откройте из Telegram", "blocker.text": "Приложение работает внутри бота: нажмите кнопку «Монтаж» в чате.",
    "blocker.failed": "Не получилось открыть",
  },
  en: {
    theme: "Toggle theme", back: "Back", forward: "Forward",
    "preview.pause": "Pause", "preview.play": "Play",
    "preview.s1": ["this", "is", "captions", "in", "style"],
    "preview.s2": ["pauses", "are", "gone"],
    "preview.s3": ["and", "the frame", "follows", "the face"],
    "edit.style": "Caption style", "edit.font": "Typeface", "edit.color": "Highlight colour",
    "edit.video": "Video", "edit.requirements": "Video requirements", "edit.upload": "Upload a video",
    "edit.params": "Settings", "edit.untitled": "Video",
    "opt.cut": "Cut the video", "opt.sfx": "Sound effects", "opt.broll": "Media cards", "opt.camera": "Camera movement",
    "help.cut": ["Cut the video", "We remove pauses longer than 1.2 seconds and long “umm”s. Normal breaths and filler words stay, so the speech sounds natural."],
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
    "cta.edit": "Create reel", "cta.topup": "Top up balance", "cta.missing": "you need 1 more video",
    "cta.free": "free for the owner", "cta.pick": "pick a video first", "cta.charge": "uses 1 of your {n} videos",
    "cta.uploading": "Uploading {p}%", "cta.checking": "Checking the video…", "cta.keep_open": "keep the app open until the upload finishes",
    "upload.lost": "The connection dropped during upload. Please try again.",
    "toast.accepted": "Video received. The finished reel will arrive in the chat; track progress in “Videos”.",
    "err.title": "Something went wrong",
    "nav.invite": "Invite", "nav.videos": "Videos", "nav.edit": "Editor", "nav.balance": "Balance", "nav.profile": "Profile",
    "balance.unlimited": "unlimited", "balance.on": "Your balance", "balance.topup": "Top up with Stars",
    "balance.note": "One video — one edited reel. If editing fails, the video goes back to your balance.",
    "balance.note_admin": "You own this bot: editing is free and your balance isn't used.",
    "status.queued": "queued", "status.running": "editing", "status.done": "ready", "status.failed": "failed", "status.cancelled": "cancelled",
    "jobs.empty": "No videos yet", "jobs.empty_hint": "Upload a video — the finished reel will arrive in the chat and show up here.",
    "jobs.to_edit": "Go to editor", "jobs.reel": "Reel", "jobs.refunded": "The video was returned to your balance.",
    "jobs.purged": "Removed from the server after the storage period. It's still in your chat.",
    "jobs.retention": "Finished videos stay on the server for {n} days and in your chat forever.",
    "jobs.send": "Send to chat", "jobs.sent": "Sent to the chat.", "jobs.not_sent": "Couldn't send",
    "pkg.per": "{n} ⭐ per video", "pkg.try": "try it out",
    "pay.title": "Payment", "pay.update": "Please update Telegram — this version can't pay with Stars.",
    "pay.failed": "Payment failed", "pay.failed_text": "No Stars were charged. Please try again.",
    "pay.crediting": "Paid. Adding to your balance…", "pay.credited": "Added: {n} {w}.", "pay.later": "Payment received; your balance will update in a minute.",
    "profile.noname": "No name", "profile.done": "edited", "profile.balance": "balance", "profile.language": "Interface language",
    "profile.hint_admin": "You're on the owners list: editing is free. To test payments as a regular user, remove your id from ADMIN_IDS.",
    "profile.hint": "You can also send videos straight to the bot chat — it will use the style you last picked here.",
    "invite.text": "I turn talking-head videos into vertical reels in a couple of minutes: captions, pauses cut, 9:16 frame.",
    "blocker.title": "Open from Telegram", "blocker.text": "This app runs inside the bot: tap the “Editor” button in the chat.",
    "blocker.failed": "Couldn't open the app",
  },
};

const LANG_NAMES = { ru: "Русский", en: "English" };

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
  fileInfo: "",
  busy: false,
  jobs: [],
  pollTimer: null,
  screen: "edit",
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

function haptic(kind = "selection") {
  if (!tg || !tg.HapticFeedback) return;
  try {
    if (kind === "selection") tg.HapticFeedback.selectionChanged();
    else tg.HapticFeedback.notificationOccurred(kind);
  } catch (_) { /* старый клиент */ }
}

let toastTimer = null;
function toast(text) {
  const el = $("#toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
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

// ------------------------------------------------------------------ тема

const THEME_KEY = "reel-theme";
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const bg = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim();
  if (tg) {
    try { tg.setHeaderColor(bg); tg.setBackgroundColor(bg); tg.setBottomBarColor && tg.setBottomBarColor(bg); } catch (_) { /* старый клиент */ }
  }
}
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch (_) { /* приватный режим */ }
  const initial = saved || (tg && tg.colorScheme === "dark" ? "dark" : "light");
  applyTheme(initial);
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem(THEME_KEY, next); } catch (_) { /* не критично */ }
    applyTheme(next);
    haptic();
  });
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
  const css = [...faces].map(([file, family]) => `@font-face{font-family:"${family}";` +
    `src:url("/fonts/${file}") format("truetype");font-weight:1 1000;font-display:swap;}`).join("\n");
  const tag = document.getElementById("font-faces") || el("style", { id: "font-faces" });
  tag.textContent = css;
  if (!tag.parentNode) document.head.appendChild(tag);
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
    if (panel.enabled && panel.opacity > 0) {
      plate.style.background = `rgba(${hexToRgb(panel.color)}, ${panel.opacity})`;
      plate.style.padding = `${Math.round(panel.padding_y * k)}px ${Math.round(panel.padding_x * k)}px`;
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
    const room = box.clientWidth - (plate.offsetWidth - plate.clientWidth);
    for (const line of plate.children) {
      const over = line.scrollWidth / room;
      if (room > 0 && over > 1) {
        for (const w of line.children) {
          const size = parseFloat(getComputedStyle(w).fontSize);
          w.style.fontSize = `${Math.floor(size / over)}px`;
        }
      }
    }
    $("#preview-tag").textContent = `${p.title} · ${activeFamily()}`;
  },
  tick() {
    this.step += 1;
    this.render();
  },
  start() {
    clearInterval(this.timer);
    this.timer = setInterval(() => { if (!this.paused) this.tick(); }, 280);
    this.render();
  },
  toggle() {
    this.paused = !this.paused;
    $("#preview").classList.toggle("is-paused", this.paused);
    $("#preview-play").setAttribute("aria-label", tr(this.paused ? "preview.play" : "preview.pause"));
  },
};

// ------------------------------------------------------------------ экран «Монтаж»

// Лента перерисовывается целиком, поэтому к активному элементу прокручиваемся
// уже ПОСЛЕ замены: старый узел к этому моменту не в документе.
//
// Двигаем САМУ ленту, а не зовём scrollIntoView: тот бы потянул и страницу по
// вертикали, а на открытии приложения это прыжок экрана ещё до первого
// касания.
function scrollToActive(box, behavior = "smooth") {
  const active = box.querySelector(".is-active");
  if (!active) return;
  const a = active.getBoundingClientRect();
  const b = box.getBoundingClientRect();
  box.scrollBy({ left: a.left - b.left - (b.width - a.width) / 2, behavior });
}

function renderPresets() {
  const box = $("#preset-tabs");
  $("#style-count").textContent = state.boot.presets.length;
  box.replaceChildren(...state.boot.presets.map((p) =>
    el("button", {
      class: `style-card${p.name === state.preset ? " is-active" : ""}`,
      type: "button",
      role: "tab",
      title: `${p.title} · ${p.font}`,
      "aria-selected": p.name === state.preset ? "true" : "false",
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
      // Плитка — это образец: название темы набрано ЕЁ шрифтом и её регистром,
      // под ним полоска её акцентного цвета. Семнадцать одинаковых пилюль с
      // подписями ничего не говорили бы о том, чем темы отличаются.
      el("span", {
        class: "style-card__name",
        text: p.uppercase ? p.title.toUpperCase() : p.title,
        style: `font-family:"${p.font}", system-ui, sans-serif;letter-spacing:${p.letter_spacing}em`,
      }),
      el("i", { class: "style-card__bar", style: `background:${p.colors.accent}` }),
    ])));
  // Длинное название ужимается, а не обрезается многоточием: «MODERNI…» и
  // «HEAVYWEIG…» — это уже не образец шрифта, ради которого плитка и нужна.
  for (const name of box.querySelectorAll(".style-card__name")) {
    const over = name.scrollWidth / name.clientWidth;
    if (name.clientWidth > 0 && over > 1) {
      name.style.fontSize = `${Math.max(11, Math.floor(18 / over))}px`;
    }
  }
}

function renderAccents() {
  const p = state.boot.presets.find((x) => x.name === state.preset);
  const box = $("#accent-chips");
  box.replaceChildren(...state.boot.accents.map((a) =>
    el("button", {
      class: `swatch${a.key === state.accent ? " is-active" : ""}`,
      type: "button",
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
}

function renderFonts() {
  const box = $("#font-chips");
  box.replaceChildren(...(state.boot.fonts || []).map((f) =>
    el("button", {
      class: `chip${f.key === state.font ? " is-active" : ""}`,
      type: "button",
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
}

function renderOptions() {
  for (const input of $$("[data-option]")) input.checked = !!state.options[input.dataset.option];
}

function setFile(file) {
  state.file = file;
  const zone = $("#dropzone");
  zone.classList.toggle("has-file", !!file);
  $("#drop-thumb").style.backgroundImage = "";
  if (!file) {
    $("#drop-title").textContent = tr("edit.upload");
    $("#drop-sub").textContent = ".mp4, .mov, .m4v, .webm, .mkv, .avi";
    $("#file-input").value = "";
    renderCta();
    return;
  }
  $("#drop-title").textContent = file.name || tr("edit.untitled");
  $("#drop-sub").textContent = fmtSize(file.size);
  // Длительность читаем браузером до загрузки: слишком длинное видео лучше
  // отбить сразу, а не после минуты заливки по мобильной сети. Заодно снимаем
  // кадр на обложку: имя файла вроде `IMG_4821.mov` не говорит, то ли это видео.
  const url = URL.createObjectURL(file);
  const probe = document.createElement("video");
  probe.preload = "metadata";
  probe.muted = true;
  probe.playsInline = true;
  const done = () => URL.revokeObjectURL(url);
  probe.onloadedmetadata = () => {
    const d = probe.duration;
    if (state.file !== file) { done(); return; }
    $("#drop-sub").textContent = `${fmtSize(file.size)} · ${fmtDuration(d)}`;
    const lim = state.boot.limits;
    if (isFinite(d) && d > lim.max_minutes * 60 + 1) toast(tr("file.too_long", { max: lim.max_minutes }));
    else if (isFinite(d) && d < lim.min_seconds) toast(tr("file.too_short", { min: lim.min_seconds }));
    try { probe.currentTime = Math.min(0.5, (isFinite(d) ? d : 1) / 2); } catch (_) { done(); }
  };
  probe.onseeked = () => {
    try {
      const side = Math.min(probe.videoWidth, probe.videoHeight);
      if (side && state.file === file) {
        const c = document.createElement("canvas");
        c.width = c.height = 108;
        c.getContext("2d").drawImage(
          probe, (probe.videoWidth - side) / 2, (probe.videoHeight - side) / 2, side, side, 0, 0, 108, 108,
        );
        $("#drop-thumb").style.backgroundImage = `url(${c.toDataURL("image/jpeg", 0.72)})`;
      }
    } catch (_) { /* кодек не отдаёт кадр в canvas — остаётся иконка */ }
    done();
  };
  probe.onerror = done;
  probe.src = url;
  renderCta();
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

  const input = $("#file-input");
  input.addEventListener("change", () => setFile(input.files[0] || null));
  const zone = $("#dropzone");
  zone.addEventListener("click", (ev) => {
    if (state.file && ev.target.closest("#drop-action")) {
      ev.preventDefault();
      if (!state.busy) setFile(null);
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
    if (f && !state.busy) setFile(f);
  });

  for (const input of $$("[data-option]")) {
    input.addEventListener("change", () => { state.options[input.dataset.option] = input.checked; haptic(); });
  }
  for (const btn of $$("[data-help]")) {
    btn.addEventListener("click", () => popup(...tr(`help.${btn.dataset.help}`)));
  }
  $("#cta").addEventListener("click", onCta);
}

function renderBalance() {
  const u = state.boot.user;
  const n = u.balance;
  $("#balance-value").textContent = u.is_admin ? "∞" : n;
  $("#nav-balance").textContent = u.is_admin ? "∞" : n;
  $("#balance-line").textContent = u.is_admin ? tr("balance.unlimited") : `${n} ${videosWord(n)}`;
  $("#balance-big").textContent = u.is_admin ? "∞" : n;
  $("#balance-unit").textContent = u.is_admin ? "" : videosWord(n);
  $("#balance-note").textContent = tr(u.is_admin ? "balance.note_admin" : "balance.note");
  $("#stat-balance").textContent = u.is_admin ? "∞" : n;
  $("#stat-done").textContent = u.videos_done;
  renderCta();
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
  if (!u.is_admin && u.balance < 1) {
    title.textContent = tr("cta.topup");
    sub.textContent = tr("cta.missing");
  } else {
    title.textContent = tr("cta.edit");
    sub.textContent = u.is_admin
      ? tr(state.file ? "cta.free" : "cta.pick")
      : (state.file ? tr("cta.charge", { n: u.balance }) : tr("cta.pick"));
  }
}

async function onCta() {
  if (state.busy) return;
  const u = state.boot.user;
  if (!u.is_admin && u.balance < 1) { go("balance"); return; }
  if (!state.file) { haptic("warning"); $("#file-input").click(); return; }

  state.busy = true;
  try {
    const job = await uploadAndStart(state.file);
    haptic("success");
    setFile(null);
    await refreshBoot();
    toast(tr("toast.accepted"));
    state.jobs = [job, ...state.jobs.filter((j) => j.id !== job.id)];
    go("videos");
  } catch (e) {
    haptic("error");
    if (e.status === 402) { await refreshBoot(); go("balance"); }
    popup(tr("err.title"), e.message);
  } finally {
    state.busy = false;
    renderCta();
  }
}

async function uploadAndStart(file) {
  const start = await api("/api/uploads", { method: "POST", body: { size: file.size, name: file.name } });
  const id = start.upload_id;
  const size = file.size;
  let offset = start.received || 0;
  let failures = 0;
  while (offset < size) {
    const end = Math.min(size, offset + start.chunk_bytes);
    renderCta(tr("cta.uploading", { p: Math.floor((offset / size) * 100) }), offset / size);
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
  renderCta(tr("cta.checking"), 1);
  const res = await api(`/api/uploads/${id}/finish`, {
    method: "POST",
    body: {
      preset: state.preset,
      options: { ...state.options, accent: state.accent, font: state.font },
    },
  });
  return res.job;
}

// ------------------------------------------------------------------ экран «Ролики»

function renderJobs() {
  const box = $("#jobs");
  const active = state.jobs.some((j) => j.status === "queued" || j.status === "running");
  $("#videos-dot").hidden = !active;
  if (!state.jobs.length) {
    box.replaceChildren(el("div", { class: "card empty" }, [
      el("b", { text: tr("jobs.empty") }),
      el("p", { class: "muted", text: tr("jobs.empty_hint") }),
      el("button", { class: "btn-soft", type: "button", text: tr("jobs.to_edit"), onclick: () => go("edit") }),
    ]));
    return;
  }
  const presets = Object.fromEntries(state.boot.presets.map((p) => [p.name, p.title]));
  const retention = state.boot.retention_days > 0
    ? [el("p", { class: "muted jobs__hint", text: tr("jobs.retention", { n: state.boot.retention_days }) })]
    : [];
  box.replaceChildren(...retention, ...state.jobs.map((j) => {
    const date = j.created_at ? new Date(j.created_at).toLocaleString(currentLang() === "en" ? "en-GB" : "ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "";
    const accent = state.boot.accents.find((a) => a.key === (j.options && j.options.accent));
    const meta = [presets[j.preset] || j.preset, accent && accent.key !== "theme" ? accent.title : null, date].filter(Boolean).join(" · ");
    const parts = [
      el("div", { class: "job__head" }, [
        el("div", {}, [el("div", { class: "job__title", text: j.result_duration ? `${tr("jobs.reel")} · ${fmtDuration(j.result_duration)}` : tr("jobs.reel") }), el("div", { class: "muted", text: meta })]),
        el("span", { class: `status status--${j.status}`, text: tr(`status.${j.status}`) }),
      ]),
    ];
    if (j.status === "queued" || j.status === "running") {
      parts.push(el("div", { class: "muted", text: j.stage || "" }));
      parts.push(el("div", { class: "bar" }, [el("i", { style: `width:${Math.round((j.progress || 0) * 100)}%` })]));
    } else if (j.status === "failed") {
      parts.push(el("div", { class: "muted", text: j.error || "" }));
      if (!state.boot.user.is_admin) parts.push(el("div", { class: "muted", text: tr("jobs.refunded") }));
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

async function loadJobs() {
  try {
    const data = await api("/api/jobs");
    const before = JSON.stringify(state.jobs.map((j) => [j.id, j.status, j.progress, j.stage]));
    state.jobs = data.jobs;
    state.boot.user.balance = data.balance;
    renderBalance();
    // Перерисовываем только при изменениях: иначе плеер готового ролика
    // сбрасывался бы каждые три секунды посреди просмотра.
    if (JSON.stringify(state.jobs.map((j) => [j.id, j.status, j.progress, j.stage])) !== before) renderJobs();
  } catch (e) {
    if (state.screen === "videos") toast(e.message);
  }
  schedulePoll();
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  const active = state.jobs.some((j) => j.status === "queued" || j.status === "running");
  $("#videos-dot").hidden = !active;
  if (active) state.pollTimer = setTimeout(loadJobs, 3000);
}

// ------------------------------------------------------------------ экран «Баланс»

function renderPackages() {
  const box = $("#packages");
  box.replaceChildren(...state.boot.packages.map((p) => {
    const per = Math.round(p.stars / p.videos);
    return el("button", { class: "card package", type: "button", onclick: () => buy(p.videos) }, [
      el("div", { class: "package__main" }, [
        el("b", { text: `${p.videos} ${videosWord(p.videos)}` }),
        el("span", { class: "muted", text: p.videos > 1 ? tr("pkg.per", { n: per }) : tr("pkg.try") }),
      ]),
      el("span", { class: "package__price", text: `${p.stars} ⭐` }),
    ]);
  }));
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

// ------------------------------------------------------------------ профиль и приглашение

function renderProfile() {
  const u = state.boot.user;
  const name = u.first_name || u.username || tr("profile.noname");
  $("#profile-name").textContent = name;
  $("#profile-username").textContent = u.username ? `@${u.username}` : "";
  $("#avatar").textContent = name.slice(0, 1).toUpperCase();
  $("#profile-hint").textContent = tr(u.is_admin ? "profile.hint_admin" : "profile.hint");
  renderLanguages();
}

function renderLanguages() {
  const box = $("#lang-tabs");
  const current = currentLang();
  box.replaceChildren(...Object.entries(LANG_NAMES).map(([code, title]) =>
    el("button", {
      class: `tab${code === current ? " is-active" : ""}`,
      type: "button",
      text: title,
      onclick: () => setLanguage(code),
    })));
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
  renderPresets();
  renderAccents();
  renderFonts();
  renderBalance();
  preview.render();
  if (state.file) setFile(state.file); else setFile(null);
  if (state.screen === "videos") renderJobs();
  if (state.screen === "balance") renderPackages();
  if (state.screen === "profile") renderProfile();
}

function invite() {
  const bot = state.boot.bot_username;
  const link = `https://t.me/${bot}`;
  const text = tr("invite.text");
  const share = `https://t.me/share/url?url=${encodeURIComponent(link)}&text=${encodeURIComponent(text)}`;
  haptic();
  if (tg && tg.openTelegramLink) tg.openTelegramLink(share);
  else window.open(share, "_blank");
}

// ------------------------------------------------------------------ навигация

function go(screen) {
  if (screen === "invite") { invite(); return; }
  state.screen = screen;
  for (const s of $$("[data-screen]")) s.hidden = s.dataset.screen !== screen;
  for (const b of $$("[data-nav]")) b.classList.toggle("is-active", b.dataset.nav === screen);
  window.scrollTo({ top: 0 });
  if (tg && tg.BackButton) {
    if (screen === "edit") tg.BackButton.hide(); else tg.BackButton.show();
  }
  if (screen === "videos") { renderJobs(); loadJobs(); }
  if (screen === "balance") renderPackages();
  if (screen === "profile") renderProfile();
}

async function refreshBoot() {
  const fresh = await api("/api/bootstrap");
  state.boot.user = fresh.user;
  renderBalance();
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
  if (tg.BackButton) tg.BackButton.onClick(() => go("edit"));

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
  $("#brand").textContent = b.bot_name || tr("nav.edit");
  applyStaticTexts();
  state.preset = b.presets.some((p) => p.name === b.user.preset) ? b.user.preset : (b.presets[0] || {}).name;
  state.options = { ...b.defaults };
  // Запомненный выбор важнее умолчания: тема уже так и работает (`user.preset`),
  // а цвет со шрифтом до сих пор сбрасывались на каждом открытии приложения.
  const saved = (b.user && b.user.settings) || {};
  state.accent = saved.accent || b.defaults.accent || "theme";
  state.font = saved.font || b.defaults.font || "theme";

  declareFonts();
  renderPresets();
  renderAccents();
  renderFonts();
  // Запомненная тема может быть семнадцатой в ленте: без подкрутки человек
  // видел бы начало списка и ни одной выбранной плитки.
  for (const id of ["#preset-tabs", "#font-chips", "#accent-chips"]) {
    scrollToActive($(id), "instant");
  }
  renderOptions();
  renderBalance();
  initEdit();
  preview.start();
  // Ширина строки мерится шрифтом темы — до его загрузки замер врёт.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => preview.render());

  for (const btn of $$("[data-nav]")) btn.addEventListener("click", () => { haptic(); go(btn.dataset.nav); });
  $("#balance-pill").addEventListener("click", () => go("balance"));

  // Вкладка приходит параметром `?screen=`, а не фрагментом: фрагмент занят
  // Telegram — в нём едут параметры запуска (`#tgWebAppData=…`).
  const start = new URLSearchParams(location.search).get("screen") || "edit";
  go(["edit", "videos", "balance", "profile"].includes(start) ? start : "edit");
  loadJobs();
}

main();
