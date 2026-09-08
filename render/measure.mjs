#!/usr/bin/env node
/**
 * Линейка: JSON на stdin → измеренные ширины строк на stdout.
 *
 * Нужна ровно для одного — проверить, что `src/core/geometry.py` предсказывает
 * ширину строки так же, как её нарисует Chromium (DoD Ф4). Меряет тот же
 * headless-браузер, которым рендерится ролик, и тем же шрифтом из
 * `assets/fonts`, поэтому расхождение здесь — это расхождение в кадре, а не
 * в эмуляции.
 *
 * Модель строки повторяет DOM субтитра: строка — один инлайновый бокс, слово —
 * спан со своим кеглем, между словами НАСТОЯЩИЙ пробел кеглем строки. Кернинг
 * внутри слова Chromium считает сам, через границу спана — не считает.
 *
 * Формат stdin:
 * {
 *   "font":     "/abs/Montserrat-ExtraBold.ttf",
 *   "weight":   800,
 *   "base_size": 88,
 *   "letter_spacing": -0.015,          // em, как в пресете
 *   "items": [                          // строки
 *     { "id": "b0l0", "words": [{ "text": "Я", "size": 88 }, ...] },
 *     { "id": "word", "text": "думаю", "size": 104 },
 *     { "id": "emo",  "words": [{ "text": "огонь", "size": 88, "emoji": "🔥" }] }
 *   ],
 *   "boxes": [                          // блоки целиком, вместе с панелью
 *     { "id": "b0", "lines": [[{ "text": "Я", "size": 88 }]],
 *       "gap": 5, "padding_x": 34, "padding_y": 22, "border_width": 1,
 *       "line_height": 1.06, "max_width": 842.4 }
 *   ]
 * }
 *
 * Формат stdout:
 * { "widths": { "<id>": 512.34 }, "heights": { "<id>": 93.28 },
 *   "parts": { "<id>": [..] },
 *   "boxes": { "<id>": { "width": 912.4, "height": 254.35 } } }
 *
 * `heights` — высота строчного бокса строки. Ширины мало: по высоте строки
 * складывается высота панели, а её предсказывает та же геометрия.
 *
 * Запуск: node render/measure.mjs < request.json
 */

import { openBrowser } from "@remotion/renderer";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const log = (...args) => console.error("[measure]", ...args);

const readStdin = () =>
  new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => (data += chunk));
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });

/**
 * Шрифт уезжает в страницу как data-URI, а не ссылкой на файл: страница живёт
 * по file://, а Chromium без отдельного флага не пускает file:// к соседним
 * файлам, и @font-face молча не сработает — измерения окажутся о системном
 * шрифте.
 */
const buildPage = (req) => {
  const base64 = fs.readFileSync(req.font).toString("base64");
  const weight = req.weight ?? 400;
  const letterSpacing = req.letter_spacing ?? 0;
  const baseSize = req.base_size ?? 100;
  return `<!doctype html>
<meta charset="utf-8">
<style>
  @font-face {
    font-family: "ReelbotMeasure";
    src: url(data:font/ttf;base64,${base64}) format("truetype");
    /* Диапазон, а не одно значение: иначе Chromium синтезирует жирность сам
       и ширина уедет на пару процентов. */
    font-weight: 1 1000;
    font-style: normal;
  }
  html, body { margin: 0; padding: 0; }
  #stage {
    position: absolute; top: 0; left: 0;
    font-family: "ReelbotMeasure";
    font-weight: ${weight};
    font-size: ${baseSize}px;
    line-height: ${req.line_height ?? "normal"};
    letter-spacing: ${letterSpacing}em;
  }
  /* content-box намеренно: max-width в Python — это бюджет ТЕКСТА, паддинги
     панели прибавляются к нему, а не съедают его. */
  .line, .panel { box-sizing: content-box; }
  .line { white-space: nowrap; }
  /* Только спан СЛОВА, не всё подряд внутри строки: у эмодзи внутри слова свой
     вложенный спан, и он в Captions.tsx инлайновый. Сделать его inline-block —
     значит схлопнуть пробел перед эмодзи по краю бокса и намерить на 24 px
     меньше, чем будет в кадре. */
  .line > span { display: inline-block; }
  .panel { display: inline-flex; flex-direction: column; align-items: center; }
</style>
<div id="stage"></div>
`;
};

/**
 * Всё измерение целиком уезжает в страницу одной функцией: `evaluate`
 * сериализует её текстом, и ничего из области видимости Node внутри не будет.
 */
const measureInPage = (items, boxes, weight, baseSize, lineHeight) => {
  /** Строка: спаны слов и настоящие пробелы между ними, без переносов. */
  const buildLine = (words, inline) => {
    const line = document.createElement("div");
    line.className = "line";
    if (inline) line.style.display = "inline-block";
    const spans = [];
    words.forEach((word, i) => {
      if (i > 0) line.appendChild(document.createTextNode(" "));
      const span = document.createElement("span");
      span.textContent = word.text;
      if (word.size) span.style.fontSize = `${word.size}px`;
      span.style.fontWeight = String(weight);
      // Эмодзи лежит ВНУТРИ спана слова, отдельным вложенным спаном и с
      // пробелом перед ним — ровно так его рисует Captions.tsx. Снаружи мерить
      // нельзя: у вложенного спана свой шейпинг, а кегль он наследует от слова.
      if (word.emoji) {
        const emoji = document.createElement("span");
        emoji.textContent = ` ${word.emoji}`;
        span.appendChild(emoji);
      }
      line.appendChild(span);
      spans.push(span);
    });
    return { line, spans };
  };

  const stage = document.getElementById("stage");
  const widths = {};
  const heights = {};
  const parts = {};
  for (const item of items) {
    const words = item.words ?? [{ text: item.text, size: item.size, emoji: item.emoji }];
    const { line, spans } = buildLine(words, true);
    stage.appendChild(line);
    const box = line.getBoundingClientRect();
    widths[item.id] = box.width;
    heights[item.id] = box.height;
    parts[item.id] = spans.map((s) => s.getBoundingClientRect().width);
    stage.removeChild(line);
  }

  const measured = {};
  for (const box of boxes) {
    const panel = document.createElement("div");
    panel.className = "panel";
    // Просвет между строками по умолчанию считает САМА страница — той же
    // формулой, что Captions.tsx (`Math.round(font.size * (line_height - 1))`).
    // Взять его готовым числом из запроса значит подставить в обе стороны
    // сверки одно и то же вычисление: ошибка в нём сократилась бы и тест
    // прошёл бы при разъехавшемся кадре.
    const gap =
      box.gap ?? Math.round(baseSize * ((box.line_height ?? lineHeight ?? 1) - 1));
    panel.style.gap = `${gap}px`;
    panel.style.padding = `${box.padding_y ?? 0}px ${box.padding_x ?? 0}px`;
    panel.style.border = `${box.border_width ?? 0}px solid transparent`;
    if (box.max_width) panel.style.maxWidth = `${box.max_width}px`;
    for (const words of box.lines) panel.appendChild(buildLine(words, false).line);
    stage.appendChild(panel);
    const rect = panel.getBoundingClientRect();
    measured[box.id] = { width: rect.width, height: rect.height };
    stage.removeChild(panel);
  }
  return { widths, heights, parts, boxes: measured };
};

const main = async () => {
  const raw = await readStdin();
  if (!raw.trim()) throw new Error("на stdin пусто: ожидался JSON с запросом на измерение");
  const req = JSON.parse(raw);
  if (!req.font) throw new Error("в запросе нет поля font (путь к TTF)");
  if (!fs.existsSync(req.font)) throw new Error(`шрифт не найден: ${req.font}`);
  const items = req.items ?? [];
  const boxes = req.boxes ?? [];

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "reelbot-measure-"));
  const page = path.join(dir, "measure.html");
  fs.writeFileSync(page, buildPage(req), "utf8");

  const browser = await openBrowser("chrome", { browserExecutable: null });
  try {
    const tab = await browser.newPage({
      context: null,
      logLevel: "error",
      indent: false,
      pageIndex: 0,
    });
    await tab.goto({ url: `file://${page}`, timeout: 30000 });
    // Шрифт из data-URI грузится не мгновенно; без ожидания первое измерение
    // придётся на фолбэк.
    await tab.evaluate(
      (family) => document.fonts.load(`100px "${family}"`).then(() => document.fonts.ready),
      "ReelbotMeasure",
    );
    const result = await tab.evaluate(
      measureInPage,
      items,
      boxes,
      req.weight ?? 400,
      req.base_size ?? 100,
      req.line_height ?? 1,
    );
    process.stdout.write(JSON.stringify(result) + "\n");
  } finally {
    await browser.close({ silent: true });
    fs.rmSync(dir, { recursive: true, force: true });
  }
};

main().catch((err) => {
  log("ОШИБКА:", err?.stack || err?.message || String(err));
  process.exit(1);
});
