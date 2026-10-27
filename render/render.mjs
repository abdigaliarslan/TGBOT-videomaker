#!/usr/bin/env node
/**
 * CLI рендера: JSON на stdin → путь к mp4 на stdout.
 *
 * Node только рендерит. Никакого HTTP-сервиса между Python и Node — это лишняя
 * точка отказа (§1). Всё состояние приходит в одном JSON.
 *
 * Формат stdin:
 * {
 *   "plan":       { ...RenderPlan... },  // пути внутри — имена файлов в assetsDir
 *   "assetsDir":  "/abs/job/dir",        // монтируется в бандл как media/<key>
 *   "out":        "/abs/out.mp4",
 *   "concurrency": 2,
 *   "crf": 18,
 *   "jobKey": "job-123"
 * }
 *
 * Служебные режимы:
 *   node render.mjs --ensure-browser   скачать Chromium заранее (на старте воркера)
 *   node render.mjs --bundle           только собрать бандл
 */

import { bundle } from "@remotion/bundler";
import { ensureBrowser, renderMedia, selectComposition } from "@remotion/renderer";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ENTRY = path.join(HERE, "src", "index.ts");
const BUNDLE_DIR = path.join(HERE, ".cache", "bundle");
// staticFile() резолвится в /public/<path>, поэтому монтируем именно в public.
const MEDIA_DIR = path.join(BUNDLE_DIR, "public", "media");
const STAMP = path.join(HERE, ".cache", "bundle.stamp");

const log = (...args) => console.error("[render]", ...args);

/** Самая свежая правка исходников рендера — по ней решаем, пересобирать ли бандл. */
const sourcesMtime = () => {
  let newest = 0;
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(p);
      else newest = Math.max(newest, fs.statSync(p).mtimeMs);
    }
  };
  walk(path.join(HERE, "src"));
  newest = Math.max(newest, fs.statSync(path.join(HERE, "package.json")).mtimeMs);
  return newest;
};

const ensureBundle = async () => {
  const newest = sourcesMtime();
  const stamped = fs.existsSync(STAMP) ? Number(fs.readFileSync(STAMP, "utf8")) : -1;
  if (fs.existsSync(path.join(BUNDLE_DIR, "index.html")) && stamped >= newest) {
    return BUNDLE_DIR;
  }
  log("собираю бандл…");
  fs.rmSync(BUNDLE_DIR, { recursive: true, force: true });
  const started = Date.now();
  await bundle({
    entryPoint: ENTRY,
    outDir: BUNDLE_DIR,
    onProgress: () => {},
    webpackOverride: (c) => c,
  });
  fs.mkdirSync(path.dirname(STAMP), { recursive: true });
  fs.writeFileSync(STAMP, String(newest));
  log(`бандл готов за ${((Date.now() - started) / 1000).toFixed(1)}с`);
  return BUNDLE_DIR;
};

/** Ассеты джоба монтируются в бандл симлинком: копировать гигабайты незачем. */
const mountAssets = (assetsDir, jobKey) => {
  fs.mkdirSync(MEDIA_DIR, { recursive: true });
  const link = path.join(MEDIA_DIR, jobKey);
  try {
    if (fs.existsSync(link) || fs.lstatSync(link)) fs.rmSync(link, { recursive: true, force: true });
  } catch {
    /* ссылки ещё нет */
  }
  fs.symlinkSync(path.resolve(assetsDir), link, "dir");
  return `media/${jobKey}`;
};

const unmountAssets = (jobKey) => {
  const link = path.join(MEDIA_DIR, jobKey);
  try {
    fs.rmSync(link, { recursive: true, force: true });
  } catch {
    /* уже убрано */
  }
};

const withPrefix = (prefix, value) => (value ? `${prefix}/${value}` : value);

const readStdin = () =>
  new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => (data += chunk));
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });

const main = async () => {
  const args = process.argv.slice(2);

  if (args.includes("--ensure-browser")) {
    // Вызывается один раз при старте воркера, а не на каждый рендер (§5 M5 ⚠).
    await ensureBrowser();
    log("браузер готов");
    return;
  }
  if (args.includes("--bundle")) {
    await ensureBundle();
    return;
  }

  const raw = await readStdin();
  if (!raw.trim()) throw new Error("на stdin пусто: ожидался JSON с планом рендера");
  const req = JSON.parse(raw);
  const plan = req.plan;
  if (!plan) throw new Error("в запросе нет поля plan");
  const out = req.out;
  if (!out) throw new Error("в запросе нет поля out");
  const jobKey = req.jobKey || `job-${process.pid}-${Math.floor(Math.random() * 1e6)}`;

  const serveUrl = await ensureBundle();
  await ensureBrowser();

  let prefix = "";
  if (req.assetsDir) prefix = mountAssets(req.assetsDir, jobKey);

  try {
    const inputProps = {
      ...plan,
      video: withPrefix(prefix, plan.video),
      audio: withPrefix(prefix, plan.audio),
      fontUrl: withPrefix(prefix, plan.fontUrl),
      broll: (plan.broll ?? []).map((b) => ({ ...b, asset: withPrefix(prefix, b.asset) })),
    };

    const composition = await selectComposition({
      serveUrl,
      id: "Reel",
      inputProps,
    });

    fs.mkdirSync(path.dirname(out), { recursive: true });
    const started = Date.now();
    let lastLogged = 0;

    await renderMedia({
      composition,
      serveUrl,
      codec: "h264",
      outputLocation: out,
      inputProps,
      crf: req.crf ?? 18,
      x264Preset: req.x264Preset ?? "medium",
      pixelFormat: "yuv420p",
      concurrency: req.concurrency ?? null,
      // Режим GL — число настройки (RENDER_GL), а не константа: на сервере без
      // GPU программная эмуляция WebGL стоит времени на каждом кадре, а слоям из
      // HTML и CSS WebGL не нужен. Не задан — прежний "swangle".
      chromiumOptions: { gl: req.gl ?? "swangle" },
      // Дефолтные 30 с на кадр — это не «кадр завис», а «машина занята». Замерено
      // на этой: при параллельном прогоне тестов рендер упал на кадре 221 с
      // «Timeout (30000ms) exceeded», не дождавшись куска ОСНОВНОГО видео с
      // собственного прокси-сервера. М4 делает это вероятнее: видео-вставка —
      // второй поток, который тянется оттуда же.
      //
      // Верхнюю границу держит вызывающий Python (таймаут процесса), поэтому
      // здесь порог нужен только чтобы отличить зависший кадр от медленного.
      // Две минуты на кадр — это заведомо зависший.
      timeoutInMilliseconds: req.timeoutMs ?? 120_000,
      // Строка `[render] progress N` — контракт с Python (`render._PROGRESS`):
      // по ней двигается полоса прогресса в чате и приложении. Шаг 2 %: рендер
      // бывает самым долгим этапом, и без этого полоса минутами стоит на месте.
      onProgress: ({ progress }) => {
        const pct = Math.round(progress * 100);
        if (pct >= lastLogged + 2) {
          lastLogged = pct;
          log(`progress ${pct}`);
        }
      },
    });

    const seconds = (Date.now() - started) / 1000;
    log(
      `готово за ${seconds.toFixed(1)}с (${(composition.durationInFrames / composition.fps).toFixed(1)}с видео, ` +
        `${(seconds / (composition.durationInFrames / composition.fps)).toFixed(2)}x)`,
    );
    process.stdout.write(out + "\n");
  } finally {
    if (prefix) unmountAssets(jobKey);
  }
};

main().catch((err) => {
  log("ОШИБКА:", err?.stack || err?.message || String(err));
  process.exit(1);
});
