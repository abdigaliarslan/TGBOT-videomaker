/**
 * Покадровый прогон появления СЛОВА и БЛОКА субтитров.
 *
 * Считает ровно ту же арифметику, что `layers/Captions.tsx`, и печатает по
 * кадрам альфу слова, его подъём и «пшик» масштаба.
 *
 *     node render/caption_probe.mjs [fps]          # для глаз
 *     node render/caption_probe.mjs [fps] --json   # для tests/test_motion.py
 *
 * Зачем: у картинок такой пробник есть с М5, а у субтитров не было, и любое
 * число про их анимацию нечем было подтвердить. Эталон меряется кадрами
 * (блок — 3 кадра альфы, слово — 4), и проверять наши темы надо тем же.
 *
 * Числа читаются из САМИХ пресетов: замер обязан отвечать за то, что реально
 * пойдёт в рендер, а не за копию чисел в этом файле.
 */
import { readdirSync, readFileSync } from "node:fs";
import { interpolate } from "remotion";

const MIN_ANIM_FRAMES = 8;
const clamp01 = (t) => Math.max(0, Math.min(1, t));
const easeOut = (t) => 1 - Math.pow(1 - clamp01(t), 3);
const animFrames = (ms, fps) => Math.max(MIN_ANIM_FRAMES, Math.round((ms / 1000) * fps));
const fadeFrames = (ms, fps) => Math.max(1, Math.round((ms / 1000) * fps));
const smoothStep = (t) => {
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
};
const popBump = (q, rise = 0.45) => {
  if (q <= 0 || q >= 1) return 0;
  return q < rise ? smoothStep(q / rise) : 1 - smoothStep((q - rise) / (1 - rise));
};
const ramp = (f, span) =>
  interpolate(f, [0, span], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

const PRESETS = new URL("../src/core/presets/", import.meta.url);
const presets = Object.fromEntries(
  readdirSync(PRESETS)
    .filter((f) => f.endsWith(".json"))
    .map((f) => [f.replace(/\.json$/, ""), JSON.parse(readFileSync(new URL(f, PRESETS), "utf-8"))]),
);

const fps = Number(process.argv[2] || 30);
const asJson = process.argv.includes("--json");
const report = {};

for (const [name, preset] of Object.entries(presets)) {
  const a = preset.animation ?? {};
  const colours = preset.colors ?? {};
  const ghost = colours.ghost_opacity ?? 0;
  const num = (key, fallback) => (typeof a[key] === "number" ? a[key] : fallback);

  const wordIn = num("word_in_ms", 0);
  const revealSpan = wordIn > 0 ? fadeFrames(wordIn, fps) : 0;
  const risePx = num("word_rise_px", 0);
  const popSpan = animFrames(num("pop_ms", 160), fps);
  const popScale = num("pop_scale", 1);

  // Слово: столько кадров, чтобы «пшик» успел кончиться.
  const wordRows = [];
  for (let f = 0; f <= popSpan + 2; f++) {
    const revealed = revealSpan > 0 ? ramp(f, revealSpan) : 1;
    wordRows.push({
      f,
      alpha: ghost + (1 - ghost) * revealed,
      rise: risePx * (1 - revealed),
      scale: 1 + (popScale - 1) * popBump(f / popSpan),
    });
  }
  // Проявление кончилось = первый кадр, на котором альфа взяла единицу.
  const revealFrames = wordRows.findIndex((r) => r.alpha >= 0.999);
  const popPeak = Math.max(...wordRows.map((r) => r.scale));

  // Блок: две ветки Captions.tsx — с паузой между блоками и с нахлёстом.
  const gap = num("block_gap_ms", 0);
  const leavesEarly = gap > 0;
  const inSpan = leavesEarly
    ? fadeFrames(num("block_in_ms", 180), fps)
    : animFrames(num("block_in_ms", 180), fps);
  const zoom = num("block_in_zoom", 0.035);
  const shift = num("block_shift_px", 24);
  const blockRows = [];
  for (let f = 0; f <= inSpan + 2; f++) {
    const rise = ramp(f, inSpan);
    const appear = leavesEarly ? rise : easeOut(rise);
    blockRows.push({ f, alpha: appear, zoom: 1 - zoom + appear * zoom, offset: (1 - appear) * shift });
  }

  report[name] = {
    fps,
    word_reveal_frames: revealFrames,
    word_rise_px: risePx,
    word_pop_peak: popPeak,
    block_in_frames: inSpan,
    block_leaves_early: leavesEarly,
    word_alpha: wordRows.map((r) => r.alpha),
    word_rise: wordRows.map((r) => r.rise),
    word_scale: wordRows.map((r) => r.scale),
    block_alpha: blockRows.map((r) => r.alpha),
    block_zoom: blockRows.map((r) => r.zoom),
  };
  if (asJson) continue;

  console.log(
    `${name} @${fps}fps: слово проявляется ${revealFrames} кадров, подъём ${risePx} px, ` +
      `пик «пшика» ${popPeak.toFixed(3)}, блок входит ${inSpan} кадров` +
      `${leavesEarly ? " (с паузой между блоками)" : " (с нахлёстом)"}`,
  );
  console.log("   альфа слова:", wordRows.map((r) => r.alpha.toFixed(3)).join(" "));
  if (risePx) console.log("   подъём:     ", wordRows.map((r) => r.rise.toFixed(1)).join(" "));
  console.log("   альфа блока:", blockRows.map((r) => r.alpha.toFixed(3)).join(" "));
}

if (asJson) console.log(JSON.stringify(report));
