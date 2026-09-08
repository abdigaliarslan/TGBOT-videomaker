/**
 * Покадровый прогон входа b-roll карточки — DoD М5.
 *
 * Считает ровно ту же арифметику, что `layers/Broll.tsx`, и печатает по кадрам
 * альфу, масштаб и прирост каждого.
 *
 *     node render/card_probe.mjs [fps]          # для глаз
 *     node render/card_probe.mjs [fps] --json   # для tests/test_motion.py
 *
 * Числа отсюда закреплены тестом: пружина живёт в пресете, и правка пресета не
 * должна молча вернуть вход в три кадра, ради которого М5 и написан.
 */
import { readdirSync, readFileSync } from "node:fs";
import { measureSpring, spring, interpolate } from "remotion";

const MIN_ANIM_FRAMES = 8;
const easeOut = (t) => 1 - Math.pow(1 - Math.max(0, Math.min(1, t)), 3);
const animFrames = (ms, fps) => Math.max(MIN_ANIM_FRAMES, Math.round((ms / 1000) * fps));

const FADE_MS = 220;
const INSET_POP = 0.72;

// Пружина читается из САМИХ пресетов, а не переписывается сюда числами: иначе
// замер отвечает за копию, а рендер идёт по оригиналу, и правка пресета
// проходит мимо проверки.
const PRESETS = new URL("../src/core/presets/", import.meta.url);
const presets = Object.fromEntries(
  readdirSync(PRESETS)
    .filter((f) => f.endsWith(".json"))
    .map((f) => {
      const a = JSON.parse(readFileSync(new URL(f, PRESETS), "utf-8")).animation ?? {};
      return [
        f.replace(/\.json$/, ""),
        { damping: a.spring_damping, stiffness: a.spring_stiffness, mass: a.spring_mass },
      ];
    })
    .filter(([, c]) => Object.values(c).every((v) => typeof v === "number")),
);

const fps = Number(process.argv[2] || 30);
const asJson = process.argv.includes("--json");
const report = {};

for (const [name, config] of Object.entries(presets)) {
  const dur = Math.max(MIN_ANIM_FRAMES, measureSpring({ fps, config }));
  const fade = animFrames(FADE_MS, fps);
  const rows = [];
  for (let f = 0; f <= Math.ceil(dur) + 6; f++) {
    const enter = spring({ frame: f, fps, durationInFrames: dur, config });
    const alpha = easeOut(
      interpolate(f, [0, fade], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
    );
    rows.push({ f, scale: interpolate(enter, [0, 1], [INSET_POP, 1]), alpha });
  }

  const peak = Math.max(...rows.map((r) => r.scale));
  const overshoot = (peak - 1) * 100;
  // «Видимое движение» — кадры, на которых масштаб меняется больше чем на 0.2 %
  // от полного хода: ниже этого порога шаг меньше пикселя на карточке 475 px.
  const step = (1 - INSET_POP) * 0.002;
  const moving = rows.filter((r, i) => i > 0 && Math.abs(r.scale - rows[i - 1].scale) > step).length;
  const alphaFull = rows.findIndex((r) => r.alpha >= 0.999);
  // Успокоился = ПОСЛЕДНИЙ кадр, на котором масштаб ещё гулял, плюс один.
  // «Первый кадр в допуске» здесь врёт: пружина проходит единицу на подъёме и
  // на спаде перелёта, и на 60 fps такой замер объявлял посадку на 150 мс,
  // когда карточка только начинала перелёт.
  let last = 0;
  rows.forEach((r, i) => {
    if (Math.abs(r.scale - 1) >= 0.002) last = i;
  });
  const scaleSettled = last + 1;

  report[name] = {
    fps,
    spring_frames: dur,
    moving_frames: moving,
    overshoot_pct: overshoot,
    alpha_full_frame: alphaFull,
    scale_settled_frame: scaleSettled,
    scale: rows.map((r) => r.scale),
    alpha: rows.map((r) => r.alpha),
  };
  if (asJson) continue;

  console.log(
    `${name} @${fps}fps: пружина ${dur.toFixed(1)} кадра, видимого движения ${moving} кадров, ` +
      `перелёт ${overshoot.toFixed(2)} %, альфа=1 на кадре ${alphaFull}, масштаб сел на ${scaleSettled}`,
  );
  console.log(
    "   масштаб:",
    rows.map((r) => r.scale.toFixed(3)).join(" "),
  );
  console.log(
    "   альфа:  ",
    rows.map((r) => r.alpha.toFixed(3)).join(" "),
  );
}

if (asJson) console.log(JSON.stringify(report));
