/**
 * Покадровый прогон входа полноэкранной b-roll вставки — DoD М5.
 *
 * Считает ровно ту же арифметику, что `layers/Broll.tsx`, и печатает по кадрам
 * альфу, масштаб наезда и расфокус.
 *
 *     node render/card_probe.mjs [fps]          # для глаз
 *     node render/card_probe.mjs [fps] --json   # для tests/test_motion.py
 *
 * Числа отсюда закреплены тестом: длительность входа живёт в пресете, и правка
 * пресета не должна молча вернуть вход в ступеньку, от которой его уводили.
 */
import { readdirSync, readFileSync } from "node:fs";
import { Easing, interpolate } from "remotion";

const MIN_ANIM_FRAMES = 8;
const animFrames = (ms, fps) => Math.max(MIN_ANIM_FRAMES, Math.round((ms / 1000) * fps));

// Копия констант Broll.tsx. Разойдутся — тест поймает это по кадрам, а не по
// тексту: пробник и рендер обязаны давать одну кривую.
const ENTER_MS = 600;
const ENTER_BLUR = 0.012;
const FULL_PUSH = 1.08;
const SOFT = Easing.bezier(0.33, 0, 0.25, 1);
const SETTLE = Easing.bezier(0.16, 1, 0.3, 1);
const PUSH = Easing.bezier(0.33, 0, 0.2, 1);
const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" };
const HEIGHT = 1920;

// Числа входа читаются из САМИХ пресетов, а не переписываются сюда: иначе
// замер отвечает за копию, а рендер идёт по оригиналу.
const PRESETS = new URL("../src/core/presets/", import.meta.url);
const presets = Object.fromEntries(
  readdirSync(PRESETS)
    .filter((f) => f.endsWith(".json"))
    .map((f) => {
      const a = JSON.parse(readFileSync(new URL(f, PRESETS), "utf-8")).animation ?? {};
      return [
        f.replace(/\.json$/, ""),
        {
          inMs: typeof a.broll_in_ms === "number" ? a.broll_in_ms : ENTER_MS,
          blur: typeof a.broll_in_blur === "number" ? a.broll_in_blur : ENTER_BLUR,
        },
      ];
    }),
);

const fps = Number(process.argv[2] || 30);
const asJson = process.argv.includes("--json");
const report = {};

for (const [name, cfg] of Object.entries(presets)) {
  const enter = animFrames(cfg.inMs, fps);
  const rows = [];
  for (let f = 0; f <= enter * 2 + 4; f++) {
    rows.push({
      f,
      alpha: interpolate(f, [0, enter], [0, 1], { ...clamp, easing: SOFT }),
      scale: interpolate(f, [0, enter * 2], [FULL_PUSH, 1], { ...clamp, easing: PUSH }),
      blur:
        HEIGHT * cfg.blur * interpolate(f, [0, enter], [1, 0], { ...clamp, easing: SETTLE }),
    });
  }

  // «Видимое движение» — кадры, на которых масштаб меняется больше чем на 0.2 %
  // от полного хода: ниже этого порога шаг меньше пикселя.
  const step = (FULL_PUSH - 1) * 0.002;
  const moving = rows.filter((r, i) => i > 0 && Math.abs(r.scale - rows[i - 1].scale) > step).length;
  const alphaSteps = rows.slice(1).map((r, i) => r.alpha - rows[i].alpha);
  const scaleSteps = rows.slice(1).map((r, i) => r.scale - rows[i].scale);
  // Сел = последний кадр, где наезду осталось больше 2 % хода, плюс один.
  let last = 0;
  rows.forEach((r, i) => {
    if (r.scale - 1 >= 0.02 * (FULL_PUSH - 1)) last = i;
  });
  const alphaNine = rows.findIndex((r) => r.alpha >= 0.9);

  report[name] = {
    fps,
    enter_frames: enter,
    moving_frames: moving,
    // Насколько наезд проскочил единицу вниз: у плавного входа — ноль.
    overshoot_pct: Math.max(0, (1 - Math.min(...rows.map((r) => r.scale))) * 100),
    alpha_max_step: Math.max(...alphaSteps),
    alpha_monotonic: alphaSteps.every((d) => d >= -1e-9),
    scale_monotonic: scaleSteps.every((d) => d <= 1e-9),
    alpha_full_frame: rows.findIndex((r) => r.alpha >= 0.999),
    // Какая доля наезда ещё впереди, когда картинка уже видна (альфа 0.9).
    push_left_when_visible: (rows[alphaNine].scale - 1) / (FULL_PUSH - 1),
    scale_settled_frame: last + 1,
    blur_gone_frame: rows.findIndex((r) => r.blur < 0.05),
    scale: rows.map((r) => r.scale),
    alpha: rows.map((r) => r.alpha),
    blur: rows.map((r) => r.blur),
  };
  if (asJson) continue;

  const r = report[name];
  console.log(
    `${name} @${fps}fps: вход ${enter} кадров, видимого движения ${moving}, ` +
      `крупнейший шаг альфы ${r.alpha_max_step.toFixed(3)}, альфа=1 на кадре ${r.alpha_full_frame}, ` +
      `фокус на ${r.blur_gone_frame}, наезд сел на ${r.scale_settled_frame}`,
  );
  console.log("   альфа:  ", rows.map((x) => x.alpha.toFixed(3)).join(" "));
  console.log("   масштаб:", rows.map((x) => x.scale.toFixed(3)).join(" "));
  console.log("   блюр px:", rows.map((x) => x.blur.toFixed(1)).join(" "));
}

if (asJson) console.log(JSON.stringify(report));
