/**
 * Покадровый прогон входа b-roll вставки — DoD М5.
 *
 * Считает ровно ту же арифметику, что `layers/Broll.tsx`, и печатает по кадрам
 * альфу, масштаб и расфокус для ТРЁХ путей входа, потому что они разные:
 *
 * - полный кадр: наезд `FULL_PUSH` 1.08 → 1, монотонный, без перелёта;
 * - карточка: подъём `CARD_POP` 0.9 → 1 плюс горка перелёта `broll_overshoot`;
 * - угловая вставка: тот же подъём, но с `INSET_POP` 0.72 — ход вдвое длиннее.
 *
 * Один путь другим не подменяется: у полного кадра перелёт означает масштаб
 * ниже единицы и открытые края исходного видео, у карточки и угловой вставки —
 * заказанную упругость. Поэтому замеров тоже три.
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
// Копия `style.ts`: горка 0 → 1 → 0, обе половины `smoothStep`.
const smoothStep = (t) => {
  const x = Math.max(0, Math.min(1, t));
  return x * x * (3 - 2 * x);
};
const popBump = (q, rise = 0.45) => {
  if (q <= 0 || q >= 1) return 0;
  return q < rise ? smoothStep(q / rise) : 1 - smoothStep((q - rise) / (1 - rise));
};

// Копия констант Broll.tsx — импортировать их нечем: TSX с хуками Remotion вне
// композиции не исполняется, а числа нужны пробнику до всякого рендера.
//
// По КАДРАМ расхождение этой копии с рендером не ловится, и прежний комментарий
// здесь утверждал обратное: кадры считает сам пробник, поэтому правка Broll.tsx
// его замеры не меняет вообще. Проверено: удаление `+ bump` из обеих ветвей
// масштаба в Broll.tsx оставило весь набор зелёным.
//
// Копию сторожит `tests/test_probe_drift.py` — сверкой ПО ТЕКСТУ обоих файлов.
// Гарантия ровно одна: разойдутся числа, безье или тело `popBump` — упадёт он и
// назовёт, что именно разошлось. Правка самой формы кривой в Broll.tsx мимо
// этих констант по-прежнему не ловится ничем, кроме глаз.
const ENTER_MS = 600;
const ENTER_BLUR = 0.012;
const FULL_PUSH = 1.08;
const CARD_POP = 0.9;
const INSET_POP = 0.72;
const OVERSHOOT_RISE = 0.75;
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
          overshoot: typeof a.broll_overshoot === "number" ? a.broll_overshoot : 0,
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
    // Путь КАРТОЧКИ: монотонный подъём с 0.9 по `SETTLE` плюс горка перелёта.
    // Окно горки — то же окно входа, поэтому к его концу она уже ноль и
    // карточка садится ровно на единицу.
    const settle = interpolate(f, [0, enter], [0, 1], { ...clamp, easing: SETTLE });
    // Та же горка перелёта, что в Broll.tsx отдельным слагаемым `bump`: обе
    // ветви подъёма (карточка и угловая вставка) получают её одинаково.
    const bump = cfg.overshoot * popBump(f / enter, OVERSHOOT_RISE);
    rows.push({
      f,
      alpha: interpolate(f, [0, enter], [0, 1], { ...clamp, easing: SOFT }),
      scale: interpolate(f, [0, enter * 2], [FULL_PUSH, 1], { ...clamp, easing: PUSH }),
      card: CARD_POP + (1 - CARD_POP) * settle + bump,
      // Путь УГЛОВОЙ ВСТАВКИ отличается только полом подъёма, зато ход у него
      // вдвое длиннее (0.72 против 0.9) — на нём и перелёт, и ступенька заметны
      // раньше, чем на карточке.
      inset: INSET_POP + (1 - INSET_POP) * settle + bump,
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

  // Перелёт подъёма: насколько масштаб заходит ЗА единицу вверх. Пружина давала
  // здесь 4-7 % и подпрыгивание, число пресета держит 2-4 %. Плюс сколько раз
  // масштаб менял направление: у горки разворот ровно один (вверх, потом вниз на
  // место), у пружины их столько, сколько у неё колебаний — отличить одно от
  // другого по величине перелёта нельзя, только по этому.
  //
  // Считается одной функцией для обеих ветвей: у карточки и угловой вставки
  // кривая одна, и мерить их разным кодом значило бы снова держать копию.
  const rise = (key) => {
    const peak = Math.max(...rows.map((r) => r[key]));
    const steps = rows.slice(1).map((r, i) => r[key] - rows[i][key]);
    const dirs = steps.filter((d) => Math.abs(d) > 1e-9).map(Math.sign);
    return {
      overshoot_pct: Math.max(0, (peak - 1) * 100),
      peak_frame: rows.findIndex((r) => r[key] === peak),
      turns: dirs.filter((s, i) => i > 0 && s !== dirs[i - 1]).length,
      lands_at: rows[rows.length - 1][key],
    };
  };
  const card = rise("card");
  const inset = rise("inset");

  report[name] = {
    fps,
    enter_frames: enter,
    moving_frames: moving,
    // Насколько наезд проскочил единицу вниз: у плавного входа — ноль.
    overshoot_pct: Math.max(0, (1 - Math.min(...rows.map((r) => r.scale))) * 100),
    // Заказанная пресетом амплитуда перелёта карточки, доля масштаба.
    card_overshoot_ratio: cfg.overshoot,
    card_overshoot_pct: card.overshoot_pct,
    card_peak_frame: card.peak_frame,
    card_turns: card.turns,
    card_lands_at: card.lands_at,
    inset_overshoot_pct: inset.overshoot_pct,
    inset_peak_frame: inset.peak_frame,
    inset_turns: inset.turns,
    inset_lands_at: inset.lands_at,
    alpha_max_step: Math.max(...alphaSteps),
    alpha_monotonic: alphaSteps.every((d) => d >= -1e-9),
    scale_monotonic: scaleSteps.every((d) => d <= 1e-9),
    alpha_full_frame: rows.findIndex((r) => r.alpha >= 0.999),
    // Какая доля наезда ещё впереди, когда картинка уже видна (альфа 0.9).
    push_left_when_visible: (rows[alphaNine].scale - 1) / (FULL_PUSH - 1),
    scale_settled_frame: last + 1,
    blur_gone_frame: rows.findIndex((r) => r.blur < 0.05),
    scale: rows.map((r) => r.scale),
    card_scale: rows.map((r) => r.card),
    inset_scale: rows.map((r) => r.inset),
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
  console.log(
    `   карточка: перелёт ${r.card_overshoot_pct.toFixed(2)} % ` +
      `(просили ${(r.card_overshoot_ratio * 100).toFixed(1)} %) на кадре ${r.card_peak_frame}, ` +
      `разворотов ${r.card_turns}, села на ${r.card_lands_at.toFixed(3)}`,
  );
  console.log(
    `   угловая: перелёт ${r.inset_overshoot_pct.toFixed(2)} % на кадре ${r.inset_peak_frame}, ` +
      `разворотов ${r.inset_turns}, села на ${r.inset_lands_at.toFixed(3)}`,
  );
  console.log("   альфа:  ", rows.map((x) => x.alpha.toFixed(3)).join(" "));
  console.log("   масштаб:", rows.map((x) => x.scale.toFixed(3)).join(" "));
  console.log("   карточка:", rows.map((x) => x.card.toFixed(3)).join(" "));
  console.log("   угловая:", rows.map((x) => x.inset.toFixed(3)).join(" "));
  console.log("   блюр px:", rows.map((x) => x.blur.toFixed(1)).join(" "));
}

if (asJson) console.log(JSON.stringify(report));
