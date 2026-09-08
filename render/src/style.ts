/** Мелкие помощники стиля. Никакой логики монтажа здесь нет. */

import { measureSpring } from "remotion";
import type { Preset } from "./types";

export const rgba = (hex: string, opacity = 1): string => {
  let v = hex.replace("#", "");
  if (v.length === 3) {
    v = v
      .split("")
      .map((c) => c + c)
      .join("");
  }
  const r = parseInt(v.slice(0, 2), 16);
  const g = parseInt(v.slice(2, 4), 16);
  const b = parseInt(v.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, opacity))})`;
};

/** Плавная ease-out кривая для входов, где пружина была бы слишком игривой. */
export const easeOut = (t: number): number => 1 - Math.pow(1 - Math.max(0, Math.min(1, t)), 3);

// ------------------------------------------------------------------ тайминг

/**
 * Пол длительности анимации в кадрах.
 *
 * Короче восьми кадров глаз читает переход не как движение, а как мигание:
 * motion blur рендерер не рисует, и вся анимация вырождается в две-три
 * ступеньки (§8.3 аудита: видимое движение картинки — 3 кадра, 90 мс).
 * Поэтому длительности из пресета — пожелание, а восемь кадров — закон.
 */
export const MIN_ANIM_FRAMES = 8;

/** Миллисекунды пресета → кадры, но не короче порога восприятия. */
export const animFrames = (ms: number, fps: number): number =>
  Math.max(MIN_ANIM_FRAMES, Math.round((ms / 1000) * fps));

export type SpringConfig = { damping: number; stiffness: number; mass: number };

/**
 * Натуральная длительность пружины в кадрах.
 *
 * `durationInFrames` короче натуральной сжимает пружину по времени и обрывает
 * её на максимуме скорости — вместо упругости выходит рывок (§8.3: задано
 * 10 кадров при 18 натуральных). `measureSpring` кеширует результат внутри
 * себя, так что вызывать его на каждом кадре дёшево.
 */
export const springFrames = (fps: number, config: SpringConfig): number =>
  Math.max(MIN_ANIM_FRAMES, measureSpring({ fps, config }));

/**
 * Транзиентный «пшик»: 0 → 1 → 0 по нормированному времени.
 *
 * Считается по времени, а не по значению пружины: на пружине пик приходился
 * на третий кадр, то есть на два кадра всей заметной фазы. Быстрый старт и
 * мягкая посадка в обеих половинах — производная в вершине и в конце нулевая,
 * поэтому масштаб не щёлкает.
 */
export const popBump = (q: number, rise = 0.35): number => {
  if (q <= 0 || q >= 1) return 0;
  return q < rise ? easeOut(q / rise) : 1 - easeOut((q - rise) / (1 - rise));
};

/**
 * Пиксельные величины движения приезжают из пресета: `Preset.scaled(height)`
 * пересчитывает их под высоту кадра, а раньше они были константами в TSX.
 *
 * В `types.ts` (чужой файл) поля ещё не объявлены, поэтому читаем через узкий
 * каст и запасное значение — старый план без этих полей рендерится по-старому.
 */
type MotionPx = { block_shift_px?: number; broll_slide_px?: number };

export const motionPx = (
  animation: Preset["animation"],
  key: keyof MotionPx,
  fallback: number,
): number => {
  const value = (animation as MotionPx)[key];
  return typeof value === "number" ? value : fallback;
};
