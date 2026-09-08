/**
 * M7: вставные картинки. Три анимации входа, включая fullscreen → inset.
 *
 * Картинка никогда не рисуется в центре по умолчанию: там лицо (§7 DoD).
 *
 * Всё время — в кадрах от округлённых до сетки границ: `item.start` лежит с
 * точностью до миллисекунды и на сетку не попадает, поэтому первый
 * отрисованный кадр давал `framesIn ∈ [0,1)` — карточка стартовала не с нуля,
 * а с альфы 0.09-0.45, и момент появления гулял от ролика к ролику (§8.3).
 */

import React from "react";
import { Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import type { Anchor, PlanBroll, Preset } from "../types";
import { animFrames, easeOut, motionPx, springFrames } from "../style";

const INSET_RATIO = 0.44; // доля ширины кадра под картинку в углу
const MARGIN = 0.05; // отступ от края в долях ширины

/** Появление и уход карточки. Обе фазы одной природы — тормозят к своему концу. */
const FADE_MS = 220;

/** Пружина входа: ведёт только геометрию, альфу ведёт FADE_MS. */
const ENTER_SPRING = { damping: 14, stiffness: 200, mass: 0.8 };

type Box = { left: number; top: number; width: number; height: number };

const anchorBox = (anchor: Anchor, W: number, H: number): Box => {
  const w = W * INSET_RATIO;
  const h = w; // карточка квадратная, содержимое кроется object-fit
  const m = W * MARGIN;
  switch (anchor) {
    case "top_left":
      return { left: m, top: H * 0.16, width: w, height: h };
    case "top_right":
      return { left: W - w - m, top: H * 0.16, width: w, height: h };
    case "bottom_left":
      return { left: m, top: H * 0.58, width: w, height: h };
    case "bottom_right":
      return { left: W - w - m, top: H * 0.58, width: w, height: h };
    case "center":
      return { left: (W - w) / 2, top: (H - h) / 2, width: w, height: h };
    case "fullscreen":
      return { left: 0, top: 0, width: W, height: H };
  }
};

const Item: React.FC<{ item: PlanBroll; preset: Preset }> = ({ item, preset }) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  if (!item.asset) return null;

  const startFrame = Math.round(item.start * fps);
  const endFrame = Math.round(item.end * fps);
  if (frame < startFrame || frame > endFrame) return null;

  const framesIn = frame - startFrame;
  const fade = animFrames(FADE_MS, fps);
  // Геометрию ведёт пружина натуральной длительности: сжатая в 1.8 раза
  // обрывалась на максимуме скорости и вставала как вкопанная.
  const enter = spring({
    frame: framesIn,
    fps,
    durationInFrames: springFrames(fps, ENTER_SPRING),
    config: ENTER_SPRING,
  });
  const appear = easeOut(
    interpolate(framesIn, [0, fade], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }),
  );
  const leave =
    1 -
    easeOut(
      interpolate(frame, [endFrame - fade, endFrame], [0, 1], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      }),
    );

  const target = anchorBox(item.anchor, width, height);
  let box: Box = target;
  const opacity = appear * leave;
  let scale = 1;
  let translateY = 0;

  if (item.anim === "pop_in") {
    scale = interpolate(enter, [0, 1], [0.72, 1], { extrapolateRight: "clamp" });
  } else if (item.anim === "slide_up") {
    const slide = motionPx(preset.animation, "broll_slide_px", Math.round(height * 0.09));
    translateY = interpolate(enter, [0, 1], [slide, 0], { extrapolateRight: "clamp" });
  } else if (item.anim === "fullscreen_to_inset") {
    // Первые 40% времени картинка держит весь кадр, потом уезжает в угол.
    const holdFrames = Math.min(Math.round(0.9 * fps), Math.round((endFrame - startFrame) * 0.4));
    const travelFrames = animFrames(420, fps);
    const k = easeOut(
      interpolate(framesIn, [holdFrames, holdFrames + travelFrames], [0, 1], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      }),
    );
    const full = anchorBox("fullscreen", width, height);
    const dest = anchorBox(item.anchor === "fullscreen" ? "top_right" : item.anchor, width, height);
    box = {
      left: full.left + (dest.left - full.left) * k,
      top: full.top + (dest.top - full.top) * k,
      width: full.width + (dest.width - full.width) * k,
      height: full.height + (dest.height - full.height) * k,
    };
  }

  const inset = box.width < width * 0.9;

  return (
    <div
      style={{
        position: "absolute",
        left: box.left,
        top: box.top,
        width: box.width,
        height: box.height,
        opacity,
        transform: `translateY(${translateY}px) scale(${scale})`,
        borderRadius: inset ? preset.panel.radius : 0,
        overflow: "hidden",
        boxShadow: inset ? `0 ${height * 0.012}px ${height * 0.03}px rgba(0,0,0,0.45)` : undefined,
        border: inset ? `${preset.panel.border_width * 2}px solid rgba(255,255,255,0.16)` : undefined,
        willChange: "transform, opacity",
      }}
    >
      {/* Сток кладёт объект в верхнюю треть кадра: центрированный cover срезает
          ему голову, поэтому смотрим выше центра. */}
      <Img
        src={staticFile(item.asset)}
        style={{ width: "100%", height: "100%", objectFit: "cover", objectPosition: "center 30%" }}
      />
    </div>
  );
};

export const Broll: React.FC<{ items: PlanBroll[]; preset: Preset }> = ({ items, preset }) => (
  <>
    {items.map((item, i) => (
      <Item key={`${item.start}-${i}`} item={item} preset={preset} />
    ))}
  </>
);
