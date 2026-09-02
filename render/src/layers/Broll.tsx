/**
 * M7: вставные картинки. Три анимации входа, включая fullscreen → inset.
 *
 * Картинка никогда не рисуется в центре по умолчанию: там лицо (§7 DoD).
 */

import React from "react";
import { Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import type { Anchor, PlanBroll, Preset } from "../types";
import { easeOut } from "../style";

const INSET_RATIO = 0.44; // доля ширины кадра под картинку в углу
const MARGIN = 0.05; // отступ от края в долях ширины

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
  const t = frame / fps;
  if (!item.asset) return null;
  if (t < item.start || t > item.end) return null;

  const framesIn = frame - item.start * fps;
  const enter = spring({
    frame: framesIn,
    fps,
    config: { damping: 14, stiffness: 200, mass: 0.8 },
    durationInFrames: Math.max(3, Math.round(0.34 * fps)),
  });
  const exit = interpolate(t, [item.end - 0.22, item.end], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const target = anchorBox(item.anchor, width, height);
  let box: Box = target;
  let opacity = enter * exit;
  let scale = 1;
  let translateY = 0;

  if (item.anim === "pop_in") {
    scale = interpolate(enter, [0, 1], [0.72, 1], { extrapolateRight: "clamp" });
  } else if (item.anim === "slide_up") {
    translateY = interpolate(enter, [0, 1], [height * 0.09, 0], { extrapolateRight: "clamp" });
  } else if (item.anim === "fullscreen_to_inset") {
    // Первые 40% времени картинка держит весь кадр, потом уезжает в угол.
    const hold = Math.min(0.9, (item.end - item.start) * 0.4);
    const k = easeOut(
      interpolate(t, [item.start + hold, item.start + hold + 0.42], [0, 1], {
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
    opacity = Math.min(1, enter * 2) * exit;
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
      <Img
        src={staticFile(item.asset)}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
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
