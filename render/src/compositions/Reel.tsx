/**
 * Композиция ролика: видео-подложка + камера + b-roll + субтитры.
 *
 * Порядок слоёв снизу вверх: видео → b-roll → субтитры. Субтитры всегда сверху:
 * картинка не должна перекрывать текст.
 */

import React from "react";
import {
  AbsoluteFill,
  Audio,
  OffthreadVideo,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import type { PlanCamera, RenderPlan } from "../types";
import { Broll } from "../layers/Broll";
import { Captions } from "../layers/Captions";

const useCameraScale = (moves: PlanCamera[]): { scale: number; dx: number; dy: number } => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;

  const active = moves.find((m) => t >= m.start && t <= m.end);
  if (!active) return { scale: 1, dx: 0, dy: 0 };

  const framesIn = frame - active.start * fps;
  const progress = spring({
    frame: framesIn,
    fps,
    config: { damping: 13, stiffness: 220, mass: 0.7 },
    durationInFrames: Math.max(3, Math.round((active.end - active.start) * fps)),
  });

  if (active.type === "shake") {
    const amp = interpolate(progress, [0, 1], [1, 0], { extrapolateRight: "clamp" });
    return {
      scale: 1 + (active.scale - 1) * 0.35,
      dx: Math.sin(frame * 1.9) * 14 * amp,
      dy: Math.cos(frame * 2.3) * 10 * amp,
    };
  }

  // zoom_punch: резкий наезд и мягкий возврат
  const scale = interpolate(progress, [0, 0.45, 1], [1, active.scale, 1], {
    extrapolateRight: "clamp",
  });
  return { scale, dx: 0, dy: 0 };
};

export const Reel: React.FC<RenderPlan> = (plan) => {
  const { preset } = plan;
  const camera = useCameraScale(plan.camera ?? []);

  return (
    <AbsoluteFill style={{ backgroundColor: preset.background }}>
      <AbsoluteFill
        style={{
          transform: `translate(${camera.dx}px, ${camera.dy}px) scale(${camera.scale})`,
          transformOrigin: "center center",
          willChange: "transform",
        }}
      >
        {plan.video ? (
          <OffthreadVideo
            src={staticFile(plan.video)}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
            // Звук берём из этой же дорожки, если отдельной нет.
            muted={Boolean(plan.audio)}
          />
        ) : null}
      </AbsoluteFill>

      {plan.audio ? <Audio src={staticFile(plan.audio)} /> : null}

      <Broll items={plan.broll ?? []} preset={preset} />
      <Captions blocks={plan.blocks ?? []} preset={preset} />
    </AbsoluteFill>
  );
};
