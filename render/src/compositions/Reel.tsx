/**
 * Композиция ролика: видео-подложка + b-roll + субтитры.
 *
 * Камеры здесь нет: движение делается сменой крупности на резе (crop.py), а не
 * зумом поверх кадра. Зум по номеру кадра давал дёрганье и спорил с субтитрами.
 *
 * Порядок слоёв снизу вверх: видео → b-roll → субтитры. Субтитры всегда сверху:
 * картинка не должна перекрывать текст.
 */

import React from "react";
import { AbsoluteFill, Audio, OffthreadVideo, staticFile } from "remotion";
import type { RenderPlan } from "../types";
import { Broll } from "../layers/Broll";
import { Captions } from "../layers/Captions";

export const Reel: React.FC<RenderPlan> = (plan) => {
  const { preset } = plan;

  return (
    <AbsoluteFill style={{ backgroundColor: preset.background }}>
      <AbsoluteFill>
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
