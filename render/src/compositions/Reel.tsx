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
import { Video } from "@remotion/media";
import { AbsoluteFill, Audio, staticFile } from "remotion";
import type { RenderPlan } from "../types";
import { Broll } from "../layers/Broll";
import { Captions } from "../layers/Captions";

export const Reel: React.FC<RenderPlan> = (plan) => {
  const { preset } = plan;

  return (
    <AbsoluteFill style={{ backgroundColor: preset.background }}>
      <AbsoluteFill>
        {/*
          `<Video>` из `@remotion/media`, а не `OffthreadVideo`: кадр декодирует
          сам Chromium через WebCodecs, без отдельного процесса-компоситора и
          прокси-сервера кадров. Замер на 31.8 с ролике, один поток: 225 → 132 с
          рендера, пик памяти 4.0 → 0.87 ГБ; кадры совпадают по времени, PSNR
          46 дБ. Файл здесь всегда собран нашим ffmpeg (`cut.py`/`crop.py`) и
          не фрагментирован: на фрагментированном MP4 (`mdhd duration 0`, сдвиг
          B-кадров только в `elst`) @remotion/media 4.0.520 показывал кадр на два
          раньше — замерено на клипе со стока, пока он был.
          Если WebCodecs кадр не берёт, компонент сам откатывается на
          `OffthreadVideo`.
        */}
        {plan.video ? (
          <Video
            src={staticFile(plan.video)}
            objectFit="cover"
            style={{ width: "100%", height: "100%" }}
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
