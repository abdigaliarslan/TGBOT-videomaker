import React from "react";
import { Composition, continueRender, delayRender, staticFile } from "remotion";
import { Reel } from "./compositions/Reel";
import { EMPTY_PLAN, type RenderPlan } from "./types";

/**
 * Шрифт грузится ИЗ БАНДЛА, а не из системы: Chromium в контейнере рендера
 * иначе подставит дефолт и кириллица поедет (§5 M5 ⚠).
 */
const useCaptionFont = (fontUrl: string | null | undefined) => {
  const [handle] = React.useState(() => delayRender("загрузка шрифта субтитров"));
  React.useEffect(() => {
    if (!fontUrl) {
      continueRender(handle);
      return;
    }
    // Диапазон весов, а не умолчание: без дескриптора лицо регистрируется как
    // `normal` (400), а Captions.tsx просит вес пресета — пара, на которой
    // Chromium вправе нарисовать синтетический жир. Замер показал, что на
    // нынешней сборке он ширину не меняет (объявленные 400 и 100 против
    // запрошенных 800 дали те же пиксели), то есть активной ошибки не было.
    // Объявление всё равно обязано совпадать с линейкой `render/measure.mjs`:
    // сверка 2 px верна ровно настолько, насколько рендер объявляет шрифт так
    // же, как страница замера. Сторожит `tests/test_font_contract.py`.
    const face = new FontFace("ReelbotCaption", `url(${staticFile(fontUrl)})`, {
      weight: "1 1000",
    });
    face
      .load()
      .then((loaded) => {
        // В типах lib.dom для этой версии TS у FontFaceSet нет add(), хотя в
        // Chromium он есть с 2016 года.
        (document.fonts as unknown as { add(f: FontFace): void }).add(loaded);
        continueRender(handle);
      })
      .catch((err) => {
        // Лучше отрендерить системным шрифтом, чем не отрендерить вовсе.
        console.warn("шрифт не загрузился:", err);
        continueRender(handle);
      });
  }, [fontUrl, handle]);
};

const ReelWithFont: React.FC<RenderPlan> = (plan) => {
  useCaptionFont(plan.fontUrl);
  return <Reel {...plan} />;
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Reel"
      component={ReelWithFont}
      defaultProps={EMPTY_PLAN}
      durationInFrames={Math.round(EMPTY_PLAN.duration * EMPTY_PLAN.fps)}
      fps={EMPTY_PLAN.fps}
      width={EMPTY_PLAN.width}
      height={EMPTY_PLAN.height}
      calculateMetadata={({ props }) => ({
        durationInFrames: Math.max(1, Math.round(props.duration * props.fps)),
        fps: props.fps,
        width: props.width,
        height: props.height,
      })}
    />
  );
};
