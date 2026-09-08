/**
 * M7: вставные картинки. Полноэкранная вставка по умолчанию, инсет — только как
 * конечная точка перелёта `fullscreen_to_inset`.
 *
 * Геометрию компонент не считает: прямоугольник покоя приходит в плане из
 * `src/core/geometry.py` (§0, Ф4). Раньше здесь жила своя таблица якорей с
 * константами `INSET_RATIO`, `MARGIN`, `0.16`, `0.58`, о которых Python не знал
 * и поэтому разводил субтитр с картинкой по половинам кадра (AUDIT §8.4).
 *
 * Всё время — в кадрах от округлённых до сетки границ: `item.start` лежит с
 * точностью до миллисекунды и на сетку не попадает, поэтому первый
 * отрисованный кадр давал `framesIn ∈ [0,1)` — карточка стартовала не с нуля,
 * а с альфы 0.09-0.45, и момент появления гулял от ролика к ролику (§8.3).
 */

import React from "react";
import { Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import type { PlanBroll, Preset, Rect } from "../types";
import { MIN_ANIM_FRAMES, animFrames, easeOut, motionPx, springFrames } from "../style";

/** Появление и уход карточки. Обе фазы одной природы — тормозят к своему концу. */
const FADE_MS = 220;

/**
 * Сколько картинка держит полный кадр, прежде чем уехать в угол: не дольше
 * HOLD_MS и не больше HOLD_RATIO от всей вставки. Эти же числа знает
 * `src/pipeline/broll.py` (`inset_floor_seconds`) — из них считается минимальная
 * длительность перелёта, поэтому менять их надо в двух местах сразу.
 */
const HOLD_MS = 900;
const HOLD_RATIO = 0.35;

/**
 * Сам перелёт полный кадр → угол. Дольше прежних 420 мс: карточка в углу
 * маленькая (её размер считает Python по свободной полосе кадра), поэтому
 * масштаб за перелёт меняется впятеро, и на 420 мс это швырок.
 */
const TRAVEL_MS = 520;

/**
 * Пружина входа: ведёт только геометрию карточки, альфу ведёт FADE_MS.
 * Числа — из пресета, как у соседнего слоя (Captions.tsx): характер движения это
 * стиль, а стиль живёт в данных, а не в константах TSX (§0).
 */
const springOf = (a: Preset["animation"]) => ({
  damping: a.spring_damping,
  stiffness: a.spring_stiffness,
  mass: a.spring_mass,
});

/**
 * Полный кадр не может «прилетать» из ничего: за краями уменьшенной картинки
 * сразу видно исходное видео, и вставка читается как отклеившаяся наклейка.
 * Поэтому у неё вход — лёгкий наезд, а прыжок с 0.72 остаётся карточке в углу,
 * у которой края и должны быть видны.
 */
const FULL_PUSH = 1.06;
const INSET_POP = 0.72;

const Item: React.FC<{ item: PlanBroll; preset: Preset }> = ({ item, preset }) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  if (!item.asset) return null;

  const startFrame = Math.round(item.start * fps);
  const endFrame = Math.round(item.end * fps);
  if (frame < startFrame || frame > endFrame) return null;

  const framesIn = frame - startFrame;
  const fade = animFrames(FADE_MS, fps);
  const enterSpring = springOf(preset.animation);
  // Геометрию карточки ведёт пружина натуральной длительности: сжатая в 1.8 раза
  // обрывалась на максимуме скорости и вставала как вкопанная.
  const enter = spring({
    frame: framesIn,
    fps,
    durationInFrames: springFrames(fps, enterSpring),
    config: enterSpring,
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

  // План без прямоугольника (собран до Ф4) — вставка полноэкранная: это
  // единственная геометрия, которую можно взять, ничего не решая самому.
  const rest: Rect = item.rect ?? { x: 0, y: 0, width, height };
  const isFull = rest.width >= width - 1 && rest.height >= height - 1;

  let scale = 1;
  let shiftX = 0;
  let shiftY = 0;
  // Скругление, рамка и тень — принадлежность карточки, а не полного кадра.
  // Во время перелёта они проявляются вместе с ним, а не включаются разом за
  // один кадр, когда бокс ещё почти полноэкранный (§8.3).
  let chrome = isFull ? 0 : 1;

  if (item.anim === "fullscreen_to_inset") {
    const span = endFrame - startFrame;
    const holdFrames = Math.min(animFrames(HOLD_MS, fps), Math.round(span * HOLD_RATIO));
    // Перелёт обязан уложиться в то, что осталось после удержания и затухания.
    // Считать его от одних констант нельзя: `broll.py` держит длительность не
    // короче `INSET_SECONDS`, но контракт EDL разрешает 0.8 с, а перерендер
    // сохранённого листа идёт мимо нормализации — на таком плане карточка
    // прилетала в угол уже растворяясь (AUDIT §8.3).
    const room = span - holdFrames - fade;
    const travel = Math.min(animFrames(TRAVEL_MS, fps), room);
    // Кадров не хватает даже на порог восприятия — не улетаем вовсе: инсет
    // рисуется статикой. Мигание в три кадра хуже, чем честно неподвижная
    // карточка.
    const k =
      travel < MIN_ANIM_FRAMES
        ? 1
        : easeOut(
            interpolate(framesIn, [holdFrames, holdFrames + travel], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          );
    // Летит МАСШТАБ И ПОЛОЖЕНИЕ при неизменной пропорции. Раньше ширина и
    // высота интерполировались одним коэффициентом: пропорция бокса плыла с
    // 9:16 до пропорции карточки, `cover` пересчитывал кроп на каждом кадре и
    // содержимое дышало прямо во время полёта (§8.3).
    const cover = Math.max(width / rest.width, height / rest.height);
    scale = cover + (1 - cover) * k;
    shiftX = (width / 2 - (rest.x + rest.width / 2)) * (1 - k);
    shiftY = (height / 2 - (rest.y + rest.height / 2)) * (1 - k);
    chrome = isFull ? 0 : k;
  } else if (isFull) {
    // Вход и выход одной природы: оба easeOut, а не пружина против линейки.
    const push = easeOut(
      interpolate(framesIn, [0, fade * 2], [0, 1], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      }),
    );
    scale = interpolate(push, [0, 1], [FULL_PUSH, 1]);
  } else if (item.anim === "slide_up") {
    const slide = motionPx(preset.animation, "broll_slide_px", Math.round(height * 0.09));
    shiftY = interpolate(enter, [0, 1], [slide, 0]);
  } else {
    scale = interpolate(enter, [0, 1], [INSET_POP, 1]);
  }

  const border = preset.panel.border_width * 2 * chrome;

  return (
    <div
      style={{
        position: "absolute",
        left: rest.x,
        top: rest.y,
        width: rest.width,
        height: rest.height,
        opacity: appear * leave,
        transform: `translate(${shiftX}px, ${shiftY}px) scale(${scale})`,
        borderRadius: preset.panel.radius * chrome,
        overflow: "hidden",
        boxShadow: `0 ${height * 0.012}px ${height * 0.03}px rgba(0,0,0,${0.45 * chrome})`,
        border: `${border}px solid rgba(255,255,255,${0.16 * chrome})`,
        // Remotion инжектит `box-sizing` только в Player: без этого рамка
        // прибавлялась к размеру и прямоугольник расходился с посчитанным.
        boxSizing: "border-box",
        willChange: "transform, opacity",
      }}
    >
      {isFull ? (
        // Полный кадр: `cover` здесь режет по-крупному и это не устранить
        // пропорцией карточки — сток отдаёт 2:3, кадр 9:16, и максимум из
        // (1080/800, 1920/1200) = 1.6 уводит за края по 100 px исходника с
        // каждой стороны, 15.6 % ширины. Поэтому объект вписывается целиком, а
        // поля закрывает размытая копия того же кадра.
        <>
          <Img
            src={staticFile(item.asset)}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "cover",
              filter: `blur(${Math.round(height * 0.025)}px) brightness(0.65)`,
              // Блюр размазывает края в прозрачность — подложку берём с запасом.
              transform: "scale(1.15)",
            }}
          />
          <Img
            src={staticFile(item.asset)}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "contain",
            }}
          />
        </>
      ) : (
        // Карточка наследует пропорцию источника, поэтому в покое `cover` не
        // режет ничего. Во время перелёта бокс масштабируется до полного кадра
        // и края уходят за него — это цена движения, а не постоянный кроп.
        // Если источник всё же другой пропорции, смотрим выше центра: сток
        // кладёт объект в верхнюю треть кадра.
        <Img
          src={staticFile(item.asset)}
          style={{ width: "100%", height: "100%", objectFit: "cover", objectPosition: "center 30%" }}
        />
      )}
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
