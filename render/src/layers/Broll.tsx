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
import { lightLeak } from "@remotion/effects/light-leak";
import {
  Easing,
  Img,
  Solid,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import type { PlanBroll, Preset, Rect } from "../types";
import { MIN_ANIM_FRAMES, animFrames, animNum, easeOut, motionPx } from "../style";

/**
 * Уход карточки. Длительность знает `src/pipeline/broll.py` (`inset_floor_seconds`),
 * поэтому число живёт здесь, а не в пресете.
 */
const FADE_MS = 220;

/**
 * Вход вставки. Её длительность и глубина расфокуса — числа пресета
 * (`broll_in_ms`, `broll_in_blur`); здесь только запасные значения для плана,
 * собранного пресетом без этих полей.
 *
 * Прежний вход был ступенькой: альфа за восемь кадров по `easeOut`, у которого
 * максимальная скорость приходится на ПЕРВЫЙ кадр (он брал 33 % альфы разом),
 * плюс вспышка света поверх реза. Полноэкранная картинка, закрывающая
 * говорящего, так не появляется — она его подменяет. Теперь три параметра
 * ведутся раздельно и ни один не прыгает:
 *
 * - альфа — мягкий старт и мягкая посадка (`SOFT`): картинка проступает, и
 *   ни один кадр не берёт больше 17 % альфы;
 * - резкость — из расфокуса в фокус, быстро и тормозя к концу (`SETTLE`);
 * - масштаб — наезд вдвое дольше альфы, без перелёта и с медленным стартом
 *   (`PUSH`). Медленный старт здесь главное: на кривой с быстрым стартом 70 %
 *   хода проходило, пока картинка была ещё полупрозрачной, и наезда не было
 *   видно. Теперь к альфе 0.9 пройдено около половины — картинка сначала есть,
 *   потом садится на место (рассинхрон, который проверяет test_motion.py).
 *
 * Кривые подобраны перебором на 24 и 30 fps, числа — в `render/card_probe.mjs`.
 */
const ENTER_MS = 600;
const ENTER_BLUR = 0.012;
const SOFT = Easing.bezier(0.33, 0, 0.25, 1);
const SETTLE = Easing.bezier(0.16, 1, 0.3, 1);
const PUSH = Easing.bezier(0.33, 0, 0.2, 1);
const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

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
 * Полный кадр не может «прилетать» из ничего: за краями уменьшенной картинки
 * сразу видно исходное видео, и вставка читается как отклеившаяся наклейка.
 * Поэтому у неё вход — наезд СВЕРХУ вниз, с 1.08 до 1: края всё время за
 * кадром. Заодно они прячут и кайму расфокуса — размытый край прозрачен, и
 * без запаса по краям сквозь него просвечивало бы видео. Подъём с 0.72
 * остаётся карточке в углу, у которой края и должны быть видны.
 */
const FULL_PUSH = 1.08;
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
  const a = preset.animation;
  const enterFrames = animFrames(animNum(a, "broll_in_ms", ENTER_MS), fps);
  const appear = interpolate(framesIn, [0, enterFrames], [0, 1], { ...clamp, easing: SOFT });
  // Геометрия карточки в углу: тот же вход, но тормозящий к концу и без
  // перелёта. Пружина здесь стояла раньше и проскакивала цель на 4-7 %, то есть
  // картинка подпрыгивала на месте — ровно то, чего при плавном входе быть не
  // должно.
  const enter = interpolate(framesIn, [0, enterFrames], [0, 1], { ...clamp, easing: SETTLE });
  // Расфокус гаснет быстрее, чем доезжает наезд: к моменту, когда края
  // полноэкранной картинки входят в кадр, кайма размытия уже исчезла.
  const blur =
    height *
    animNum(a, "broll_in_blur", ENTER_BLUR) *
    interpolate(framesIn, [0, enterFrames], [1, 0], { ...clamp, easing: SETTLE });
  const push = interpolate(framesIn, [0, enterFrames * 2], [FULL_PUSH, 1], {
    ...clamp,
    easing: PUSH,
  });
  // Уход той же мягкой кривой, что и вход: `easeOut` стартовал на полной
  // скорости, и картинка не таяла, а обрывалась.
  const leave = interpolate(frame, [endFrame - fade, endFrame], [1, 0], {
    ...clamp,
    easing: SOFT,
  });

  // План без прямоугольника (собран до Ф4) — вставка полноэкранная: это
  // единственная геометрия, которую можно взять, ничего не решая самому.
  const rest: Rect = item.rect ?? { x: 0, y: 0, width, height };
  const isFull = rest.width >= width - 1 && rest.height >= height - 1;
  // Источник уже в пропорции кадра — значит `cover` ничего не срежет и
  // подложка под ним не нужна. Допуск в 1 % берёт и ровные 9:16, и то, что
  // модель отдаёт на пиксель мимо (1080×1920 против 1024×1820).
  const fitsFrame = item.aspect != null && Math.abs(item.aspect - width / height) < 0.01;

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
    // Вход у перелёта тот же, что у полного кадра: картинка проступает с
    // наездом и только потом летит в угол.
    scale = (cover + (1 - cover) * k) * push;
    shiftX = (width / 2 - (rest.x + rest.width / 2)) * (1 - k);
    shiftY = (height / 2 - (rest.y + rest.height / 2)) * (1 - k);
    chrome = isFull ? 0 : k;
  } else if (isFull) {
    scale = push;
  } else if (item.anim === "slide_up") {
    const slide = motionPx(preset.animation, "broll_slide_px", Math.round(height * 0.09));
    shiftY = interpolate(enter, [0, 1], [slide, 0]);
  } else {
    scale = interpolate(enter, [0, 1], [INSET_POP, 1]);
  }

  const border = preset.panel.border_width * 2 * chrome;

  /**
   * Ken Burns (М3): картинка едет всю свою жизнь, от появления до ухода.
   * Неподвижный кадр поверх видео читается как наклейка — это и есть основная
   * причина, по которой вставки выглядят вставленными.
   *
   * Живёт на отдельном слое ВНУТРИ карточки, а не на самой карточке: скругление,
   * рамка и тень принадлежат карточке и ехать вместе с содержимым не должны.
   *
   * Сдвиг считается от высоты БОКСА, а не кадра. В плане стоит доля высоты кадра
   * (0.012), и для полноэкранной вставки это одно и то же, но у карточки в углу
   * высота вчетверо меньше: 23 px сдвига при запасе `kb_scale/2 · 475 = 9.5 px`
   * вытащили бы из-под краёв пустоту. От высоты бокса запас держится сам —
   * `kb_shift_ratio` меньше половины `kb_scale`, и оба растут по `p` линейно.
   */
  const life = Math.max(1, endFrame - startFrame);
  const p = interpolate(framesIn, [0, life], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const kbScale = 1 + animNum(preset.animation, "kb_scale", 0.04) * p;
  const kbRatio = animNum(preset.animation, "kb_shift_ratio", 0.012);
  const kbShift = (item.kb_dir ?? 1) * rest.height * kbRatio * p;

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
      <div
        style={{
          position: "absolute",
          inset: 0,
          transform: `translateY(${kbShift}px) scale(${kbScale})`,
          // Ноль — фильтра нет вовсе: размытие стоит отрисовки, и платить за
          // него после входа незачем.
          filter: blur > 0.05 ? `blur(${blur.toFixed(2)}px)` : undefined,
          willChange: "transform",
        }}
      >
        {isFull && !fitsFrame ? (
          // Полный кадр КАРТИНКИ ЧУЖОЙ ПРОПОРЦИИ (старые листы — фотографии 2:3):
          // `cover` здесь режет по-крупному, и это не устранить пропорцией
          // карточки — максимум из (1080/800, 1920/1200) = 1.6 уводит за края по
          // 100 px исходника с каждой стороны, 15.6 % ширины. Поэтому объект
          // вписывается целиком, а поля закрывает размытая копия того же кадра.
          //
          // Картинку, нарисованную сразу под 9:16, этот приём только портит:
          // резать нечего, а размытая рамка по краям превращает полноэкранную
          // вставку обратно в «фотографию в рамке» — ровно то, от чего уходили.
          // Отсюда `fitsFrame`.
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
          // Если источник всё же другой пропорции, смотрим выше центра: объект
          // в кадре обычно лежит в верхней трети.
          <Img
            src={staticFile(item.asset)}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "cover",
              objectPosition: "center 30%",
            }}
          />
        )}
      </div>
    </div>
  );
};

/**
 * Вспышка света поверх реза на вход полноэкранной вставки. Пик засвета стоит
 * ровно на первом кадре картинки: вспышка разгорается над видео и гаснет уже
 * над вставкой, поэтому сама подмена кадра прячется в самом ярком месте.
 *
 * Карточке в углу вспышка не положена — у неё видны края, и засвет на весь
 * кадр ради маленькой картинки читался бы как сбой экспозиции. Из этого же
 * правила исключение — `fullscreen_to_inset`: он входит полным кадром.
 *
 * Шейдер WebGL2, в рендере без GPU — программный (swangle). Стоит он только
 * на кадрах самой вспышки: вне окна компонент ничего не рисует.
 *
 * ⚠ WebGL2 есть не в каждом режиме: на сервере без GPU `angle` его не даёт, и
 * эффект ронял ВЕСЬ рендер («Failed to acquire WebGL2 context»). Вспышка —
 * украшение, поэтому без WebGL2 её просто нет, а ролик собирается.
 */
let webgl2: boolean | null = null;
const hasWebGL2 = (): boolean => {
  if (webgl2 === null) {
    try {
      webgl2 = document.createElement("canvas").getContext("webgl2") !== null;
    } catch {
      webgl2 = false;
    }
    if (!webgl2) console.warn("[broll] WebGL2 недоступен — вспышка на вставках отключена");
  }
  return webgl2;
};

const LightLeak: React.FC<{ item: PlanBroll; preset: Preset; seed: number }> = ({
  item,
  preset,
  seed,
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const leakMs = animNum(preset.animation, "leak_ms", 0);
  if (!item.asset || leakMs <= 0) return null;
  const span = animFrames(leakMs, fps);

  const rest = item.rect;
  const isFull = !rest || (rest.width >= width - 1 && rest.height >= height - 1);
  if (!isFull && item.anim !== "fullscreen_to_inset") return null;

  const cut = Math.round(item.start * fps);
  const from = cut - Math.floor(span / 2);
  if (frame < from || frame > from + span) return null;
  if (!hasWebGL2()) return null;

  return (
    <Solid
      width={width}
      height={height}
      color="transparent"
      style={{ position: "absolute", inset: 0 }}
      effects={[
        lightLeak({
          seed,
          hueShift: animNum(preset.animation, "leak_hue", 0),
          progress: interpolate(frame, [from, from + span], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }),
      ]}
    />
  );
};

export const Broll: React.FC<{ items: PlanBroll[]; preset: Preset }> = ({ items, preset }) => (
  <>
    {items.map((item, i) => (
      <React.Fragment key={`${item.start}-${i}`}>
        <Item item={item} preset={preset} />
        {/* Сид по номеру вставки: две вспышки в ролике не повторяют одна другую. */}
        <LightLeak item={item} preset={preset} seed={i} />
      </React.Fragment>
    ))}
  </>
);
