/**
 * Пословные субтитры: три состояния слова, пружина на появление,
 * glass-панель, контейнер растёт под контент сам.
 *
 * Анимация — код, а не модель (§0.2). Пресет задаёт только числа.
 *
 * Всё время считается в кадрах от округлённых до сетки границ. Секунды в плане
 * лежат с точностью до миллисекунды и между кадрами не попадают, поэтому по
 * времени первый отрисованный кадр приходился на середину подъёма: слово
 * стартовало с альфы 0.09-0.45, и момент появления гулял от ролика к ролику.
 */

import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import type { PlanBlock, PlanWord, Preset } from "../types";
import { animFrames, easeOut, motionPx, popBump, rgba, springFrames } from "../style";

type WordState = "ghost" | "active" | "past";

const stateOf = (word: PlanWord, frame: number, fps: number): WordState => {
  if (frame < Math.round(word.start * fps)) return "ghost";
  if (frame <= Math.round(word.end * fps)) return "active";
  return "past";
};

/**
 * Нижняя граница нахлёста блоков: уходящий блок обязан быть ещё виден, когда
 * пришёл новый, иначе между репликами моргает пустой кадр. Пресеты просят
 * 100-140 мс — этого мало на любом fps, поэтому берём максимум с 220 мс.
 */
const MIN_OVERLAP_MS = 220;

/** Хвост блока в кадрах. Считается в одном месте: им живут и Block, и фильтр. */
const blockOutFrames = (preset: Preset, fps: number): number =>
  animFrames(Math.max(preset.animation.block_out_ms, MIN_OVERLAP_MS), fps);

const emphasisScale = (word: PlanWord, preset: Preset): number => {
  if (word.emphasis === "accent") return preset.animation.accent_scale;
  if (word.emphasis === "shout") return preset.animation.shout_scale;
  return 1;
};

const emphasisColor = (word: PlanWord, preset: Preset): string => {
  if (word.emphasis === "accent") return preset.colors.accent;
  if (word.emphasis === "shout") return preset.colors.shout;
  return preset.colors.active;
};

/**
 * Мягкая тень вместо обводки.
 *
 * Жирный шрифт с чёрным контуром читается как дефолтный шаблон монтажки, а под
 * панелью обводка вдобавок избыточна. Читаемость держат четыре слоя: гало без
 * смещения работает как размытый контур, дальше контактная тень, средняя и
 * широкая подложка, которая гасит светлый фон под всей строкой.
 */
const softShadow = (preset: Preset): string => {
  const s = preset.font.size;
  const c = preset.colors.shadow;
  return [
    `0 0 ${Math.round(s * 0.07)}px ${rgba(c, 0.5)}`,
    `0 ${Math.max(1, Math.round(s * 0.012))}px ${Math.round(s * 0.03)}px ${rgba(c, 0.55)}`,
    `0 ${Math.round(s * 0.05)}px ${Math.round(s * 0.1)}px ${rgba(c, 0.42)}`,
    `0 ${Math.round(s * 0.11)}px ${Math.round(s * 0.26)}px ${rgba(c, 0.3)}`,
  ].join(", ");
};

const Word: React.FC<{ word: PlanWord; preset: Preset }> = ({ word, preset }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const state = stateOf(word, frame, fps);
  const a = preset.animation;

  // Пружина стартует в кадре, когда слово прозвучало. Ghost-слова уже
  // нарисованы на своих местах, поэтому контейнер не прыгает при появлении.
  const framesSinceStart = frame - Math.round(word.start * fps);
  const config = {
    damping: a.spring_damping,
    stiffness: a.spring_stiffness,
    mass: a.spring_mass,
  };
  const pop =
    framesSinceStart >= 0
      ? spring({
          frame: framesSinceStart,
          fps,
          // Натуральная длительность: сжатая пружина обрывается на максимуме
          // скорости (было 5 кадров при 17 натуральных для glass).
          durationInFrames: springFrames(fps, config),
          config,
        })
      : 0;

  const base = emphasisScale(word, preset);
  // Рост до размера акцента ведёт пружина, а транзиентный «пшик» — время.
  // На пружине пик приходился на 0.55 её значения, то есть на третий кадр:
  // масштаб успевал вернуться раньше, чем глаз замечал прибавку.
  const bump = state === "ghost" ? 0 : popBump(framesSinceStart / animFrames(a.pop_ms, fps));
  const scale =
    state === "ghost"
      ? 1
      : interpolate(pop, [0, 1], [1, base], { extrapolateRight: "clamp" }) *
        (1 + (a.pop_scale - 1) * bump);

  const rotate =
    word.emphasis === "shout" && state !== "ghost"
      ? interpolate(pop, [0, 1], [0, a.shout_rotate], { extrapolateRight: "clamp" })
      : 0;

  const color =
    state === "active"
      ? emphasisColor(word, preset)
      : state === "past"
        ? preset.colors.past
        : preset.colors.ghost;

  // ghost_opacity = 0 (glass) гасит слово, но не убирает его из потока: место
  // под него зарезервировано, и панель не меняет размер, когда слово зазвучит.
  const opacity =
    state === "active" ? 1 : state === "past" ? preset.colors.past_opacity : preset.colors.ghost_opacity;

  return (
    <span
      style={{
        display: "inline-block",
        color,
        opacity,
        transform: `scale(${scale}) rotate(${rotate}deg)`,
        transformOrigin: "center bottom",
        willChange: "transform, opacity",
        textShadow: softShadow(preset),
      }}
    >
      {word.text}
      {/* Эмодзи — цветной глиф: тень под ним читается как грязь, гасим. */}
      {word.emoji ? <span style={{ textShadow: "none" }}> {word.emoji}</span> : null}
    </span>
  );
};

const Block: React.FC<{ block: PlanBlock; preset: Preset }> = ({ block, preset }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const a = preset.animation;

  const startFrame = Math.round(block.start * fps);
  const endFrame = Math.round(block.end * fps);
  const inFrames = animFrames(a.block_in_ms, fps);
  const outFrames = blockOutFrames(preset, fps);

  // Блок появляется от своего начала и уходит ПОСЛЕ своего конца. Раньше он
  // гас за block_out_ms до конца, то есть таял прямо на последнем слове, а
  // между соседними блоками (`a.end == b.start` после build_plan) оставалась
  // дыра. Теперь уходящий блок ещё виден, когда приходит новый: нахлёст равен
  // outFrames. Обе кривые одной природы — тормозят к своему конечному
  // состоянию, поэтому выход не линейный и не щёлкает.
  const appear = easeOut(
    interpolate(frame, [startFrame, startFrame + inFrames], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }),
  );
  const disappear =
    1 -
    easeOut(
      interpolate(frame, [endFrame, endFrame + outFrames], [0, 1], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      }),
    );
  const presence = Math.min(appear, disappear);
  if (presence <= 0) return null;

  const p = preset.panel;
  const shift = motionPx(a, "block_shift_px", Math.round(preset.font.size * 0.28));

  return (
    <div
      style={{
        // flex-контейнер: панель растёт под текст, ничего не считаем руками
        display: "inline-flex",
        flexDirection: "column",
        alignItems: preset.caption.align === "left" ? "flex-start" : "center",
        gap: Math.round(preset.font.size * (preset.caption.line_height - 1)),
        padding: p.enabled ? `${p.padding_y}px ${p.padding_x}px` : 0,
        borderRadius: p.enabled ? p.radius : 0,
        background: p.enabled ? rgba(p.color, p.opacity) : "transparent",
        backdropFilter: p.enabled && p.blur > 0 ? `blur(${p.blur}px) saturate(1.4)` : undefined,
        WebkitBackdropFilter:
          p.enabled && p.blur > 0 ? `blur(${p.blur}px) saturate(1.4)` : undefined,
        border:
          p.enabled && p.border_width > 0
            ? `${p.border_width}px solid ${rgba(p.border_color, p.border_opacity)}`
            : undefined,
        opacity: presence,
        transform: `translateY(${(1 - presence) * shift}px) scale(${0.965 + presence * 0.035})`,
        maxWidth: "88%",
        willChange: "transform, opacity",
      }}
    >
      {block.lines.map((line, li) => (
        <div
          key={li}
          style={{
            display: "flex",
            flexWrap: "nowrap",
            gap: Math.round(preset.font.size * 0.24),
            alignItems: "baseline",
            justifyContent: preset.caption.align === "left" ? "flex-start" : "center",
          }}
        >
          {line.map((word) => (
            <Word key={word.word_idx} word={word} preset={preset} />
          ))}
        </div>
      ))}
    </div>
  );
};

/**
 * Минимальная безопасная зона у края кадра. Пресет может попросить margin 0, но
 * прижатый к краю текст обрезается плеером и интерфейсом ленты — этого не хочет
 * ни один пресет, поэтому нижняя граница жёсткая.
 */
const SAFE_AREA_RATIO = 0.07;

export const Captions: React.FC<{ blocks: PlanBlock[]; preset: Preset }> = ({ blocks, preset }) => {
  const frame = useCurrentFrame();
  const { fps, height } = useVideoConfig();
  const margin = Math.max(preset.caption.margin, Math.round(height * SAFE_AREA_RATIO));
  // Хвост ровно тот же, что у Block: блок живёт на outFrames дольше своего
  // конца и обязан пережить фильтр, иначе нахлёст срежется здесь.
  const tail = blockOutFrames(preset, fps);

  const visible = blocks.filter(
    (b) => frame >= Math.round(b.start * fps) && frame <= Math.round(b.end * fps) + tail,
  );
  if (visible.length === 0) return null;

  const justify = (position: string) =>
    position === "top" ? "flex-start" : position === "center" ? "center" : "flex-end";

  return (
    <>
      {/* Порядок сохраняем: во время нахлёста уходящий блок лежит под приходящим. */}
      {visible.map((block, i) => (
        <div
          key={`${block.start}-${i}`}
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: justify(block.position),
            justifyContent: "center",
            paddingTop: block.position === "top" ? margin : 0,
            paddingBottom: block.position === "bottom" ? margin : 0,
            fontFamily: `"ReelbotCaption", "${preset.font.family}", sans-serif`,
            fontWeight: preset.font.weight,
            fontSize: preset.font.size,
            lineHeight: preset.caption.line_height,
            letterSpacing: `${preset.font.letter_spacing}em`,
            textAlign: preset.caption.align,
            pointerEvents: "none",
          }}
        >
          <Block block={block} preset={preset} />
        </div>
      ))}
    </>
  );
};
