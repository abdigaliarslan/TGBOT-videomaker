/**
 * Пословные субтитры: три состояния слова, пружина на появление,
 * glass-панель, контейнер растёт под контент сам.
 *
 * Анимация — код, а не модель (§0.2). Пресет задаёт только числа.
 */

import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import type { PlanBlock, PlanWord, Preset } from "../types";
import { rgba } from "../style";

type WordState = "ghost" | "active" | "past";

const stateOf = (word: PlanWord, t: number): WordState => {
  if (t < word.start) return "ghost";
  if (t <= word.end) return "active";
  return "past";
};

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

const Word: React.FC<{ word: PlanWord; preset: Preset }> = ({ word, preset }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const state = stateOf(word, t);
  const a = preset.animation;

  // Пружина стартует в момент, когда слово прозвучало. Ghost-слова уже
  // нарисованы на своих местах, поэтому контейнер не прыгает при появлении.
  const framesSinceStart = frame - word.start * fps;
  const pop =
    framesSinceStart >= 0
      ? spring({
          frame: framesSinceStart,
          fps,
          config: {
            damping: a.spring_damping,
            stiffness: a.spring_stiffness,
            mass: a.spring_mass,
          },
          durationInFrames: Math.max(2, Math.round((a.pop_ms / 1000) * fps)),
        })
      : 0;

  const base = emphasisScale(word, preset);
  // 0 → обычный размер, 1 → размер акцента с лёгким перелётом.
  const scale =
    state === "ghost" ? 1 : interpolate(pop, [0, 1], [1, base], { extrapolateRight: "clamp" }) *
      interpolate(pop, [0, 0.55, 1], [1, a.pop_scale, 1], { extrapolateRight: "clamp" });

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
        // Обводка обязательна: субтитр должен читаться на любом кадре.
        WebkitTextStroke: `${Math.round(preset.font.size * 0.02)}px ${preset.colors.outline}`,
        paintOrder: "stroke fill",
        textShadow: `0 ${Math.round(preset.font.size * 0.05)}px ${Math.round(
          preset.font.size * 0.09,
        )}px ${rgba(preset.colors.shadow, 0.55)}`,
      }}
    >
      {word.text}
      {word.emoji ? <span style={{ WebkitTextStroke: "0" }}> {word.emoji}</span> : null}
    </span>
  );
};

const Block: React.FC<{ block: PlanBlock; preset: Preset }> = ({ block, preset }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const a = preset.animation;

  const inS = a.block_in_ms / 1000;
  const outS = a.block_out_ms / 1000;
  const appear = interpolate(t, [block.start - inS, block.start], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const disappear = interpolate(t, [block.end - outS, block.end], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const presence = Math.min(appear, disappear);
  if (presence <= 0) return null;

  const p = preset.panel;

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
        transform: `translateY(${(1 - presence) * preset.font.size * 0.28}px) scale(${
          0.965 + presence * 0.035
        })`,
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
  const t = frame / fps;
  const margin = Math.max(preset.caption.margin, Math.round(height * SAFE_AREA_RATIO));
  const lead = preset.animation.block_in_ms / 1000;
  const tail = preset.animation.block_out_ms / 1000;

  const visible = blocks.filter((b) => t >= b.start - lead && t <= b.end + tail);
  if (visible.length === 0) return null;

  const justify = (position: string) =>
    position === "top" ? "flex-start" : position === "center" ? "center" : "flex-end";

  return (
    <>
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
