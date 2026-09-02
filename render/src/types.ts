/**
 * Зеркало src/core/render_plan.py. Рендерер ничего не считает и не решает:
 * все секунды, строки и цвета приходят готовыми (§0).
 */

export type Emphasis = "none" | "accent" | "shout";
export type Position = "top" | "center" | "bottom";
export type BrollAnim = "pop_in" | "slide_up" | "fullscreen_to_inset";
export type Anchor =
  | "top_left"
  | "top_right"
  | "bottom_left"
  | "bottom_right"
  | "center"
  | "fullscreen";

export type PlanWord = {
  text: string;
  start: number;
  end: number;
  emphasis: Emphasis;
  emoji: string | null;
  word_idx: number;
};

export type PlanBlock = {
  start: number;
  end: number;
  position: Position;
  lines: PlanWord[][];
};

export type PlanBroll = {
  start: number;
  end: number;
  anim: BrollAnim;
  anchor: Anchor;
  asset: string | null;
  query: string;
};

export type PlanCamera = {
  start: number;
  end: number;
  type: "zoom_punch" | "shake";
  scale: number;
};

export type Preset = {
  name: string;
  title: string;
  font: {
    family: string;
    file: string | null;
    weight: number;
    size: number;
    uppercase: boolean;
    letter_spacing: number;
  };
  colors: {
    active: string;
    past: string;
    ghost: string;
    past_opacity: number;
    ghost_opacity: number;
    accent: string;
    shout: string;
    outline: string;
    shadow: string;
  };
  panel: {
    enabled: boolean;
    color: string;
    opacity: number;
    blur: number;
    radius: number;
    padding_x: number;
    padding_y: number;
    border_color: string;
    border_opacity: number;
    border_width: number;
  };
  caption: {
    position: Position;
    margin: number;
    max_line_chars: number;
    line_height: number;
    align: "center" | "left";
  };
  animation: {
    pop_scale: number;
    pop_ms: number;
    spring_damping: number;
    spring_stiffness: number;
    spring_mass: number;
    block_in_ms: number;
    block_out_ms: number;
    accent_scale: number;
    shout_scale: number;
    shout_rotate: number;
  };
  background: string;
};

export type RenderPlan = {
  version: number;
  fps: number;
  width: number;
  height: number;
  duration: number;
  /** Путь относительно публичной директории бандла. */
  video: string;
  audio: string | null;
  preset: Preset;
  blocks: PlanBlock[];
  broll: PlanBroll[];
  camera: PlanCamera[];
  meta: Record<string, unknown>;
  /** Шрифт: путь относительно публичной директории. Ставится в рендер, не в систему. */
  fontUrl?: string | null;
};

export const EMPTY_PLAN: RenderPlan = {
  version: 1,
  fps: 30,
  width: 1080,
  height: 1920,
  duration: 5,
  video: "",
  audio: null,
  blocks: [],
  broll: [],
  camera: [],
  meta: {},
  fontUrl: null,
  preset: {
    name: "glass",
    title: "Glass",
    font: {
      family: "Montserrat ExtraBold",
      file: null,
      weight: 800,
      size: 88,
      uppercase: false,
      letter_spacing: -0.015,
    },
    colors: {
      active: "#FFFFFF",
      past: "#FFFFFF",
      ghost: "#FFFFFF",
      past_opacity: 0.7,
      ghost_opacity: 0.3,
      accent: "#FFD54A",
      shout: "#FF6A3D",
      outline: "#05070A",
      shadow: "#000000",
    },
    panel: {
      enabled: true,
      color: "#0B0F14",
      opacity: 0.32,
      blur: 18,
      radius: 30,
      padding_x: 34,
      padding_y: 22,
      border_color: "#FFFFFF",
      border_opacity: 0.14,
      border_width: 1,
    },
    caption: {
      position: "bottom",
      margin: 380,
      max_line_chars: 15,
      line_height: 1.06,
      align: "center",
    },
    animation: {
      pop_scale: 1.16,
      pop_ms: 160,
      spring_damping: 12,
      spring_stiffness: 190,
      spring_mass: 0.7,
      block_in_ms: 180,
      block_out_ms: 140,
      accent_scale: 1.18,
      shout_scale: 1.34,
      shout_rotate: -4,
    },
    background: "#000000",
  },
};
