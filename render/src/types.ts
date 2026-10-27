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

/**
 * Прямоугольник в пикселях кадра, левый верхний угол — начало координат.
 *
 * Считает его Python (`src/core/geometry.py`) по метрикам шрифта: до этого
 * геометрия жила константами здесь, в компонентах, и Python не мог предсказать,
 * куда попадёт строка. Необязательный: план, собранный до Ф4, прямоугольников
 * не содержит и раскладывается по-старому.
 */
export type Rect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

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
  /** Куда блок попадёт в кадре вместе с панелью. */
  rect?: Rect | null;
  /** Средняя яркость кадра под блоком, 0..1. Меряет Python (Ф9). */
  bg_luma?: number | null;
  /**
   * Носит ли ЭТОТ блок матовое стекло. В эталоне плашка есть у двух блоков из
   * тринадцати, и решение о ней режиссёрское — правило по фону измерить не
   * удалось. Место под неё зарезервировано у всех блоков одинаково, поэтому
   * флаг меняет только вид: заливку, размытие и цвет рамки.
   */
  panel?: boolean;
  /**
   * Путь блока, привязанного к голове (М2): смещение от прямоугольника покоя в
   * пикселях кадра, по кадрам от `start`. Считает Python вместе с
   * предохранителями и плавным откатом — рендерер только выбирает элемент.
   */
  anchor_path?: number[][] | null;
};

export type PlanBroll = {
  start: number;
  end: number;
  anim: BrollAnim;
  anchor: Anchor;
  asset: string | null;
  query: string;
  /** Место картинки в покое: `fullscreen_to_inset` едет с полного кадра сюда. */
  rect?: Rect | null;
  /**
   * Направление медленного наезда, −1 или 1 (М3). Считает Python из позиции
   * вставки в ролике. `Math.random` здесь запрещён: перерендер обязан давать
   * то же самое видео.
   */
  kb_dir?: number;
  /** Фото или клип (М4). Решает добыча, а не режиссёр: видео есть не на всякий запрос. */
  kind?: "photo" | "video";
  /** С какой секунды играть клип. У фото 0 — кадр выбирает Python, не рендерер. */
  asset_start?: number;
  /**
   * Пропорция файла (ширина/высота). Нужна ровно для одного решения: класть ли
   * под полнокадровое фото размытую копию. Стоковые 2:3 без неё режутся,
   * нарисованная вертикаль 9:16 с ней только портится. По `rect` этого не
   * понять — у полного кадра он всегда кадр целиком.
   */
  aspect?: number | null;
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
    /** Матовое стекло: украшение помеченных блоков, не предохранитель Ф9. */
    glass_color: string;
    glass_opacity: number;
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
    /** Ход движения в пикселях кадра 1920; пересчитывает Preset.scaled(). */
    block_shift_px?: number;
    block_leave_px?: number;
    broll_slide_px?: number;
    block_overlap_min_ms?: number;
    block_leave_falloff?: number;
    block_leave_scale?: number;
    /** Пауза с пустым экраном между блоками; > 0 — блок уходит ДО конца (С4). */
    block_gap_ms?: number;
    /** Наезд блока на входе; 0 — масштабной анимации нет вовсе (С4). */
    block_in_zoom?: number;
    /** Ken Burns на вставке: наезд и сползание за всю жизнь картинки (М3). */
    kb_scale?: number;
    kb_shift_ratio?: number;
    /** Вспышка света на входе полноэкранной вставки; 0 — нет. Оттенок в градусах. */
    leak_ms?: number;
    leak_hue?: number;
    /**
     * Сколько слово ПРОЯВЛЯЕТСЯ, когда прозвучало. 0 — мгновенно, как было.
     *
     * У эталона это 4 кадра (~130 мс): слово набирает яркость, а место под
     * него зарезервировано заранее, и соседи не шевелятся. У нас слово
     * включалось за один кадр — единственное измеримое расхождение с
     * эталоном, которое осталось после С1-С6.
     */
    word_in_ms?: number;
    /**
     * Насколько слово выезжает снизу, пока проявляется, в пикселях кадра 1920.
     * Только `transform`: раскладку считает Python и сверяет с Chromium с
     * точностью 2 px, поэтому двигать можно, а менять кегль и трекинг — нет.
     */
    word_rise_px?: number;
  };
  /** Геометрия в долях кадра: раньше эти числа были константами в компонентах. */
  geometry: {
    max_line_width_ratio: number;
    safe_area_ratio: number;
    broll_inset_ratio: number;
    broll_margin_ratio: number;
    broll_top_ratio: number;
    broll_bottom_ratio: number;
  };
  /**
   * Секции, которые исполняет не Remotion, а ffmpeg и Python: цветокор, кроп,
   * звуковой пост. В плане они есть всегда (пресет уезжает в него целиком),
   * рендереру не нужны — поэтому необязательные.
   */
  grade?: {
    enabled: boolean;
    contrast: number;
    saturation: number;
    gamma: number;
    vignette: number;
  };
  crop?: {
    face_height_ratio: number;
    tight_face_height_ratio: number;
    eye_line_ratio: number;
    headroom_ratio: number;
    max_scale: number;
  };
  audio?: {
    target_lufs: number;
    music_gain_db: number;
    duck_db: number;
    sfx_gain_db: number;
    music_fade_s: number;
    music: string | null;
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
      ghost_opacity: 0.0,
      accent: "#FFD54A",
      shout: "#FF6A3D",
      outline: "#05070A",
      shadow: "#000000",
    },
    panel: {
      enabled: true,
      color: "#0B0F14",
      opacity: 0.5,
      blur: 24,
      radius: 30,
      padding_x: 34,
      padding_y: 22,
      border_color: "#FFFFFF",
      border_opacity: 0.14,
      border_width: 1,
      glass_color: "#FFFFFF",
      glass_opacity: 0,
    },
    caption: {
      position: "bottom",
      margin: 380,
      max_line_chars: 15,
      line_height: 1.06,
      align: "center",
    },
    animation: {
      pop_scale: 1.05,
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
    geometry: {
      max_line_width_ratio: 0.78,
      safe_area_ratio: 0.07,
      broll_inset_ratio: 0.44,
      broll_margin_ratio: 0.05,
      broll_top_ratio: 0.16,
      broll_bottom_ratio: 0.58,
    },
    background: "#000000",
  },
};
