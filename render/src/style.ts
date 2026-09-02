/** Мелкие помощники стиля. Никакой логики монтажа здесь нет. */

export const rgba = (hex: string, opacity = 1): string => {
  let v = hex.replace("#", "");
  if (v.length === 3) {
    v = v
      .split("")
      .map((c) => c + c)
      .join("");
  }
  const r = parseInt(v.slice(0, 2), 16);
  const g = parseInt(v.slice(2, 4), 16);
  const b = parseInt(v.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, opacity))})`;
};

/** Плавная ease-out кривая для входов, где пружина была бы слишком игривой. */
export const easeOut = (t: number): number => 1 - Math.pow(1 - Math.max(0, Math.min(1, t)), 3);
