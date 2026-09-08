import { measureSpring, spring, interpolate } from "remotion";
const cfgs = {
  glass: { damping: 12.0, stiffness: 190.0, mass: 0.7 },
  bold: { damping: 10.0, stiffness: 210.0, mass: 0.65 },
};
for (const [name, config] of Object.entries(cfgs)) {
  for (const fps of [24, 30, 60]) {
    const natural = measureSpring({ fps, config });
    const dur = Math.max(8, natural);
    const raw = [];
    for (let f = 0; f <= dur + 4; f++) raw.push(spring({ frame: f, fps, config, durationInFrames: dur }));
    const max = Math.max(...raw);
    const first1 = raw.findIndex((v) => v >= 1);
    const clamped = raw.filter((v) => v > 1).length;
    console.log(name, "fps", fps, "natural", natural, "dur", dur, "max", max.toFixed(4),
      "первый >=1.0 кадр", first1, "кадров выше 1.0", clamped, "/", raw.length);
    console.log("   raw:", raw.map((v) => v.toFixed(3)).join(" "));
  }
}
