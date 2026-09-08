/**
 * Пословные субтитры: три состояния слова, glass-панель, контейнер растёт под
 * контент сам.
 *
 * Анимация — код, а не модель (§0.2). Пресет задаёт только числа.
 *
 * Всё время считается в кадрах от округлённых до сетки границ. Секунды в плане
 * лежат с точностью до миллисекунды и между кадрами не попадают, поэтому по
 * времени первый отрисованный кадр приходился на середину подъёма: слово
 * стартовало с альфы 0.09-0.45, и момент появления гулял от ролика к ролику.
 *
 * Раскладка целиком приезжает из Python (Ф5). Три следствия, ради которых это
 * сделано:
 *
 * 1. Акцент — это `fontSize`, а не `transform: scale`. Масштаб не меняет
 *    layout-ширину, поэтому увеличенное слово просто накрывало соседа и
 *    оставалось увеличенным до конца блока (AUDIT §8.1). Кегль меряется
 *    браузером, и наезда нет по построению.
 * 2. Слова разделяет НАСТОЯЩИЙ пробел кеглем строки, а не flex-`gap`: `gap`
 *    не растёт вместе с акцентом, и Python не может его предсказать. Тем же
 *    пробелом меряет строку `src/core/geometry.py` и линейка `measure.mjs`.
 * 3. Перенос строк уже сделан: строка гарантированно влезает в бюджет, поэтому
 *    здесь стоит `nowrap` и никто ничего не режет. Ровно до тех пор, пока в
 *    блоке есть прямоугольник: без него (метрик шрифта не было) Python ширину
 *    не считал, и переносит браузер.
 */

import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import type { PlanBlock, PlanWord, Preset } from "../types";
import {
  MIN_ANIM_FRAMES,
  animFrames,
  animNum,
  easeOut,
  motionPx,
  popBump,
  rgba,
  springFrames,
} from "../style";

type WordState = "ghost" | "active" | "past";

const stateOf = (word: PlanWord, frame: number, fps: number): WordState => {
  if (frame < Math.round(word.start * fps)) return "ghost";
  if (frame <= Math.round(word.end * fps)) return "active";
  return "past";
};

/**
 * Хвост блока в кадрах: столько он живёт после своего конца. Считается в одном
 * месте — им живут и Block, и фильтр видимости.
 *
 * Нижняя граница нахлёста приезжает из пресета (`block_overlap_min_ms`): блок
 * обязан быть ещё виден, когда пришёл новый, иначе между репликами моргает
 * пустой кадр, а 100-140 мс пресетов на это не хватает ни на одном fps.
 *
 * Известная цена: `animFrames` держит пол в восемь кадров, и на 24 fps хвост
 * выходит 333 мс вместо заказанных планом 200-250 мс (на 30 fps — 267 мс, на
 * 60 — 217 мс). Два закона здесь спорят, и спор решается в пользу восьми
 * кадров: полоса в миллисекундах — это про то, что уходящий блок ещё виден, а
 * восемь кадров — про то, что его уход вообще читается как движение, а не как
 * мигание. Сама альфа при этом падает быстрее окна: к середине хвоста она уже
 * 0.06, так что ЧИТАЕМЫЙ нахлёст держится в полосе, а за неё выходит только
 * формальное окно.
 */
const blockOutFrames = (preset: Preset, fps: number): number => {
  const a = preset.animation;
  return animFrames(Math.max(a.block_out_ms, animNum(a, "block_overlap_min_ms", 220)), fps);
};

const emphasisScale = (word: PlanWord, preset: Preset): number => {
  if (word.emphasis === "accent") return preset.animation.accent_scale;
  if (word.emphasis === "shout") return preset.animation.shout_scale;
  return 1;
};

/**
 * Кегль слова в пикселях.
 *
 * Считает его Python: он же им мерил строку и решал перенос, и длинное слово,
 * ужатое под ширину строки, известно только ему. План, собранный до Ф5, поля
 * не несёт — тогда кегль выводится из выделения здесь, но всё равно как
 * РАЗМЕР, а не как масштаб. Поле объявлено в `types.ts` (чужой файл) ещё не
 * будет, поэтому читаем через узкий каст.
 */
const wordSize = (word: PlanWord, preset: Preset): number => {
  const own = (word as PlanWord & { size?: number | null }).size;
  return typeof own === "number" && own > 0 ? own : preset.font.size * emphasisScale(word, preset);
};

// ------------------------------------------------------------------ контраст (Ф9)

/**
 * Правило контраста из пресета. Секции ещё нет в `types.ts` (чужой файл), а
 * пресет уезжает в план целиком — читаем через узкий каст.
 */
type ContrastRule = {
  enabled: boolean;
  min_ratio: number;
  luma_dark: number;
  luma_light: number;
  panel_opacity_dark: number;
  panel_opacity_light: number;
  text_dark: string;
  text_light: string;
};

const contrastRule = (preset: Preset): ContrastRule | null => {
  const rule = (preset as Preset & { contrast?: ContrastRule }).contrast;
  return rule && rule.enabled ? rule : null;
};

const clamp01 = (v: number): number => Math.max(0, Math.min(1, v));

const channels = (hex: string): [number, number, number] => {
  let v = hex.replace("#", "");
  if (v.length === 3) {
    v = v
      .split("")
      .map((c) => c + c)
      .join("");
  }
  return [
    parseInt(v.slice(0, 2), 16) / 255,
    parseInt(v.slice(2, 4), 16) / 255,
    parseInt(v.slice(4, 6), 16) / 255,
  ];
};

const toHex = (rgb: [number, number, number]): string =>
  "#" +
  rgb
    .map((c) =>
      Math.round(clamp01(c) * 255)
        .toString(16)
        .padStart(2, "0"),
    )
    .join("");

/** sRGB → линейный свет и обратно: смешивать и сравнивать яркости можно только там. */
const toLinear = (c: number): number =>
  c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
const toSrgb = (l: number): number =>
  l <= 0.0031308 ? l * 12.92 : 1.055 * Math.pow(l, 1 / 2.4) - 0.055;

/** Относительная яркость по WCAG. `bg_luma` из плана — уже она, мерить нечего. */
const luminance = (hex: string): number => {
  const [r, g, b] = channels(hex).map(toLinear);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};

const contrastRatio = (a: number, b: number): number =>
  (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);

const mixHex = (a: string, b: string, t: number): string => {
  const ca = channels(a);
  const cb = channels(b);
  return toHex([0, 1, 2].map((i) => ca[i] + (cb[i] - ca[i]) * t) as [number, number, number]);
};

/**
 * Минимальная прозрачность панели, при которой контраст берёт порог.
 *
 * Панель кладётся альфой поверх фона, и Chromium смешивает их в sRGB, поэтому
 * решается уравнение тоже в sRGB: яркость под текстом линейна по альфе именно
 * там. Светлый текст просит фон темнее порога, тёмный — светлее; знак решает
 * сравнение с яркостью самой панели.
 */
const panelOpacityFor = (
  bgLuma: number,
  panelLuma: number,
  textLuma: number,
  minRatio: number,
): number => {
  const target =
    textLuma > panelLuma
      ? (textLuma + 0.05) / minRatio - 0.05
      : minRatio * (textLuma + 0.05) - 0.05;
  const from = toSrgb(clamp01(bgLuma));
  const to = toSrgb(clamp01(panelLuma));
  if (Math.abs(to - from) < 1e-6) return 0;
  return clamp01((toSrgb(clamp01(target)) - from) / (to - from));
};

type Look = { panelOpacity: number; text: string };

/**
 * Прозрачность панели и цвет обычного слова под конкретным фоном.
 *
 * Между опорными точками пресета ведём линейно — это стиль. Дальше считаем
 * получившийся контраст и, если он ниже порога, включаем закон, и оба рычага
 * работают в паре:
 *
 * 1. Панель поднимается ровно до порога (там, где панель вообще есть).
 * 2. Цвет текста переключается на ту опорную точку, которая на этом фоне даёт
 *    БОЛЬШИЙ контраст — ступенькой, а не смесью.
 *
 * Второй рычаг обязателен, иначе у пресета без панели правило не просто не
 * работает, а вредит: линейная смесь #FFFFFF → #0A0D12 проходит через яркость
 * самого фона, и на bg_luma 0.40 в bold контраст падал до 1.05 — текст сливался
 * с фоном, тогда как белый без всякого правила давал 2.33. Ниже порога 4.5
 * сидел весь диапазон 0.20…0.65, то есть обычное дневное видео.
 *
 * Гарантия держится на данных пресета: опорные точки обязаны расходиться
 * настолько, чтобы на любом фоне хоть одна брала порог (для bold это #FFFFFF и
 * #000000: в худшей точке bg ≈ 0.19 обе дают 4.58). Это проверяет
 * test_contrast_rule_has_a_lever_on_every_background.
 *
 * `bg_luma = null` — никто не мерил, пресет играет как играл.
 */
const lookFor = (preset: Preset, bgLuma: number | null | undefined): Look => {
  const base: Look = { panelOpacity: preset.panel.opacity, text: preset.colors.active };
  const rule = contrastRule(preset);
  if (!rule || bgLuma == null || !isFinite(bgLuma)) return base;

  const span = rule.luma_light - rule.luma_dark;
  const t = span > 1e-6 ? clamp01((bgLuma - rule.luma_dark) / span) : bgLuma > rule.luma_dark ? 1 : 0;
  const styled = mixHex(rule.text_dark, rule.text_light, t);
  const styledOpacity =
    rule.panel_opacity_dark + (rule.panel_opacity_light - rule.panel_opacity_dark) * t;

  const panelLuma = luminance(preset.panel.color);
  /** Яркость ПОД текстом: фон, прикрытый панелью этой прозрачности. */
  const under = (opacity: number): number =>
    preset.panel.enabled
      ? toLinear(toSrgb(clamp01(bgLuma)) + (toSrgb(panelLuma) - toSrgb(clamp01(bgLuma))) * opacity)
      : clamp01(bgLuma);

  /** Цвет + минимальная панель, при которой он берёт порог (панели нет — как есть). */
  const attempt = (text: string): Look & { ratio: number } => {
    const textLuma = luminance(text);
    let opacity = styledOpacity;
    if (preset.panel.enabled && contrastRatio(textLuma, under(opacity)) < rule.min_ratio) {
      opacity = Math.max(opacity, panelOpacityFor(bgLuma, panelLuma, textLuma, rule.min_ratio));
    }
    opacity = clamp01(opacity);
    return { panelOpacity: opacity, text, ratio: contrastRatio(textLuma, under(opacity)) };
  };

  const stylish = attempt(styled);
  if (stylish.ratio >= rule.min_ratio) {
    return { panelOpacity: stylish.panelOpacity, text: stylish.text };
  }
  // Порог не взят ни панелью, ни стилевой смесью — берём лучшую опорную точку.
  // Хуже обеих опорных точек не отдаём НИКОГДА: это и есть провал середины.
  return [stylish, attempt(rule.text_dark), attempt(rule.text_light)].reduce((a, b) =>
    b.ratio > a.ratio ? b : a,
  );
};

// ------------------------------------------------------------------ слово

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

const Word: React.FC<{ word: PlanWord; preset: Preset; neutral: string }> = ({
  word,
  preset,
  neutral,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const state = stateOf(word, frame, fps);
  const a = preset.animation;

  // Пружина стартует в кадре, когда слово прозвучало. Ghost-слова уже
  // нарисованы на своих местах и своим кеглем, поэтому контейнер не прыгает.
  const framesSinceStart = frame - Math.round(word.start * fps);

  // Транзиентный «пшик» и ничего больше. Раньше сюда домножался accent_scale,
  // и слово навсегда застывало увеличенным: пик 1.53, постоянный наезд до
  // 122 px (AUDIT §8.1). Теперь размер — это layout, а масштаб живёт восемь
  // кадров (`MIN_ANIM_FRAMES` длиннее, чем pop_ms 130-160 в пресетах: на 30 fps
  // это 267 мс) и на 1.05 физически не может дотянуться до соседа.
  const bump = state === "ghost" ? 0 : popBump(framesSinceStart / animFrames(a.pop_ms, fps));
  const scale = 1 + (a.pop_scale - 1) * bump;

  // Угол умножается на пружину НАПРЯМУЮ, без `interpolate`. Раньше её значение
  // проходило через `interpolate(..., { extrapolateRight: "clamp" })`, и весь
  // перелёт выбрасывался: пружина выходит на 1.0 к четвёртому кадру и дальше
  // 5-19 кадров стоит полкой на своём максимуме (замер: glass 24 fps — пик
  // 1.143, пять кадров зажаты; bold 60 fps — пик 1.224, двадцать два кадра).
  // То есть длительность привели к натуральной, а упругость, ради которой это
  // делали, срезали клампом — и видимого движения оставалось 3-4 кадра при
  // законе в восемь (AUDIT §8.3). Теперь угол проскакивает цель на 14-22 % и
  // возвращается, как соседний Broll.tsx, который клампа на выходе не ставит.
  const rotate =
    word.emphasis === "shout" && state !== "ghost"
      ? a.shout_rotate *
        spring({
          frame: Math.max(0, framesSinceStart),
          fps,
          durationInFrames: springFrames(fps, {
            damping: a.spring_damping,
            stiffness: a.spring_stiffness,
            mass: a.spring_mass,
          }),
          config: {
            damping: a.spring_damping,
            stiffness: a.spring_stiffness,
            mass: a.spring_mass,
          },
        })
      : 0;

  // Акцент и крик держат свой цвет при любом фоне: это опознавательный знак
  // стиля. Контраст правит только обычное слово. Цена решения известна: в
  // пресете без панели (bold) салатовый #28E07B над светлым фоном (bg_luma
  // 0.85) даёт контраст 1.34 при пороге 4.5 — выделенное слово там держится
  // только на тени. Дать ему рычаг можно лишь данными (подложка или своя пара
  // цветов в contrast); пока этого нет, знание живёт здесь.
  //
  // Цвет держится за РОЛЬЮ слова, а не за его состоянием (С2). Пока он жил
  // только в `active`, золото вспыхивало на слове ровно на время его
  // произнесения и тут же сменялось белым: в кадре 55 эталонного плана
  // «человек» уже белый, хотя выделен. У эталона выделенное слово держит свой
  // цвет весь блок, а состояние выражено присутствием — слова ещё нет или оно
  // уже стоит в полную силу.
  const spoken = state !== "ghost";
  const color =
    word.emphasis === "accent" && spoken
      ? preset.colors.accent
      : word.emphasis === "shout" && spoken
        ? preset.colors.shout
        : neutral;

  // ghost_opacity гасит слово, но не убирает его из потока: место под него
  // зарезервировано, и панель не меняет размер, когда слово зазвучит.
  const opacity =
    state === "active"
      ? 1
      : state === "past"
        ? preset.colors.past_opacity
        : preset.colors.ghost_opacity;

  return (
    <span
      style={{
        display: "inline-block",
        fontSize: wordSize(word, preset),
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
  const { fps, width, durationInFrames } = useVideoConfig();
  const a = preset.animation;

  const startFrame = Math.round(block.start * fps);
  const endFrame = Math.round(block.end * fps);
  const inFrames = animFrames(a.block_in_ms, fps);
  // Хвост обязан уложиться В КОМПОЗИЦИЮ. Её длину задаёт `plan.duration`, и она
  // ничего не знает про хвост: длины хвоста нет ни в EDL, ни в плане, её решает
  // рендерер. У последнего блока эталонного ролика end приходится на кадр 441
  // при 443 кадрах композиции — из восьми кадров ухода рисуется ОДИН, и ролик
  // обрывается на альфе 0.58 жёстким срезом. Поэтому: места хватает — уходим
  // как обычно; места меньше восьми кадров — не уходим ВОВСЕ и честно стоим до
  // конца композиции. Огрызок анимации хуже её отсутствия — так же решает
  // Broll.tsx при `travel < MIN_ANIM_FRAMES`, и так же велит DoD Ф8: ни одна
  // анимация не короче восьми кадров. Настоящее лекарство — продлить
  // композицию на хвост последнего блока, но это `render_plan.py`.
  const room = durationInFrames - 1 - endFrame;
  const outFrames = room >= MIN_ANIM_FRAMES ? Math.min(blockOutFrames(preset, fps), room) : 0;

  // Блок появляется от своего начала и уходит ПОСЛЕ своего конца. Раньше он
  // гас за block_out_ms до конца, то есть таял прямо на последнем слове, а
  // между соседними блоками (`a.end == b.start` после build_plan) оставалась
  // дыра. Теперь уходящий блок ещё виден, когда приходит новый: нахлёст равен
  // outFrames. Обе кривые одной природы — тормозят к своему конечному
  // состоянию, поэтому выход не линейный и не щёлкает.
  //
  // Цена нахлёста: блоки идут ВСТЫК и стоят в одном месте кадра, поэтому всё
  // время нахлёста два текста лежат друг на друге. Чистая перекрёстная
  // прозрачность это не лечит: сумма альф обязана держаться около единицы
  // (иначе кадр моргает), значит в точке пересечения обе кривые не ниже 0.4 —
  // и оба текста читаются. Замерено глазом на кадре 228 эталонного прогона:
  // «когда есть / несколько» поверх «других целей», две плашки разной ширины с
  // двойным краем. Поэтому уходящий блок не просто гаснет, а УХОДИТ: смещается
  // от места приходящего и уменьшается, а его альфа падает быстрее.
  //
  // Показатель падения берётся ОТ ОТНОШЕНИЯ хвоста ко входу, а не как есть.
  // Голое число подобрано под частный случай inFrames == outFrames (24 и
  // 30 fps): точка пересечения кривых опускается с 0.500 до 0.448. Но хвост и
  // вход округляются до кадров по-разному, и на 60 fps выходит 11 против 13 —
  // уходящий блок держится дольше приходящего, пересечение уползает обратно к
  // 0.480, а у пресета без нижней границы входа (bold: 8 против 13) — к 0.541,
  // то есть правило почти перестаёт работать ровно там, где кадров больше
  // всего. Домножение на outFrames/inFrames делает точку пересечения
  // инвариантной к fps и к пресету: 0.448 / 0.451 / 0.455 на тех же трёх парах.
  const appear = easeOut(
    interpolate(frame, [startFrame, startFrame + inFrames], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }),
  );
  const falloff = animNum(a, "block_leave_falloff", 1.35) * (outFrames / inFrames);
  const disappear =
    outFrames === 0
      ? 1
      : Math.pow(
          1 -
            easeOut(
              interpolate(frame, [endFrame, endFrame + outFrames], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            ),
          falloff,
        );
  const presence = Math.min(appear, disappear);
  if (presence <= 0) return null;

  const p = preset.panel;
  const shift = motionPx(a, "block_shift_px", Math.round(preset.font.size * 0.28));
  // Уходим ОТ приходящего блока: он встаёт на то же место, поэтому вниз для
  // нижних и центральных, вверх для верхних — то есть к своему краю кадра.
  //
  // Ход — не меньше СОБСТВЕННОЙ высоты блока. Числом из пресета эту задачу не
  // решить: 64 px при межстрочном шаге 127 px и высоте двухстрочного блока
  // 254 px уводят уходящий текст на пол-строки, и его глифы всё равно ложатся
  // на приходящую строку (кадр 228 эталонного прогона: «целей» перечёркнуто
  // буквами «несколько»). Высота приезжает в плане прямоугольником, считана по
  // тем же метрикам шрифта и не зависит ни от кегля, ни от числа строк —
  // поэтому она и есть гарантия, что блоки разошлись. Пресетное число остаётся
  // нижней границей: оно работает там, где прямоугольника нет.
  const leave = Math.max(
    motionPx(a, "block_leave_px", Math.round(preset.font.size * 0.73)),
    block.rect ? block.rect.height : 0,
  );
  const leaveDir = block.position === "top" ? -1 : 1;
  const offsetY = (1 - appear) * shift + (1 - disappear) * leave * leaveDir;
  const leaveScale = animNum(a, "block_leave_scale", 0.9);
  const zoom = (0.965 + appear * 0.035) * (leaveScale + (1 - leaveScale) * disappear);
  const look = lookFor(preset, (block as PlanBlock & { bg_luma?: number | null }).bg_luma);
  // Прямоугольник блока посчитан в Python по этим же метрикам: строка в бюджет
  // уложена, и ограничивать ширину здесь незачем. План без прямоугольника
  // (пресет без метрик шрифта) — другое дело: там перенос по символам, ширину
  // строки никто не мерил, и защита от строки во весь кадр только здесь. Одного
  // потолка ширины для этого мало: строка стоит `nowrap`, `overflow` не задан,
  // и текст спокойно вылезает за зажатую панель (замерено: панель 842 px,
  // строка внутри 1186 px). Поэтому на этом пути переносит браузер — см.
  // `whiteSpace` ниже.
  //
  // Бюджет тот же, которым меряет основной путь: `geometry.max_line_width_ratio`
  // от ширины кадра плюс хром панели (`geometry.caption_text_width` вычитает его
  // из бюджета ТЕКСТА, а здесь ограничивается контейнер вместе с паддингами).
  // Раньше здесь стояли 88 % — константа раскладки мимо пресета, из-за которой
  // запасной путь переносил строку по одному бюджету, а Python по другому.
  const chrome = p.enabled ? 2 * (p.padding_x + p.border_width) : 0;
  const maxWidth = block.rect
    ? undefined
    : Math.round(width * preset.geometry.max_line_width_ratio) + chrome;

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
        background: p.enabled ? rgba(p.color, look.panelOpacity) : "transparent",
        backdropFilter: p.enabled && p.blur > 0 ? `blur(${p.blur}px) saturate(1.4)` : undefined,
        WebkitBackdropFilter:
          p.enabled && p.blur > 0 ? `blur(${p.blur}px) saturate(1.4)` : undefined,
        border:
          p.enabled && p.border_width > 0
            ? `${p.border_width}px solid ${rgba(p.border_color, p.border_opacity)}`
            : undefined,
        opacity: presence,
        transform: `translateY(${offsetY}px) scale(${zoom})`,
        maxWidth,
        willChange: "transform, opacity",
      }}
    >
      {block.lines.map((line, li) => (
        <div
          key={li}
          style={{
            // Строка — один инлайновый бокс: слова разделяет настоящий пробел
            // кеглем строки. Перенос уже сделан в Python — но только там, где
            // есть прямоугольник. Без него ширину не мерил никто, и переносить
            // обязан браузер, иначе строка запасного пути уезжает за кадр.
            whiteSpace: block.rect ? "nowrap" : "normal",
            textAlign: preset.caption.align,
          }}
        >
          {line.map((word, wi) => (
            <React.Fragment key={word.word_idx}>
              {wi > 0 ? " " : null}
              <Word word={word} preset={preset} neutral={look.text} />
            </React.Fragment>
          ))}
        </div>
      ))}
    </div>
  );
};

export const Captions: React.FC<{ blocks: PlanBlock[]; preset: Preset }> = ({ blocks, preset }) => {
  const frame = useCurrentFrame();
  const { fps, height } = useVideoConfig();
  // Безопасная зона у края кадра: пресет может попросить margin 0, но прижатый
  // к краю текст обрезается плеером и интерфейсом ленты. Доля приезжает из
  // пресета — тем же числом её знает `geometry.safe_margin_y`.
  const margin = Math.max(
    preset.caption.margin,
    Math.round(height * preset.geometry.safe_area_ratio),
  );
  // Хвост ровно тот же, что у Block: блок живёт на outFrames дольше своего
  // конца и обязан пережить фильтр, иначе нахлёст срежется здесь.
  const tail = blockOutFrames(preset, fps);

  const visible = blocks.filter(
    (b) => frame >= Math.round(b.start * fps) && frame <= Math.round(b.end * fps) + tail,
  );
  if (visible.length === 0) return null;

  const justify = (position: string) =>
    position === "top" ? "flex-start" : position === "center" ? "center" : "flex-end";

  /**
   * Смещение блока, привязанного к голове (М2). Путь приходит ГОТОВЫМ: его
   * считает `src/core/anchor.py` вместе с предохранителями, мёртвой зоной и
   * плавным откатом. Здесь только выборка по кадру — рендерер решений не
   * принимает (§0), и физика привязки не исключение: кадров конечное число,
   * трек известен заранее, значит и путь известен заранее.
   *
   * Индекс зажимается в конец: блок живёт на `tail` кадров дольше своего конца
   * (нахлёст с приходящим), а путь посчитан по самому блоку. Держать последнее
   * смещение во время ухода правильнее, чем возвращать блок в покой: он в этот
   * момент уже уезжает и растворяется, и лишний скачок был бы виден.
   */
  const anchorShift = (block: PlanBlock): [number, number] => {
    const path = block.anchor_path;
    if (!path || path.length === 0) return [0, 0];
    const i = Math.min(path.length - 1, Math.max(0, frame - Math.round(block.start * fps)));
    const point = path[i];
    return [point[0] ?? 0, point[1] ?? 0];
  };

  return (
    <>
      {/* Порядок сохраняем: во время нахлёста уходящий блок лежит под приходящим. */}
      {visible.map((block) => (
        <div
          // Ключ — начало блока, а не его номер в ОТФИЛЬТРОВАННОМ списке: номер
          // сдвигается, как только уходящий блок выпадает из фильтра, и React
          // пересоздаёт узел приходящего прямо посреди его появления.
          key={block.start}
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
            // Смещение живёт на контейнере, а не на самом блоке: у блока свой
            // transform под появление и уход, и складывать два движения в одну
            // строку значило бы, что правка одного молча ломает другое.
            transform: (([dx, dy]) =>
              dx || dy ? `translate(${dx}px, ${dy}px)` : undefined)(anchorShift(block)),
            willChange: block.anchor_path ? "transform" : undefined,
          }}
        >
          <Block block={block} preset={preset} />
        </div>
      ))}
    </>
  );
};
