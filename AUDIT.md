# ReelBot — разбор для консультации по качеству картинки

## Что это за документ и что я прошу

Я делаю телеграм-бота, который автоматически монтирует вертикальные ролики
(Reels / TikTok / Shorts): пословные анимированные субтитры, вырезание пауз,
автокроп 9:16, вставка картинок по смыслу.

Технически всё работает. **Проблема в том, что выглядит это хуже, чем у
коммерческих инструментов того же класса** (Submagic, Opus Clip, Captions.ai).
Три конкретные жалобы, которые видно глазом:

1. Акцентное слово наезжает на соседнее и перекрывает его.
2. Картинки вставляются криво и резко.
3. Экран дёргается на ключевых словах.

Я провёл аудит и нашёл механизмы всех трёх (раздел 7). Но я хочу не только
починить эти три, а понять, **чего системно не хватает, чтобы результат
выглядел дорого**.

### Как я хочу вести разговор

Не выдавай сразу готовый патч и не вываливай список из тридцати пунктов.

Сначала **задай мне вопросы**, если чего-то не хватает для совета: про
референсы, которые мне нравятся, про приоритеты, про то, чем я готов
пожертвовать. Потом предложи направления с развилками и объясни компромиссы,
чтобы я выбрал. Разбирайся по одному блоку за раз.

Отдельно интересует твоё мнение о том, чего в списке нет вообще, но что делают
все хорошие инструменты, а я упустил.

---

## 1. Продукт

Пользователь присылает боту горизонтальное или вертикальное видео (до 3 минут,
говорящая голова). Бот возвращает готовый вертикальный ролик 1080×1920 с
субтитрами, вырезанными паузами и вставленными картинками.

Целевая эстетика: то, что сейчас считается стандартом в коротком вертикальном
видео — крупные читаемые пословные субтитры с подсветкой текущего слова,
аккуратные акценты, ненавязчивое движение.

---

## 2. Архитектурный принцип

```
Восприятие (детерминированное)  →  LLM-режиссёр (JSON)  →  Рендерер (тупой)
```

**EDL (Edit Decision List)** — контракт между слоями.

Инварианты, которые я ломать не хочу (это фундамент, на нём всё держится):

1. **LLM не возвращает таймкоды.** Только индексы слов. Время подставляет Python
   из вывода распознавания. Модели ошибаются в арифметике с float — это источник
   съехавших субтитров.
2. **Анимация — это код, а не модель.** Всё детерминированно.
3. **Пресет стиля — данные (JSON), а не код.** Новый стиль не должен требовать
   правки компонентов.
4. **Любой EDL переживает перерендер.** Сохранённый EDL + другой пресет = новое
   видео без повторного распознавания и LLM.

Если совет требует нарушить один из этих пунктов — скажи прямо и объясни, почему
оно того стоит.

---

## 3. Стек

| Слой | Технология |
|---|---|
| Бот | Python 3.12, aiogram 3 |
| Очередь | arq + Redis |
| БД | PostgreSQL + SQLAlchemy 2 (async) |
| Хранилище | MinIO (S3) |
| Распознавание | faster-whisper `small`, CPU, русский |
| LLM | Gemini через свой абстрактный клиент (есть также Anthropic и OpenAI-совместимые) |
| Видео-препроцесс | ffmpeg |
| **Рендер** | **Remotion 4.0.520 + React 18 (Node)** |
| Картинки | Pexels API → фолбэк на генерацию (WaveSpeed) |

Python владеет пайплайном и состоянием. Node только рендерит. Общаются через
subprocess: JSON на stdin, путь к файлу на stdout.

---

## 4. Пайплайн

| Этап | Что делает | Замер на 21-сек ролике |
|---|---|---|
| `probe` | ffprobe: длительность, fps, разрешение, поворот, звук | 0.06 с |
| `asr` | faster-whisper → слова с таймингами | 4.5 с |
| `clean` | LLM правит расшифровку индексами, не переписывает текст | 2.2 с |
| `direct` | LLM: что вырезать, разбивка на блоки, акценты, камера | 2.2 с |
| `broll` | vision по 4 кадрам → план вставок → Pexels | 8.9 с |
| `cut` | ffmpeg `concat` с перекодированием | 7.2 с |
| `crop` | автокроп 9:16 по трекингу лиц (YuNet) | 12.0 с |
| `render` | Remotion → mp4 | **119.6 с** |

Итого ~2.5 минуты на 20-секундный ролик. Рендер — 6× реального времени, самая
дорогая позиция. Машина — ноутбук с Ryzen 5 5500U (12 потоков, 15 Вт),
рендер ограничен одним потоком, ffmpeg — четырьмя (иначе ноутбук выключался
по просадке питания).

Стоимость LLM на ролик: **$0.0034**.

---

## 5. Контракт EDL

```python
class Word:        idx, text, start, end, prob
class Cut:         src_in, src_out, reason: "hook"|"keep"|"drop_silence"|"drop_filler"
class Emphasis:    "none" | "accent" | "shout"
class CaptionWord: word_idx, emphasis, emoji
class CaptionBlock: words: list[CaptionWord], position: "top"|"center"|"bottom"|"auto",
                    layout: "stack"|"ragged"|"single_line"
class BrollAnim:   "pop_in" | "slide_up" | "fullscreen_to_inset"
class Broll:       word_idx, duration (0.8..3.0), query, prompt, anim, anchor, asset
                   anchor: "top_left"|"top_right"|"bottom_left"|"bottom_right"|"center"|"fullscreen"
class CameraMove:  word_idx, type: "zoom_punch"|"shake", scale (1.0..1.25)

class EDL:         version, source_duration, words, cuts, captions, broll, camera, preset
```

**Кто что решает.**

- Python считает: куски по паузам, границы блоков, все таймкоды, потолок
  картинок, перенос строк, геометрию кропа.
- LLM решает: какие слова выбросить, где разорвать блок, какое слово акцентное,
  какая картинка и куда, где движение камеры.

**Семантический валидатор** (13 проверок, отдельно от pydantic): индексы слов
совпадают с позицией; куски не пересекаются и отсортированы; каждое оставленное
слово попало ровно в один блок; порядок возрастает; **не больше одного акцента
на блок**; картинок не больше `ceil(длительность/8)` и не ближе 8 секунд друг к
другу; `word_idx` камеры существует.

Провал валидации → один ретрай с текстом ошибки в промпте → **детерминированный
фолбэк** (ролик соберётся всегда, просто без акцентов и картинок).

`layout` в `CaptionBlock` объявлен, но рендерером игнорируется.

---

## 6. Как устроен рендер — весь код, который отвечает за картинку

### 6.1 Композиция

Порядок слоёв снизу вверх: **видео → b-roll → субтитры**.

```tsx
export const Reel: React.FC<RenderPlan> = (plan) => {
  const { preset } = plan;
  const camera = useCameraScale(plan.camera ?? []);

  return (
    <AbsoluteFill style={{ backgroundColor: preset.background }}>
      <AbsoluteFill
        style={{
          transform: `translate(${camera.dx}px, ${camera.dy}px) scale(${camera.scale})`,
          transformOrigin: "center center",
        }}
      >
        {plan.video ? (
          <OffthreadVideo
            src={staticFile(plan.video)}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : null}
      </AbsoluteFill>

      <Broll items={plan.broll ?? []} preset={preset} />
      <Captions blocks={plan.blocks ?? []} preset={preset} />
    </AbsoluteFill>
  );
};
```

**Важно:** трансформ камеры обёрнут только вокруг видео. `Broll` и `Captions`
стоят снаружи и не двигаются вместе с камерой.

### 6.2 Камера

```tsx
const useCameraScale = (moves: PlanCamera[]) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;

  const active = moves.find((m) => t >= m.start && t <= m.end);
  if (!active) return { scale: 1, dx: 0, dy: 0 };

  const framesIn = frame - active.start * fps;
  const progress = spring({
    frame: framesIn,
    fps,
    config: { damping: 13, stiffness: 220, mass: 0.7 },
    durationInFrames: Math.max(3, Math.round((active.end - active.start) * fps)),
  });

  if (active.type === "shake") {
    const amp = interpolate(progress, [0, 1], [1, 0], { extrapolateRight: "clamp" });
    return {
      scale: 1 + (active.scale - 1) * 0.35,
      dx: Math.sin(frame * 1.9) * 14 * amp,
      dy: Math.cos(frame * 2.3) * 10 * amp,
    };
  }

  // zoom_punch: резкий наезд и мягкий возврат
  const scale = interpolate(progress, [0, 0.45, 1], [1, active.scale, 1], {
    extrapolateRight: "clamp",
  });
  return { scale, dx: 0, dy: 0 };
};
```

Окна движения строит Python: `start = word_window[0] - 0.05`,
`end = word_window[0] + span`, где `span = 0.45` для `zoom_punch` и `0.3` для
`shake`. Сортировки по времени, слияния перекрытий и минимального зазора между
движениями **нет**.

### 6.3 Субтитры

Три состояния слова: `ghost` (ещё не прозвучало) → `active` → `past`.
Ghost-слова уже нарисованы на своих местах, поэтому контейнер не прыгает при
появлении.

```tsx
const stateOf = (word, t) => {
  if (t < word.start) return "ghost";
  if (t <= word.end) return "active";
  return "past";
};

const Word = ({ word, preset }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const state = stateOf(word, t);
  const a = preset.animation;

  const framesSinceStart = frame - word.start * fps;
  const pop = framesSinceStart >= 0
    ? spring({
        frame: framesSinceStart,
        fps,
        config: { damping: a.spring_damping, stiffness: a.spring_stiffness, mass: a.spring_mass },
        durationInFrames: Math.max(2, Math.round((a.pop_ms / 1000) * fps)),
      })
    : 0;

  const base = emphasisScale(word, preset);   // accent_scale | shout_scale | 1
  const scale = state === "ghost"
    ? 1
    : interpolate(pop, [0, 1], [1, base], { extrapolateRight: "clamp" }) *
      interpolate(pop, [0, 0.55, 1], [1, a.pop_scale, 1], { extrapolateRight: "clamp" });

  const rotate = word.emphasis === "shout" && state !== "ghost"
    ? interpolate(pop, [0, 1], [0, a.shout_rotate], { extrapolateRight: "clamp" })
    : 0;

  return (
    <span style={{
      display: "inline-block",
      color, opacity,
      transform: `scale(${scale}) rotate(${rotate}deg)`,
      transformOrigin: "center bottom",
      WebkitTextStroke: `${Math.round(preset.font.size * 0.02)}px ${preset.colors.outline}`,
      paintOrder: "stroke fill",
      textShadow: `0 ${Math.round(preset.font.size * 0.05)}px ${Math.round(preset.font.size * 0.09)}px ...`,
    }}>
      {word.text}
      {word.emoji ? <span style={{ WebkitTextStroke: "0" }}> {word.emoji}</span> : null}
    </span>
  );
};
```

Блок — flex-контейнер, растёт под текст сам:

```tsx
<div style={{
  display: "inline-flex", flexDirection: "column",
  alignItems: "center",
  gap: Math.round(preset.font.size * (preset.caption.line_height - 1)),
  padding: `${p.padding_y}px ${p.padding_x}px`,
  borderRadius: p.radius,
  background: rgba(p.color, p.opacity),
  backdropFilter: `blur(${p.blur}px) saturate(1.4)`,
  border: `${p.border_width}px solid ${rgba(p.border_color, p.border_opacity)}`,
  opacity: presence,
  transform: `translateY(${(1 - presence) * preset.font.size * 0.28}px) scale(${0.965 + presence * 0.035})`,
  maxWidth: "88%",
}}>
  {block.lines.map((line, li) => (
    <div key={li} style={{
      display: "flex",
      flexWrap: "nowrap",
      gap: Math.round(preset.font.size * 0.24),
      alignItems: "baseline",
      justifyContent: "center",
    }}>
      {line.map((word) => <Word key={word.word_idx} word={word} preset={preset} />)}
    </div>
  ))}
</div>
```

`presence` — появление и уход блока: `interpolate` по времени за `block_in_ms`
и `block_out_ms`, берётся минимум из двух.

Перенос строк считает Python **жадно по количеству символов** (не по пикселям),
потолок 3 строки, длинное слово никогда не режется:

```python
def wrap_words(words, max_chars, max_lines=3):
    lines = [[]]
    length = 0
    for w in words:
        wl = len(w.text)                       # эмодзи не учитывается
        if lines[-1] and length + 1 + wl > max_chars and len(lines) < max_lines:
            lines.append([]); length = 0
        lines[-1].append(w)
        length += wl + (1 if length else 0)
    return [line for line in lines if line]
```

Минимальная безопасная зона: `margin = max(preset.caption.margin, height * 0.07)`.

### 6.4 Картинки (b-roll)

```tsx
const INSET_RATIO = 0.44;   // доля ширины кадра под картинку в углу
const MARGIN = 0.05;        // отступ от края в долях ширины

const anchorBox = (anchor, W, H) => {
  const w = W * INSET_RATIO;
  const h = w;              // карточка КВАДРАТНАЯ, содержимое кроется object-fit
  const m = W * MARGIN;
  switch (anchor) {
    case "top_left":     return { left: m,         top: H * 0.16, width: w, height: h };
    case "top_right":    return { left: W - w - m, top: H * 0.16, width: w, height: h };
    case "bottom_left":  return { left: m,         top: H * 0.58, width: w, height: h };
    case "bottom_right": return { left: W - w - m, top: H * 0.58, width: w, height: h };
    case "center":       return { left: (W-w)/2,   top: (H-h)/2,  width: w, height: h };
    case "fullscreen":   return { left: 0, top: 0, width: W, height: H };
  }
};

const Item = ({ item, preset }) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const t = frame / fps;
  if (!item.asset) return null;
  if (t < item.start || t > item.end) return null;     // жёсткий обрыв

  const framesIn = frame - item.start * fps;
  const enter = spring({
    frame: framesIn, fps,
    config: { damping: 14, stiffness: 200, mass: 0.8 },
    durationInFrames: Math.max(3, Math.round(0.34 * fps)),
  });
  const exit = interpolate(t, [item.end - 0.22, item.end], [1, 0], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });

  const target = anchorBox(item.anchor, width, height);
  let box = target, opacity = enter * exit, scale = 1, translateY = 0;

  if (item.anim === "pop_in") {
    scale = interpolate(enter, [0, 1], [0.72, 1], { extrapolateRight: "clamp" });
  } else if (item.anim === "slide_up") {
    translateY = interpolate(enter, [0, 1], [height * 0.09, 0], { extrapolateRight: "clamp" });
  } else if (item.anim === "fullscreen_to_inset") {
    const hold = Math.min(0.9, (item.end - item.start) * 0.4);
    const k = easeOut(interpolate(t, [item.start + hold, item.start + hold + 0.42], [0, 1], {
      extrapolateLeft: "clamp", extrapolateRight: "clamp",
    }));
    const full = anchorBox("fullscreen", width, height);
    const dest = anchorBox(item.anchor === "fullscreen" ? "top_right" : item.anchor, width, height);
    box = {
      left:   full.left   + (dest.left   - full.left)   * k,
      top:    full.top    + (dest.top    - full.top)    * k,
      width:  full.width  + (dest.width  - full.width)  * k,
      height: full.height + (dest.height - full.height) * k,
    };
    opacity = Math.min(1, enter * 2) * exit;
  }

  const inset = box.width < width * 0.9;

  return (
    <div style={{
      position: "absolute",
      left: box.left, top: box.top, width: box.width, height: box.height,
      opacity,
      transform: `translateY(${translateY}px) scale(${scale})`,
      borderRadius: inset ? preset.panel.radius : 0,
      overflow: "hidden",
      boxShadow: inset ? `0 ${height * 0.012}px ${height * 0.03}px rgba(0,0,0,0.45)` : undefined,
      border: inset ? `${preset.panel.border_width * 2}px solid rgba(255,255,255,0.16)` : undefined,
    }}>
      <Img src={staticFile(item.asset)} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
    </div>
  );
};
```

Источник картинок — Pexels, версия `portrait` = **800×1200 (2:3)**.

Python пытается развести субтитр с картинкой по вертикали:

```python
BOTTOM_ANCHORS = {"bottom_left", "bottom_right"}
TOP_ANCHORS = {"top_left", "top_right"}

def occupied(start, end):
    """Какие половины кадра заняты картинкой в этом промежутке."""
    bottom = top = False
    for item in broll:
        if item.end <= start or item.start >= end:
            continue
        if item.anchor in BOTTOM_ANCHORS or item.anchor in ("center", "fullscreen"):
            bottom = True
        if item.anchor in TOP_ANCHORS or item.anchor in ("center", "fullscreen"):
            top = True
    return bottom, top

def _resolve_position(requested, preset_position, *, collides_bottom, collides_top):
    if requested in ("top", "center", "bottom"):
        return requested
    base = preset_position
    if base == "bottom" and collides_bottom and not collides_top:
        return "top"
    if base == "top" and collides_top and not collides_bottom:
        return "bottom"
    return base
```

---

## 7. Пресеты (стиль как данные)

`scaled(height)` пересчитывает размеры под высоту кадра, но **только шесть
полей**: `font.size`, `caption.margin`, `panel.padding_x/y`, `panel.radius`,
`panel.blur`. `border_width` и все анимации не масштабируются.

### glass.json (основной, им отрендерены все ролики)

```json
{
  "name": "glass", "title": "Glass", "background": "#000000",
  "font": {
    "family": "Montserrat ExtraBold", "file": "Montserrat-ExtraBold.ttf",
    "weight": 800, "size": 88, "uppercase": false, "letter_spacing": -0.015
  },
  "colors": {
    "active": "#FFFFFF", "past": "#FFFFFF", "ghost": "#FFFFFF",
    "past_opacity": 0.7, "ghost_opacity": 0.3,
    "accent": "#FFD54A", "shout": "#FF6A3D",
    "outline": "#05070A", "shadow": "#000000"
  },
  "panel": {
    "enabled": true, "color": "#0B0F14", "opacity": 0.32, "blur": 18,
    "radius": 30, "padding_x": 34, "padding_y": 22,
    "border_color": "#FFFFFF", "border_opacity": 0.14, "border_width": 1
  },
  "caption": {
    "position": "bottom", "margin": 380, "max_line_chars": 15,
    "line_height": 1.06, "align": "center"
  },
  "animation": {
    "pop_scale": 1.16, "pop_ms": 160,
    "spring_damping": 12.0, "spring_stiffness": 190.0, "spring_mass": 0.7,
    "block_in_ms": 180, "block_out_ms": 140,
    "accent_scale": 1.18, "shout_scale": 1.34, "shout_rotate": -4.0
  }
}
```

### bold.json (второй пресет, без панели)

```json
{
  "font": { "size": 104, "weight": 900, "uppercase": true, "letter_spacing": -0.02 },
  "colors": { "past_opacity": 0.55, "ghost_opacity": 0.22,
              "accent": "#28E07B", "shout": "#FF3B30", "outline": "#000000" },
  "panel": { "enabled": false },
  "caption": { "position": "center", "margin": 0, "max_line_chars": 12, "line_height": 1.0 },
  "animation": { "pop_scale": 1.24, "pop_ms": 130,
                 "spring_damping": 10.0, "spring_stiffness": 240.0, "spring_mass": 0.6,
                 "block_in_ms": 120, "block_out_ms": 100,
                 "accent_scale": 1.22, "shout_scale": 1.42, "shout_rotate": -6.0 }
}
```

---

## 8. Найденные дефекты (с числами)

Всё ниже проверено расчётом с настоящими `spring`/`interpolate` из
установленного Remotion и реальными метриками шрифта Montserrat ExtraBold для
кириллицы (строчная 0.6706 em, прописная 0.7864 em).

### 8.1 Акцентное слово наезжает на соседнее

**Механизм.** `transform: scale()` — операция стадии отрисовки. Она не меняет
layout-ширину элемента. Flex-строка резервирует место по неувеличенному слову,
`gap` фиксирован (`0.24 × font.size` = 21 px для glass) и не растёт вместе с
масштабом, а `flexWrap: "nowrap"` не даёт строке ни перенестись, ни сжаться.
При `transformOrigin: center` слово расширяется симметрично на
`(scale − 1) · W / 2` в каждую сторону и просто накрывает соседа, который стоит
на месте.

**Наезд не временный, а постоянный.** Для состояния `past` пружина уже
успокоилась (`pop ≈ 1`), второй множитель вернулся в 1, а первый остался равен
`accent_scale`. Акцентное слово **навсегда застывает увеличенным** до конца
блока.

**`pop_scale` перемножается с акцентом**, хотя задумывался как лёгкий перелёт.
Пик достигается при `pop = 0.55`: `peak = (1 + 0.55·(base − 1)) · pop_scale`.

| Пресет | emphasis | пик | в покое (`past`) |
|---|---|---|---|
| glass | none | 1.160 | 1.0 |
| glass | accent | 1.275 | **1.18** |
| glass | shout | 1.377 | **1.34** |
| bold | none | 1.240 | 1.0 |
| bold | accent | 1.390 | **1.22** |
| bold | shout | **1.526** | **1.42** |

Наезд в пикселях (glass, gap 21 px; bold, gap 25 px):

- glass, accent, слово 5 букв (W=288) → **18.6 px на пике, 5.0 px постоянно**
- glass, shout, 7 букв (W=404) → **55.1 px на пике, 47.7 px постоянно**
- bold, accent, 5 букв (W=399) → **52.7 px на пике, 18.8 px постоянно**
- bold, shout, 7 букв (W=558) → **121.9 px** (полтора символа под соседом)

В bold наезжает даже слово **без акцента**: порог безубыточности —
`2·gap/(peak−1)` = 208 px ≈ 2.6 символа, то есть почти любое слово.

**Смежные проблемы переноса строк.** Контейнер `maxWidth: 88%` = 950 px, минус
паддинги панели → 880 px полезных для glass.

| | реальная ширина символа | влезает в контейнер | стоит в пресете |
|---|---|---|---|
| glass (88 px) | 57.7 px | 15.3 симв. | `max_line_chars: 15` — запас 15 px (1.7 %) |
| bold (104 px, UPPERCASE) | 79.7 px | 11.9 симв. | `max_line_chars: 12` — **уже переполнение** |

Плюс: `wrap_words` не учитывает эмодзи вовсе (+114 px на слово), пробел в
бюджете стоит 1 символ, а рендерится как gap другой ширины; широкие глифы
(`Ш` = 1.135 em) ломают лимит на любом пресете; длинное слово не режется никогда.

### 8.2 Экран дёргается

**Главная причина — ступенька масштаба.** В ветке `shake` масштаб — константа
`1 + (scale−1)·0.35` **без огибающей**. Включается и выключается за один кадр:

```
Δscale = ±0.0525 мгновенно
края кадра: 1080·0.0525/2 = 28.3 px по X,  1920·0.0525/2 = 50.4 px по Y
```

Два жёстких разрыва на каждый shake — в 2–4 раза больше самой «тряски».

**Синус считается по номеру кадра, а не по времени.**

```
ω_x = 1.9 рад/кадр → период 3.31 кадра    ω_y = 2.3 рад/кадр → период 2.73 кадра
```

| fps | f_x | f_y | сэмплов на период |
|---|---|---|---|
| 24 | 7.26 Гц | 8.79 Гц | 3.31 / 2.73 |
| 30 | 9.07 Гц | 10.98 Гц | 3.31 / 2.73 |
| 60 | 18.14 Гц | 21.96 Гц | 3.31 / 2.73 |

Для слитного движения нужно ≥8–10 сэмплов на период. Фаза прыгает на 109°/кадр
по X и 132°/кадр по Y (за 180° направление читается инвертированным).
Настоящая тряска камеры лежит в 2–6 Гц; 9–11 Гц глаз читает не как камеру, а
как сбой декодера. **Частота линейно зависит от fps** — один ролик на 24 и на
30 fps трясётся по-разному.

Фаза привязана к абсолютному `frame`, а не к началу движения: первый кадр даёт
случайное смещение до 14 px без фейд-ина.

**Тряски фактически нет — она живёт 3 кадра.** `durationInFrames = 10` при
натуральной длительности пружины 16 кадров: 70 % окна занимает звон, а не
подъём. Замер при 30 fps: `dx` = −6.59 → +6.05 → −0.06 → 0, дальше 8 мёртвых
кадров.

**zoom_punch.** ζ = 0.524, перерегулирование 14.5 %. Замер при 30 fps, S = 1.08:
наезд длится **4 кадра из 15**, максимум на экране **1.0547 вместо заказанных
1.08** (68 % — вершина тента попадает между кадрами и не сэмплируется), затем
6 мёртвых кадров, затем **паразитный второй отскок** до 1.0030 из-за звона
пружины и одностороннего клампа.

**Движения не разводятся по времени.** У картинок есть жёсткое правило «не ближе
8 секунд», у камеры такого нет. Окно движения 0.5 с, слова в русской речи идут
через 0.3–0.4 с → соседние движения штатно перекрываются. При перекрытии
`moves.find()` берёт **первый элемент массива** (порядок ответа модели), а не
ближайший по времени: одно движение молча съедает другое, а второе
активируется уже с `framesIn ≥ durationInFrames` и рендерится как статичная
полка без всякой анимации.

Схема разрешает `maxItems: 6` **без привязки к длительности** — для 20-секундного
ролика это движение каждые 3.3 с. Python не ограничивает ни количество, ни
плотность, ни порядок.

### 8.3 Картинки криво и резко

**Появление.** ζ = 0.553, перелёт 12.4 % — и он полностью выброшен клампом.
Пружина сжата в 1.8 раза (10 кадров задано при 18 натуральных).

| | 24 fps | 30 fps |
|---|---|---|
| `durationInFrames` | 8 кадров | 10 кадров |
| натуральная длительность | 15 кадров | 18 кадров |
| первый переход через 1.0 | кадр 2.10 (88 мс) | кадр 2.75 (92 мс) |

Всё видимое движение — **3 кадра (90 мс)**, и оно **обрывается на максимуме
скорости**, без торможения. Для `slide_up` карточка идёт ~1460 px/с (шаги
52 → 81 → 40 → 0 px) и встаёт мгновенно. Для `pop_in` прозрачность
0 → 0.30 → 0.77 → 1 за три кадра. Motion blur Remotion не рисует — на трёх
кадрах это стробоскоп.

`fullscreen_to_inset` хуже всех: `min(1, enter*2)` доводит альфу до 1 на кадре
1.55 — **полноэкранная картинка 1080×1920 вспыхивает из ничего за 52 мс**.

Ещё: `framesIn = frame − item.start * fps`, а `item.start` округлён до 3 знаков
и не лежит на сетке кадров — первый отрисованный кадр попадает на
`framesIn ∈ [0,1)`, то есть карточка стартует не с нуля, а с альфы 0.09–0.45,
и момент появления гуляет от ролика к ролику.

**Исчезновение** — линейная альфа за 0.22 с, без easing, **без изменения
геометрии**. Вход пружиной 333 мс, выход линейно 220 мс: разная природа входа и
выхода сама по себе читается как брак. `easeOut` есть в проекте, но к
входу/выходу не применяется.

**Расположение.** Карточка 475×475 (10.9 % площади кадра), зазор между левой и
правой всего 21.6 px.

| anchor | полоса по Y | что под ней |
|---|---|---|
| `top_*` | 307 … 782 | лицо говорящего (обычно 350–900) |
| `bottom_*` | 1114 … 1589 | **панель субтитров** (y 1206–1540) |
| `center` | 722 … 1197 | ничего (зазор до субтитров 8.6 px) |

Полоса `bottom_*` накрывает панель субтитров **на 100 % по вертикали** при любом
числе строк и примерно на 46 % по ширине. Субтитры рисуются поверх, но панель
полупрозрачна (`opacity 0.32`) — картинка просвечивает и убивает контраст.
`top_right` закрывает примерно половину лица по ширине и почти всю по высоте.

**Защита сломана логической инверсией.** В `occupied()` якоря `center` и
`fullscreen` помечают **обе** половины занятыми → условие
`collides_bottom and not collides_top` не выполняется → уклонение отключается.
При этом `center` — единственный якорь, который с субтитрами не пересекается
вообще.

Геометрия живёт в двух несвязанных местах: `INSET_RATIO`, `MARGIN`, `0.16`,
`0.58` зашиты в TSX; `caption.margin` — в JSON пресета. Python знает только про
«половину кадра» и никогда — про реальный прямоугольник карточки.

**Обрезка.** Карточка квадратная, источник 800×1200 (2:3), `object-fit: cover`
без `object-position`:

```
scale = max(475.2/800, 475.2/1200) = 0.594
отрисовано: 475.2 × 712.8,  видно: 475.2 × 475.2
обрезается 33.3 % — по 200 px исходника сверху и снизу
```

Сток кладёт объект в верхнюю треть кадра, а `cover` по центру срезает верхние
200 px — предметы и люди приезжают обезглавленными.

Во время `fullscreen_to_inset` `width` и `height` интерполируются одним `k`, то
есть **пропорция бокса плывёт с 9:16 до 1:1 на каждом кадре**, и `cover`
пересчитывает кроп под каждую новую пропорцию — содержимое дышит прямо во время
полёта.

`duration` разрешён от 0.8 с, но `fullscreen_to_inset` физически не успевает при
длительности меньше 1.07 с — картинка прилетает в угол уже растворяясь.

**Прочее.** Нет глобального `box-sizing: border-box` (Remotion инжектит его
только в Player, но не в `renderMedia`), поэтому `border: 2px` добавляется сверх
размеров → реальный элемент 479.2 px вместо 475.2. При `fullscreen_to_inset`
скругление, рамка и тень включаются **разом за один кадр** через 0.8 кадра после
старта перелёта, когда бокс ещё почти полноэкранный. B-roll стоит вне трансформа
камеры: при совпадении `zoom_punch` фон наезжает, а карточка стоит намертво и
визуально отклеивается.

### 8.4 Общее у всех трёх

**Геометрия и тайминги живут в TSX константами, а Python о них не знает.**
Поэтому вся питоновская логика «не перекрыть лицо», «разъехаться с картинкой»
оперирует половинами кадра, а реальные прямоугольники ей неизвестны.

---

## 9. Что в промпте режиссёра

Модель получает пронумерованный транскрипт и возвращает индексы. Про интересующее:

> **`accent_word` + `accent`.** В блоке не больше одного акцентного слова, и
> далеко не в каждом блоке. Хорошая плотность — одно акцентное слово на 2–3
> блока. `accent` — смысловое ударение: цифра, название, ключевой глагол;
> `shout` — эмоциональный пик: ровно один-два на весь ролик.
>
> **`position`.** Почти всегда `auto`. Не переключай позицию чаще, чем раз в
> несколько блоков: прыгающие субтитры выглядят дёшево.
>
> **`camera`.** Резкий наезд на смысловой пик. Максимум одно на 10 секунд,
> `scale` 1.05–1.15. `shake` — только на настоящем восклицании. Пустой список —
> нормальный ответ, и чаще всего правильный.

Блоки — 3–5 слов, разрыв по смыслу и дыханию, а не по счёту.

---

## 10. Что уже сделано и работает

- Полный пайплайн от приёма файла до отдачи ролика, с прогрессом в телеграме.
- Детерминированный фолбэк: при любом отказе LLM ролик всё равно собирается.
- Перерендер из сохранённого EDL с другим пресетом — без распознавания и LLM.
- Автокроп 9:16 по трекингу лиц со сглаживанием и мёртвой зоной.
- Вырезание пауз через `concat` с перекодированием.
- Учёт токенов и стоимости каждого вызова LLM в БД.
- 227 тестов (валидатор EDL, режиссёр, LLM-клиент и провайдеры, автокроп,
  рендер-план, ASS-генератор, добыча картинок, отдача ролика).
- Не покрыто тестами: композиция этапов, воркер, хендлеры бота, `cut.py`,
  `render.py`, `preset.py`.

Известные ограничения, не связанные с картинкой: распознавание на модели `small`
(редкие слова слышит неточно), приём файла до 20 МБ (локальный Bot API не
поднят), Mini App и биллинг не сделаны.

---

## 11. О чём я хочу совет

Основное — как сделать, чтобы **выглядело дорого**. Три найденных дефекта я
починю в любом случае; интереснее понять, что отличает мой результат от
результата хорошего инструмента на уровне решений, а не багов.

Конкретные развилки, по которым мне нужно твоё мнение:

1. **Акцент.** Чинить наезд резервированием места под увеличенное слово, или
   вообще отказаться от масштабирования как способа акцента (цвет, подложка,
   вес)? Что делают инструменты, которые выглядят хорошо?
2. **Karaoke-подсветка.** Сейчас три состояния слова (ghost / active / past)
   с разной альфой. Это правильная модель или устаревшая?
3. **Камера.** Стоит ли вообще держать `shake`? Или ограничиться одним хорошо
   сделанным наездом?
4. **Картинки.** Инсет в углу — правильный формат для говорящей головы, или
   лучше полноэкранная вставка с уходом? Как их вообще подают в хороших роликах?
5. **Типографика.** Montserrat ExtraBold, кегль 88 при высоте 1920, обводка
   2 % кегля, тень. Это разумная отправная точка для русского языка?
6. **Чего нет вообще**, но должно быть.

Задавай уточняющие вопросы — я отвечу. Не нужно сразу писать код.
