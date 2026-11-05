# Режим «Два видео» — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ролик из двух видео пользователя: сверху немое верхнее, снизу говорящий; когда верхнее кончается, нижнее мгновенно раскрывается на весь экран.

**Architecture:** разделённый кадр собирается ДО рендера, в «подготовленное» видео 1080×1920 (ffmpeg `vstack` верхнего окна и нижней половины, затем склейка с полнокадровым нижним). Момент раскрытия хранится в листе (`EDL.split_until`), поэтому перерендер даёт то же видео. `build_plan` ставит блоки разделённой части на шов по центру. Remotion не меняется.

**Tech Stack:** Python 3.12, ffmpeg, aiohttp, SQLAlchemy + alembic, Remotion (без правок), vanilla JS Mini App.

**Spec:** `docs/superpowers/specs/2026-11-05-split-screen-design.md`

## Global Constraints

- Звук ролика — только нижнее видео; у верхнего дорожка выбрасывается (`-an`).
- B-roll в режиме выключен целиком, независимо от `options.broll`.
- Длина ролика — длина речи снизу (`edl.out_duration`); хвост длинного верхнего обрезается.
- Верхнее видео идёт непрерывно в выходном времени: кадр ролика в момент t — кадр верхнего источника в t.
- Раскрытие — мгновенная склейка на границе блоков субтитров, не позже конца верхнего видео.
- Окно половины — 1080×960 (половина `cfg.render_width × cfg.render_height`); верхнее — по центру с заполнением, нижнее — `autocrop` по лицу.
- Пока кадр поделён, блоки стоят `center`, в регистре плаката, без привязки к голове и без ритма.
- Режим только в Mini App. Один ролик — одно списание (`cfg.charges`).
- Обычный режим (`split_until is None`) ведёт себя байт в байт как до работы.
- Коммитить только свои файлы/ханки: дерево общее с другими сессиями (память `shared-tree-commit-hygiene`).

## Review Focus

1. Верхнее видео с другим fps, разрешением или поворотом (вертикальное с телефона с rotate) — подготовленное видео обязано выйти ровно 1080×1920 и с fps нижнего. Тест в задаче 2.
2. Верхнее видео короче первого блока субтитров или короче секунды — разделения не будет вовсе, ролик обычный, без мелькания половин. Тест в задаче 2.
3. Верхний файл не видео (аудио, картинка, мусор) — отказ 415 до списания, обе папки приёма убраны. Тест в задаче 5.
4. Перерендер после удаления файлов по сроку хранения — понятная ошибка «исходный файл потерялся», а не падение; уборка удаляет и верхний ключ. Тест в задаче 3.
5. Чужой или уже забранный `top_upload_id` — 404, нижний приём не списан и не потерян. Тест в задаче 5.

---

### Task 1: Шов в плане рендера

**Files:**
- Modify: `src/core/edl.py` (поле `EDL.split_until`)
- Modify: `src/core/render_plan.py` (`block_windows`, правило шва в `build_plan`)
- Test: `tests/test_render_plan.py`

**Interfaces:**
- Produces: `EDL.split_until: float | None = None` — секунды выходного таймлайна, до которых кадр поделён; `None` — обычный режим.
- Produces: `block_windows(edl: EDL, *, apply_cuts: bool = True, cfg: Settings | None = None) -> list[tuple[float, float]]` — окна блоков субтитров ровно так, как их потом поставит `build_plan` (start = начало первого слова, end = конец последнего + `cfg.sub_tail`, затем обрезка перекрытий `a.end = b.start` и по длительности). `build_plan` обязан пользоваться ею же, чтобы окна не разошлись.

- [ ] **Step 1: Write the failing tests**

```python
def test_block_windows_match_the_plan():
    edl = long_edl()
    plan = build_plan(edl, "in.mp4", preset=load_preset("glass"), cfg=cfg())
    caps = [(b.start, b.end) for b in plan.blocks if not b.title_card]
    assert block_windows(edl, cfg=cfg()) == pytest.approx(caps)

def test_blocks_of_the_split_part_sit_on_the_seam():
    edl = long_edl(split_until=20.0)
    plan = build_plan(edl, "in.mp4", preset=load_preset("pulse"), cfg=cfg())
    caps = [b for b in plan.blocks if not b.title_card]
    before = [b for b in caps if b.start < 20.0]
    after = [b for b in caps if b.start >= 20.0]
    assert before and {b.position for b in before} == {"center"}
    assert not any(b.quiet or b.anchor_path for b in before)
    assert "center" not in {b.position for b in after}

def test_a_plan_without_split_is_unchanged():
    edl = long_edl()
    assert build_plan(edl, "in.mp4", preset=load_preset("glass"), cfg=cfg()).to_dict() == \
        build_plan(edl.model_copy(update={"split_until": None}), "in.mp4", preset=load_preset("glass"), cfg=cfg()).to_dict()

def test_the_title_card_stays_over_the_top_half():
    # шов по центру не занимает полосу титра: заголовок не должен пропасть
    edl = long_edl(split_until=20.0, title="20 тысяч за стол")
    plan = build_plan(edl, "in.mp4", preset=load_preset("pulse"), cfg=cfg())
    assert any(b.title_card for b in plan.blocks)

def test_split_until_survives_the_json_round_trip():
    edl = long_edl(split_until=12.5)
    assert EDL.model_validate_json(edl.model_dump_json()).split_until == 12.5
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest -q tests/test_render_plan.py -k "block_windows or seam or without_split or split_until"`
Expected: FAIL (`split_until` — extra field forbidden; `block_windows` not defined).

- [ ] **Step 3: Implement**

`EDL.split_until: float | None = None` (комментарий: зачем в листе — перерендер). В `build_plan`: блок, чьё окно начинается раньше `edl.split_until`, получает `places=("center",)` в регистре плаката (`quiet=False` при раскладке), `roams` для него ложно, `anchor_path` не считается; в ритм он идёт `fixed="center"`. Ритм для дома после раскрытия отсчитывается от `split_until`.

- [ ] **Step 4: Run tests** — `.venv/bin/pytest -q tests/test_render_plan.py` → PASS.

- [ ] **Step 5: Commit** — `git add src/core/edl.py src/core/render_plan.py tests/test_render_plan.py && git commit -m "План рендера знает про шов разделённого кадра"`

---

### Task 2: Сборка разделённого кадра

**Files:**
- Create: `src/pipeline/split.py`
- Test: `tests/test_split.py`

**Interfaces:**
- Consumes: `block_windows` (Task 1), `autocrop` (`src/pipeline/crop.py`), `probe` → `MediaInfo(duration, fps, width, height, has_audio)`.
- Produces: `MIN_SPLIT_SECONDS = 1.0`
- Produces: `choose_split_until(top_duration: float, out_duration: float, windows: Sequence[tuple[float, float]]) -> float | None` — `out_duration`, если верхнее не короче ролика; иначе наибольшая граница из `{0} ∪ starts ∪ ends`, не большая `top_duration`; результат меньше `MIN_SPLIT_SECONDS` → `None` (разделения нет).
- Produces: `top_window(src: Path, dst: Path, *, width: int, height: int, fps: float, duration: float) -> Path` — ffmpeg: scale с заполнением + crop по центру до width×height, `-an`, fps нижнего, обрезка до `duration`.
- Produces: `stage_split(bottom_cut: Path, top_src: Path, dst: Path, *, until: float, out_duration: float, work_dir: Path, cfg: Settings, preset, framing, track_out: Path | None, crop=autocrop) -> Path` — нижняя половина: `crop(bottom_cut, …, out_height=cfg.render_height // 2)`; если `until < out_duration`, ещё полнокадровый `crop(…, out_height=cfg.render_height, track_out=track_out)`; ffmpeg: `[top][half]vstack` на `[0, until)`, затем полный кадр на `[until, out_duration]`, склейка, звук — дорожка `bottom_cut`. Трек лица — только полнокадрового прохода.

- [ ] **Step 1: Write the failing tests** (видео синтезируются `ffmpeg -f lavfi testsrc/color`, `crop` подменён простым scale, чтобы не гонять YuNet)

```python
def test_split_lasts_the_whole_reel_when_top_is_long_enough():
    assert choose_split_until(30.0, 20.0, [(0, 5), (5, 20)]) == 20.0

def test_switch_lands_on_a_caption_boundary_before_top_ends():
    assert choose_split_until(7.3, 20.0, [(0.2, 3.0), (3.0, 6.1), (6.1, 9.0)]) == 6.1

def test_a_top_shorter_than_a_second_means_no_split():
    assert choose_split_until(0.8, 20.0, [(0.2, 3.0)]) is None
    # единственная граница до конца верхнего — 0.5 с, это меньше MIN_SPLIT_SECONDS
    assert choose_split_until(2.5, 20.0, [(0.5, 3.0)]) is None

def test_staged_video_is_full_frame_with_bottom_sound(tmp_path):
    # нижнее 1280x720 30fps со звуком, верхнее 720x1280 25fps с rotate=90 и своим звуком
    out = stage_split(bottom, top, tmp_path / "staged.mp4", until=2.0, out_duration=4.0, ..., crop=fake_crop)
    info = probe(out)
    assert (info.width, info.height) == (1080, 1920)
    assert info.duration == pytest.approx(4.0, abs=0.1)
    assert info.fps == pytest.approx(30, abs=0.5)
    assert info.has_audio  # звук нижнего; громкость сравнить с дорожкой нижнего, а не верхнего

def test_top_half_shows_top_before_the_switch_and_bottom_after(tmp_path):
    # верхнее — сплошной красный, нижнее — сплошной синий
    # кадр t=1.0: пиксель (540, 400) красный, (540, 1500) синий; кадр t=3.0: (540, 400) синий
```

- [ ] **Step 2: Run** — `.venv/bin/pytest -q tests/test_split.py` → FAIL (module not found).
- [ ] **Step 3: Implement** `src/pipeline/split.py` по Interfaces. ffmpeg-вызовы — через `src/core/proc.run` с таймаутом, как в `crop.py`.
- [ ] **Step 4: Run** — `.venv/bin/pytest -q tests/test_split.py` → PASS.
- [ ] **Step 5: Commit** — `git add src/pipeline/split.py tests/test_split.py && git commit -m "Сборка кадра из двух видео: половины и раскрытие на весь экран"`

---

### Task 3: Второй исходник у задачи

**Files:**
- Modify: `src/db/models.py` (`Job.top_source_key`)
- Create: `alembic/versions/<rev>_job_top_source.py`
- Modify: `src/db/repo.py` (`mark_files_purged` обнуляет и `top_source_key`)
- Modify: `src/worker/tasks.py` (`fetch_top_source`, уборка удаляет верхний ключ)
- Test: `tests/test_retention.py`, `tests/test_money.py`

**Interfaces:**
- Produces: `Job.top_source_key: Mapped[str | None]` (String(512), nullable).
- Produces: `fetch_top_source(job, dst: Path) -> Path | None` в `src/worker/tasks.py` — `None`, если у задачи нет верхнего; `JobError("Исходный файл потерялся. Пришлите видео ещё раз.", "source_lost")`, если ключ был, а файла нет (то есть ключ обнулён уборкой при живом `edl.split_until`).
- Produces: имя объекта в хранилище — `storage.key_for(SOURCES, str(job.id), "top.mp4")`.

- [ ] **Step 1: Failing tests:** `test_purge_deletes_the_top_source_too` (в `purge_expired_files` удаляются оба ключа и результат, `mark_files_purged` обнуляет `top_source_key`); на настоящем Postgres — `create_job` + `update_job(top_source_key=…)` читается обратно.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** столбец, миграцию (`alembic revision`, down_revision — текущая голова), уборку и `fetch_top_source`.
- [ ] **Step 4: Run** `.venv/bin/pytest -q tests/test_retention.py tests/test_money.py` и `alembic upgrade head` на локальной базе → PASS.
- [ ] **Step 5: Commit.**

---

### Task 4: Пайплайн и перерендер

**Files:**
- Modify: `src/pipeline/job.py` (`process_video(..., top_src: Path | None = None)`)
- Modify: `src/worker/tasks.py` (`process_job` качает верхний, `rerender_job` собирает кадр так же)
- Test: `tests/test_worker_loop.py` (тот же `harness` с подменёнными тяжёлыми стадиями)

**Interfaces:**
- Consumes: `choose_split_until`, `stage_split`, `top_window` (Task 2); `block_windows` (Task 1); `fetch_top_source` (Task 3).
- Produces: в `process_video` при `top_src`: `probe(top_src)` (без требований к звуку и длине), b-roll пропущен, после нарезки и звука `edl.split_until = choose_split_until(top.duration, edl.out_duration, block_windows(edl, apply_cuts=cut_pauses, cfg=cfg))`, при `split_until is not None` стадия crop — `stage_split(...)` вместо `autocrop`. Прогресс стадии — тот же ключ `"crop"`.
- Produces: `rerender_job` при `job.top_source_key` и `edl.split_until` — `stage_split` с сохранённым `edl.split_until` (не пересчитывать).

- [ ] **Step 1: Failing tests:** `test_split_job_skips_broll_and_stages_two_videos` (с `top_src` вызван `stage_split`, `plan_and_fetch` не вызван, `edl.split_until` записан в результат); `test_rerender_of_a_split_job_reuses_the_saved_switch` (`stage_split` получил `until` из листа); `test_a_job_without_top_is_built_as_before` (вызван `autocrop`, `stage_split` не вызывался).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Тяжёлые вызовы — в `asyncio.to_thread`, как соседние стадии (`tests/test_worker_loop.py` это проверяет).
- [ ] **Step 4: Run** `.venv/bin/pytest -q tests/test_worker_loop.py tests/test_stuck_jobs.py` → PASS.
- [ ] **Step 5: Commit.**

---

### Task 5: Приём двух файлов в API

**Files:**
- Modify: `src/webapp/server.py` (`upload_finish`, `MAX_OPEN_UPLOADS = 4`)
- Modify: `src/core/i18n.py` (`api.top_unreadable` ru/en)
- Test: `tests/test_webapp_server.py`

**Interfaces:**
- Consumes: `Job.top_source_key` (Task 3).
- Produces: `POST /api/uploads/{id}/finish` принимает необязательное `top_upload_id` в теле. С ним: верхний приём проверяется тем же `_own_upload`, забирается `_claim_upload` после нижнего; `probe` верхнего, ошибка → 415 `api.top_unreadable`; после `create_job` оба файла в хранилище (`source.mp4`, `top.mp4`), `update_job(top_source_key=…)`. Все отказы убирают ОБЕ папки. В чат-пути ничего не меняется.

- [ ] **Step 1: Failing tests:** `test_a_foreign_top_upload_is_not_found` (404, нижняя папка не забрана и цела); `test_an_unreadable_top_is_refused_before_charging` (415, `create_job` не вызван, обе папки удалены); `test_a_user_can_open_two_uploads_for_one_reel` (два приёма плюс один брошенный не упираются в квоту).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `.venv/bin/pytest -q tests/test_webapp_server.py` → PASS.
- [ ] **Step 5: Commit** (только свои ханки `server.py`; чужие правки шрифтов в дереве не трогать).

---

### Task 6: Интерфейс «Одно видео / Два видео»

**Files:**
- Modify: `src/webapp/static/index.html`, `src/webapp/static/app.js`, `src/webapp/static/app.css`

**Interfaces:**
- Consumes: `top_upload_id` в finish (Task 5).
- Produces: `state.layout: "single" | "split"`, `state.topFile: File | null` (нижнее — прежний `state.file`); `uploadOne(file, onBytes) -> Promise<string>` (upload_id) — вынести из `uploadAndStart` без изменения протокола; `uploadAndStart` грузит нижнее, затем верхнее, прогресс — сумма байт обоих.

Копирайт (ru / en):
- переключатель: «Одно видео» / «Два видео»; «One video» / «Two videos»
- верхнее поле: «Сверху — что показываем» + «без звука»; «Top — what you show» + «muted»
- нижнее поле: «Снизу — вы в кадре» + «звук и субтитры отсюда»; «Bottom — you on camera» + «sound and captions come from here»
- подсказка кнопки без второго файла: «добавьте оба видео»; «add both videos»

- [ ] **Step 1:** разметка: переключатель над `#dropzone`; в режиме split два поля в рамке 9:16 друг над другом, у каждого превью (`<video>` первый кадр, как сейчас) и крестик; строка «Медиа-карточки» скрыта.
- [ ] **Step 2:** логика: кнопка активна только при обоих файлах; `send-in-chat` в split скрыт; выбор режима помнится в `localStorage` (обёрнуто в try/catch).
- [ ] **Step 3: Verify:** `node --check src/webapp/static/app.js`; снимок экрана headless-стендом (память `miniapp-headless-preview`) на 390 px в обоих режимах, светлая и тёмная тема; загрузка двух файлов на локальном сервере доходит до finish с `top_upload_id`.
- [ ] **Step 4: Commit** (только свои ханки: Mini App параллельно правит другая сессия).

---

### Task 7: Проверка вживую и выкладка

- [ ] **Step 1:** рендер настоящего ролика: нижнее — `samples/phase-0-source.mp4`, верхнее — любой клип короче нижнего; кадры до и после раскрытия глазами: половины 960/960, субтитр на шве, титр наверху, звук только нижнего, без стоп-кадра у верхней половины.
- [ ] **Step 2:** `tools/check.sh` в чистом worktree только с этой работой → «всё зелёное».
- [ ] **Step 3:** push; если CI не довёз — выкладка по файлам (память `tgbot-video-deploy-pushes-to-prod`), `alembic upgrade` через `reelbot-migrate`, перезапуск только при пустой очереди, запись в `.deployed`.
- [ ] **Step 4:** обычный ролик после выкладки собирается как раньше (одна задача на сервере или локальный прогон `tests/run_pipeline.py`).
