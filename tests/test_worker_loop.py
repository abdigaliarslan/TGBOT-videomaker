"""Прогон не имеет права занимать event loop воркера на минуты.

`process_video` — корутина, но внутри неё ffmpeg, Whisper и YuNet работают
синхронно. Пока такой этап идёт, loop стоит: не ставится пульс задачи
(`worker.tasks.heartbeat`), не доходит SIGTERM от выкладки — и systemd
добивает воркер SIGKILL'ом на середине ролика, оставляя задачу висеть
в RUNNING. Ровно та авария, которую потом разбирает сторож.

Проверка простая: тяжёлый этап обязан выполняться НЕ в том потоке, где
крутится loop. Внутри рабочего потока `asyncio.get_running_loop()` кидает
RuntimeError — это и есть признак.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
from pathlib import Path

import pytest

from src.core.config import Settings
from src.core.edl import EDL, CaptionBlock, CaptionWord, Cut, Word
from src.pipeline.probe import MediaInfo

HEAVY = [
    "transcribe",      # Whisper, минуты
    "voice_clip",      # ffmpeg, секунды
    "apply_cuts",      # ffmpeg, минуты
    "autocrop",        # YuNet + ffmpeg, минуты
    "probe",           # ffprobe по большому файлу
    "is_playable",     # он же
    "_measure_luma",   # ffmpeg, секунды
]


def on_the_loop() -> bool:
    """Правда, если вызов идёт в потоке с работающим loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def info_of() -> MediaInfo:
    return MediaInfo(
        path="—", duration=12.0, width=1080, height=1920, fps=30.0, has_audio=True,
        audio_channels=2, audio_rate=48000, rotation=0, video_codec="h264", bitrate=0,
    )


def edl_of() -> EDL:
    words = [Word(idx=i, text=f"сл{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(8)]
    return EDL(
        preset="pulse",
        source_duration=12.0,
        words=words,
        cuts=[Cut(src_in=0.0, src_out=12.0, reason="keep", framing="tight")],
        captions=[
            CaptionBlock(words=[CaptionWord(word_idx=w.idx) for w in words[:4]]),
            CaptionBlock(words=[CaptionWord(word_idx=w.idx) for w in words[4:]]),
        ],
    )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Подменяет все тяжёлые этапы двойниками, которые запоминают, где их
    вызвали. Двойник делает минимум работы — тест про поток, а не про монтаж."""
    edl = edl_of()
    seen: dict[str, bool] = {}
    src = tmp_path / "src.mp4"
    src.write_bytes(b"\0" * 1024)

    def note(name: str):
        def mark() -> None:
            seen[name] = on_the_loop()
        return mark

    def fake_transcribe(*a, **k):
        from src.pipeline.asr import ASRResult

        note("transcribe")()
        return ASRResult(words=list(edl.words), language="ru", backend="test", model="—")

    def fake_apply_cuts(e, s, d, **k):
        note("apply_cuts")()
        shutil.copy2(s, d)

    def fake_autocrop(s, d, **k):
        note("autocrop")()
        shutil.copy2(s, d)
        return d

    def fake_probe(*a, **k):
        note("probe")()
        return info_of()

    async def fake_direct(*a, **k):
        return edl

    async def fake_metadata(*a, **k):
        return None

    async def fake_clean(words, **k):
        # Подменён, чтобы тест не ходил в сеть: смысл проверки — поток, в
        # котором считается ГОЛОС для корректора, а не сам корректор.
        return list(words), 0

    async def fake_render(plan, out, **k):
        out.write_bytes(b"\0" * 64)

    monkeypatch.setattr("src.pipeline.job.transcribe", fake_transcribe)
    monkeypatch.setattr("src.pipeline.job.apply_cuts", fake_apply_cuts)
    monkeypatch.setattr("src.pipeline.job.autocrop", fake_autocrop)
    monkeypatch.setattr("src.pipeline.job.probe", fake_probe)
    monkeypatch.setattr(
        "src.pipeline.job.is_playable", lambda *a, **k: (note("is_playable")(), True)[1]
    )
    monkeypatch.setattr(
        "src.pipeline.job.measure_bg_luma", lambda *a, **k: note("_measure_luma")()
    )
    monkeypatch.setattr(
        "src.pipeline.job.voice_clip", lambda *a, **k: (note("voice_clip")(), None)[1]
    )
    monkeypatch.setattr("src.pipeline.job.direct", fake_direct)
    monkeypatch.setattr("src.pipeline.job.make_metadata", fake_metadata)
    monkeypatch.setattr("src.pipeline.job.clean_transcript", fake_clean)
    monkeypatch.setattr("src.pipeline.job.arender_remotion", fake_render)
    return src, tmp_path / "work", seen


async def test_no_heavy_stage_runs_on_the_event_loop(harness):
    """Каждый тяжёлый этап — в рабочем потоке. Иначе пульс задачи не ставится
    и воркер не слышит SIGTERM ровно те минуты, когда это важнее всего."""
    from src.llm.client import LLMClient
    from src.pipeline.job import process_video

    src, work, seen = harness
    cfg = Settings(_env_file=None, audio_post_enabled=False, asr_fix_by_audio=True)

    class ReadyLLM(LLMClient):
        @property
        def ready(self) -> bool:
            return True

    await process_video(src, work, preset="pulse", cfg=cfg, llm=ReadyLLM())

    missing = [name for name in HEAVY if name not in seen]
    assert not missing, f"этап не вызвался — тест ничего про него не проверил: {missing}"
    on_loop = [name for name, loop in seen.items() if loop]
    assert not on_loop, f"этап занимает event loop воркера: {on_loop}"


async def test_the_rerender_leaves_the_loop_alone_too(tmp_path, monkeypatch):
    """Перерендер — второй путь того же пула `cpu`, и ffmpeg в нём тот же.

    Пульса у него нет (задача остаётся DONE, сторожу тут нечего освобождать), но
    SIGTERM от выкладки он обязан слышать так же, как основной прогон."""
    from types import SimpleNamespace

    from src.worker import tasks

    edl = edl_of()
    seen: dict[str, bool] = {}

    def note(name: str, out=None):
        def fn(*a, **k):
            seen[name] = on_the_loop()
            return out
        return fn

    async def fake_get_job(job_id):
        return SimpleNamespace(id="job-1", edl=edl.model_dump(), options={"cut": True})

    async def fake_fetch(job, dst):
        Path(dst).write_bytes(b"\0" * 1024)
        return Path(dst)

    async def fake_upload(path, key, **k):
        return key

    def fake_autocrop(s, d, **k):
        seen["autocrop"] = on_the_loop()
        shutil.copy2(s, d)
        return d

    def fake_cuts(e, s, d, **k):
        seen["apply_cuts"] = on_the_loop()
        shutil.copy2(s, d)

    async def fake_render(plan, out, **k):
        Path(out).write_bytes(b"\0" * 64)

    monkeypatch.setattr(tasks, "settings", Settings(_env_file=None, work_dir=tmp_path))
    monkeypatch.setattr(tasks.repo, "get_job", fake_get_job)
    monkeypatch.setattr(tasks, "fetch_source", fake_fetch)
    monkeypatch.setattr(tasks.storage, "upload", fake_upload)
    monkeypatch.setattr("src.pipeline.probe.probe", note("probe", info_of()))
    monkeypatch.setattr("src.pipeline.cut.apply_cuts", fake_cuts)
    monkeypatch.setattr("src.pipeline.crop.autocrop", fake_autocrop)
    monkeypatch.setattr("src.pipeline.render.arender_remotion", fake_render)

    class FakeRedis:
        async def set(self, *a, **k):
            return None

    await tasks.rerender_job({"redis": FakeRedis()}, "job-1", "pulse")

    on_loop = [name for name, loop in seen.items() if loop]
    assert set(seen) >= {"probe", "apply_cuts", "autocrop"}, f"этапы не вызвались: {seen}"
    assert not on_loop, f"перерендер занимает event loop воркера: {on_loop}"


async def test_the_loop_keeps_breathing_while_a_stage_grinds(harness, monkeypatch):
    """То же самое, но с другой стороны: пока этап работает, loop успевает
    сделать свои дела. Это и есть то, чего ждут пульс и SIGTERM."""
    from src.pipeline.job import process_video

    src, work, _ = harness
    cfg = Settings(_env_file=None, audio_post_enabled=False)
    grinding = threading.Event()

    def slow_autocrop(s, d, **k):
        grinding.set()
        time.sleep(0.3)  # так же, как это делает ffmpeg: синхронно и глухо
        shutil.copy2(s, d)
        return d

    monkeypatch.setattr("src.pipeline.job.autocrop", slow_autocrop)

    beats = 0

    async def beat() -> None:
        nonlocal beats
        while True:
            await asyncio.sleep(0.02)
            beats += 1

    ticker = asyncio.create_task(beat())
    try:
        await process_video(src, work, preset="pulse", cfg=cfg)
    finally:
        ticker.cancel()

    assert grinding.is_set(), "медленный этап не вызвался"
    assert beats >= 5, f"loop стоял, пока работал этап: ударов {beats}"


# ------------------------------------------------------------ два видео


def split_doubles(harness, monkeypatch, *, top_seconds: float = 3.0):
    """Верхнее видео в обвязке: свой probe по имени файла и подменённая сборка
    кадра, которая запоминает, с чем её позвали."""
    src, work, seen = harness
    top = src.with_name("top.mp4")
    top.write_bytes(b"\0" * 1024)
    calls: dict[str, dict] = {}

    def fake_probe(path, *a, **k):
        seen["probe"] = on_the_loop()
        return info_of()  # у верхнего контейнер «длиной в ролик» — врёт, как mp4 с длинным звуком

    def fake_top_seconds(path):
        seen["top_video_seconds"] = on_the_loop()
        return top_seconds

    def fake_stage(bottom, top_src, dst, **k):
        seen["stage_split"] = on_the_loop()
        calls["stage_split"] = k
        shutil.copy2(bottom, dst)
        return Path(dst)

    async def no_broll(*a, **k):
        calls["broll"] = {}
        return 0

    monkeypatch.setattr("src.pipeline.job.probe", fake_probe)
    monkeypatch.setattr("src.pipeline.job.top_video_seconds", fake_top_seconds, raising=False)
    monkeypatch.setattr("src.pipeline.job.stage_split", fake_stage)
    monkeypatch.setattr("src.pipeline.job.broll_mod.plan_and_fetch", no_broll)
    return src, top, work, seen, calls


class ReadyLLM:
    """Модель «готова», чтобы b-roll в обычном прогоне точно позвался."""

    ready = True
    usages: list = []

    def report(self):
        return {}


async def test_split_job_skips_broll_and_stages_two_videos(harness, monkeypatch):
    from src.pipeline.job import process_video

    src, top, work, seen, calls = split_doubles(harness, monkeypatch)
    cfg = Settings(_env_file=None, audio_post_enabled=False, asr_fix_by_audio=False, broll_enabled=True)

    result = await process_video(src, work, preset="pulse", cfg=cfg, llm=ReadyLLM(), top_src=top)

    assert "broll" not in calls, "в режиме «Два видео» позвали подбор картинок"
    assert "autocrop" not in seen, "кадр собран обычным кропом, а не из двух видео"
    until = calls["stage_split"]["until"]
    # Верх меряется по видеодорожке (3 с), а не по контейнеру (12 с): иначе
    # кадр считался бы поделённым весь ролик, и верх замер бы на 3-й секунде.
    assert result.edl.split_until == until == pytest.approx(2.0)
    assert seen["stage_split"] is False, "сборка кадра занимает event loop"
    assert seen["top_video_seconds"] is False, "замер верхнего видео занимает event loop"


async def test_a_job_without_top_is_built_as_before(harness, monkeypatch):
    from src.pipeline.job import process_video

    src, top, work, seen, calls = split_doubles(harness, monkeypatch)
    cfg = Settings(_env_file=None, audio_post_enabled=False, asr_fix_by_audio=False, broll_enabled=True)

    result = await process_video(src, work, preset="pulse", cfg=cfg, llm=ReadyLLM())

    assert "stage_split" not in calls and "autocrop" in seen
    assert result.edl.split_until is None


async def test_a_top_too_short_to_split_gives_an_ordinary_reel(harness, monkeypatch):
    from src.pipeline.job import process_video

    src, top, work, seen, calls = split_doubles(harness, monkeypatch, top_seconds=0.6)
    cfg = Settings(_env_file=None, audio_post_enabled=False, asr_fix_by_audio=False)

    result = await process_video(src, work, preset="pulse", cfg=cfg, top_src=top)

    assert result.edl.split_until is None
    assert "stage_split" not in calls and "autocrop" in seen


async def test_rerender_of_a_split_job_reuses_the_saved_switch(tmp_path, monkeypatch):
    """Перерендер не пересчитывает момент раскрытия: он берёт его из листа."""
    from types import SimpleNamespace

    from src.worker import tasks

    edl = edl_of()
    edl.split_until = 2.0
    got: dict = {}

    async def fake_get_job(job_id):
        return SimpleNamespace(id="job-1", edl=edl.model_dump(), options={"cut": True},
                               top_source_key="sources/job-1/top.mp4")

    async def fake_fetch(job, dst):
        Path(dst).write_bytes(b"\0" * 1024)
        return Path(dst)

    async def fake_upload(path, key, **k):
        return key

    def fake_stage(bottom, top_src, dst, **k):
        got.update(k, top=Path(top_src).name)
        shutil.copy2(bottom, dst)
        return Path(dst)

    def fake_cuts(e, s, d, **k):
        shutil.copy2(s, d)

    async def fake_render(plan, out, **k):
        Path(out).write_bytes(b"\0" * 64)

    monkeypatch.setattr(tasks, "settings", Settings(_env_file=None, work_dir=tmp_path))
    monkeypatch.setattr(tasks.repo, "get_job", fake_get_job)
    monkeypatch.setattr(tasks, "fetch_source", fake_fetch)
    monkeypatch.setattr(tasks, "fetch_top_source", fake_fetch)
    monkeypatch.setattr(tasks.storage, "upload", fake_upload)
    monkeypatch.setattr("src.pipeline.probe.probe", lambda *a, **k: info_of())
    monkeypatch.setattr("src.pipeline.cut.apply_cuts", fake_cuts)
    monkeypatch.setattr("src.pipeline.split.stage_split", fake_stage)
    monkeypatch.setattr("src.pipeline.render.arender_remotion", fake_render)

    class FakeRedis:
        async def set(self, *a, **k):
            return None

    await tasks.rerender_job({"redis": FakeRedis()}, "job-1", "pulse")
    assert got.get("until") == 2.0 and got.get("top") == "top.mp4"


class RecordingRedis:
    """Redis-двойник перерендера: помнит, что записали и что стёрли."""

    def __init__(self) -> None:
        self.written: dict[str, dict] = {}
        self.deleted: list[str] = []

    async def set(self, key, value, **k):
        import json

        self.written[key] = json.loads(value)

    async def delete(self, key):
        self.deleted.append(key)


async def test_a_failed_rerender_tells_the_chat_and_frees_the_user(tmp_path, monkeypatch):
    """Перерендер, который не получился, обязан сказать об этом.

    Наблюдатель в чате читает только ключ прогресса, а в базе перерендер не
    живёт. Раньше ключ писался лишь на успехе: при любой поломке пользователь
    три часа смотрел на «Пересобираю…». Заодно снимается замок «один перерендер
    на пользователя» — иначе отказ запирал бы кнопки тем на час.
    """
    from types import SimpleNamespace

    from src.worker import tasks

    async def fake_get_job(job_id):
        return SimpleNamespace(
            id="job-1", user_id=7, edl=edl_of().model_dump(), options={"cut": True}
        )

    async def lost(job, dst):
        raise tasks.JobError("Исходный файл потерялся. Пришлите видео ещё раз.", "source_lost")

    monkeypatch.setattr(tasks, "settings", Settings(_env_file=None, work_dir=tmp_path))
    monkeypatch.setattr(tasks.repo, "get_job", fake_get_job)
    monkeypatch.setattr(tasks, "fetch_source", lost)
    redis = RecordingRedis()

    with pytest.raises(tasks.JobError):
        await tasks.rerender_job({"redis": redis}, "job-1", "glass")

    said = redis.written.get(tasks.progress_key("job-1:glass"))
    assert said is not None, "об отказе никто не узнал"
    assert said["stage"] == "failed" and said["error_code"] == "source_lost"
    assert tasks.rerender_lock_key(7) in redis.deleted, "замок пользователя остался висеть"


async def test_the_rerender_rebuilds_the_sound_and_measures_its_own_frame(tmp_path, monkeypatch):
    """Смена темы — это новый ролик целиком, а не только новые субтитры.

    Перерендер пропускал два шага основного прогона. Звуковой пост: ролик
    выходил без подложки и звуковых акцентов, с громкостью нарезки. Замер
    яркости фона: в листе оставалось число прошлого прогона, снятое под другим
    кропом и другой темой, и правило контраста красило текст по нему.
    """
    from types import SimpleNamespace

    from src.worker import tasks

    edl = edl_of()
    for block in edl.captions:
        block.bg_luma = 0.9  # замер прошлого прогона
    seen: dict = {}

    async def fake_get_job(job_id):
        return SimpleNamespace(
            id="job-1", user_id=7, edl=edl.model_dump(), options={"cut": True, "sfx": False}
        )

    async def fake_fetch(job, dst):
        Path(dst).write_bytes(b"\0" * 1024)
        return Path(dst)

    async def fake_upload(path, key, **k):
        return key

    def fake_copy(s, d, **k):
        shutil.copy2(s, d)
        return d

    def fake_audio(e, video, out, **k):
        seen["audio"] = {"video": Path(video).name, "on_loop": on_the_loop(), **k}
        return SimpleNamespace(path=Path(out))

    def fake_luma(e, video, **k):
        seen["luma"] = {"video": Path(video).name, "on_loop": on_the_loop()}
        for block in e.captions:
            block.bg_luma = None

    async def fake_render(plan, out, **k):
        seen["plan"] = plan
        Path(out).write_bytes(b"\0" * 64)

    monkeypatch.setattr(tasks, "settings", Settings(_env_file=None, work_dir=tmp_path))
    monkeypatch.setattr(tasks.repo, "get_job", fake_get_job)
    monkeypatch.setattr(tasks, "fetch_source", fake_fetch)
    monkeypatch.setattr(tasks.storage, "upload", fake_upload)
    monkeypatch.setattr("src.pipeline.probe.probe", lambda *a, **k: info_of())
    monkeypatch.setattr("src.pipeline.cut.apply_cuts", lambda e, s, d, **k: fake_copy(s, d))
    monkeypatch.setattr("src.pipeline.crop.autocrop", fake_copy)
    monkeypatch.setattr("src.pipeline.audio_post.build_audio", fake_audio)
    monkeypatch.setattr("src.pipeline.job.measure_bg_luma", fake_luma)
    monkeypatch.setattr("src.pipeline.render.arender_remotion", fake_render)
    redis = RecordingRedis()

    await tasks.rerender_job({"redis": redis}, "job-1", "glass")

    assert seen["audio"]["video"] == "cut.mp4", "звук считается по нарезке, как в основном прогоне"
    assert seen["audio"]["sfx"] is False, "галочка «звуки» из опций задачи потерялась"
    assert seen["audio"]["apply_cuts"] is True
    assert seen["plan"].audio and seen["plan"].audio.endswith("audio.wav")
    assert seen["luma"]["video"] == "vertical.mp4", "фон меряется по кадру ЭТОГО перерендера"
    assert not seen["audio"]["on_loop"] and not seen["luma"]["on_loop"]
    assert redis.written[tasks.progress_key("job-1:glass")]["stage"] == "done"
    assert tasks.rerender_lock_key(7) in redis.deleted


async def test_the_worker_hands_the_top_video_to_the_pipeline(tmp_path, monkeypatch):
    """Задача с верхним исходником доходит до пайплайна с ним."""
    from types import SimpleNamespace

    from src.worker import tasks

    job = SimpleNamespace(id="job-1", user_id=42, preset="pulse", options=None, source_local_path=None,
                          source_key="sources/job-1/source.mp4", top_source_key="sources/job-1/top.mp4",
                          edl=None)
    got: dict = {}

    async def fake_get_job(job_id):
        return job

    async def took(job_id, *, resume=False):
        return True

    async def noop(*a, **k):
        return None

    async def fake_fetch(j, dst):
        return Path(dst)

    async def fake_process_video(src, work, **k):
        got.update(k)
        raise tasks.JobError("стоп", "too_short")

    class FakeRedis:
        async def set(self, *a, **k):
            return None

        async def delete(self, *a, **k):
            return None

    monkeypatch.setattr(tasks, "settings", Settings(_env_file=None, work_dir=tmp_path))
    monkeypatch.setattr(tasks.repo, "get_job", fake_get_job)
    monkeypatch.setattr(tasks.repo, "mark_running", took)
    monkeypatch.setattr(tasks.repo, "update_job", noop)
    monkeypatch.setattr(tasks.repo, "mark_failed", noop)
    monkeypatch.setattr(tasks.repo, "refund_job", noop)
    monkeypatch.setattr(tasks, "fetch_source", fake_fetch)
    monkeypatch.setattr(tasks, "fetch_top_source", fake_fetch)
    monkeypatch.setattr(tasks, "process_video", fake_process_video)

    with pytest.raises(tasks.JobError):
        await tasks.process_job({"redis": FakeRedis()}, "job-1")
    assert got.get("top_src") is not None and Path(got["top_src"]).name == "top.mp4"


async def test_a_split_sheet_is_rebuilt_from_both_videos(harness, monkeypatch):
    """Ручная пересборка листа с разделением собирает тот же кадр из двух видео."""
    from src.pipeline.job import process_from_edl

    src, top, work, seen, calls = split_doubles(harness, monkeypatch)
    edl = edl_of()
    edl.split_until = 2.0
    cfg = Settings(_env_file=None, audio_post_enabled=False)

    await process_from_edl(src, work, edl, cfg=cfg, top_src=top)
    assert calls["stage_split"]["until"] == 2.0 and "autocrop" not in seen


async def test_a_split_sheet_without_the_top_video_is_refused(harness, monkeypatch):
    """Без верхнего видео субтитры встали бы на шов поверх обычного кадра."""
    from src.pipeline.job import JobError, process_from_edl

    src, top, work, seen, calls = split_doubles(harness, monkeypatch)
    edl = edl_of()
    edl.split_until = 2.0
    with pytest.raises(JobError) as err:
        await process_from_edl(src, work, edl, cfg=Settings(_env_file=None, audio_post_enabled=False))
    assert err.value.code == "source_lost"
