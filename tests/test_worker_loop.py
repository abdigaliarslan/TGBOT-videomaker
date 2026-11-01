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
