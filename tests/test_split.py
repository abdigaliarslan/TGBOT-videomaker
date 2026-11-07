"""Режим «Два видео»: момент раскрытия и сборка разделённого кадра.

Кадр собирается до рендера, в подготовленное видео 1080×1920: сверху окно
верхнего видео, снизу половина говорящего, а после момента раскрытия — он же на
весь экран. Видео здесь синтетические (`lavfi`), а кадрирование по лицу
подменено простым масштабом: смысл тестов в склейке, а не в YuNet.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import Settings
from src.pipeline.probe import probe
from src.pipeline.split import MIN_SPLIT_SECONDS, choose_split_until, stage_split


def cfg() -> Settings:
    return Settings(_env_file=None, render_width=1080, render_height=1920, render_fps=30)


def make_video(
    path: Path, *, color: str, size: str, fps: int, seconds: float, tone: int | None
) -> Path:
    args = ["ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s={size}:r={fps}:d={seconds}"]
    if tone:
        args += ["-f", "lavfi", "-i", f"sine=frequency={tone}:duration={seconds}:sample_rate=48000"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest"]
    if tone:
        args += ["-c:a", "aac"]
    args.append(str(path))
    subprocess.run(args, check=True)
    return path


def fake_crop(src, dst, *, out_width, out_height, track_out=None, **_):
    """Кадрирование без поиска лица: тот же размер на выходе, звук сохранён."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src),
         "-vf", f"scale={out_width}:{out_height},setsar=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(dst)],
        check=True,
    )
    return Path(dst)


def pixel(path: Path, t: float, x: int, y: int) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"кадр {t} с не прочитан"
    b, g, r = (int(v) for v in frame[y, x])
    return r, g, b


def dominant_hz(path: Path) -> float:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", "48000", "-"],
        check=True, capture_output=True,
    ).stdout
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(samples))
    return float(np.fft.rfftfreq(len(samples), 1 / 48000)[int(np.argmax(spectrum))])


# --------------------------------------------------------------- момент раскрытия


def test_split_lasts_the_whole_reel_when_top_is_long_enough():
    assert choose_split_until(30.0, 20.0, [(0, 5), (5, 20)]) == 20.0


def test_switch_lands_on_a_caption_boundary_before_top_ends():
    assert choose_split_until(7.3, 20.0, [(0.2, 3.0), (3.0, 6.1), (6.1, 9.0)]) == 6.1


def test_a_top_shorter_than_a_second_means_no_split():
    assert choose_split_until(0.8, 20.0, [(0.2, 3.0)]) is None
    # единственная граница до конца верхнего — 0.5 с, это меньше MIN_SPLIT_SECONDS
    assert MIN_SPLIT_SECONDS > 0.5
    assert choose_split_until(2.5, 20.0, [(0.5, 3.0)]) is None


def test_a_pause_between_blocks_is_a_boundary_too():
    """Между блоками тишина: склейка там не задевает ни одного субтитра."""
    assert choose_split_until(5.0, 20.0, [(0.2, 3.0), (4.0, 6.0)]) == 4.0


# --------------------------------------------------------------- сборка кадра


@pytest.fixture
def media(tmp_path):
    bottom = make_video(tmp_path / "bottom.mp4", color="blue", size="1280x720",
                        fps=30, seconds=4.0, tone=440)
    top = make_video(tmp_path / "top.mp4", color="red", size="720x1280",
                     fps=25, seconds=6.0, tone=1000)
    return bottom, top


def test_staged_video_is_full_frame_with_bottom_sound(tmp_path, media):
    """Ровно 1080×1920, fps нижнего, длина ролика и звук только нижнего — хотя
    верхнее длиннее, другой ориентации, с другим fps и громким своим тоном."""
    bottom, top = media
    out = stage_split(
        bottom, top, tmp_path / "staged.mp4", until=2.0, out_duration=4.0,
        work_dir=tmp_path, cfg=cfg(), preset=None, framing=None, track_out=None, crop=fake_crop,
    )
    info = probe(out)
    assert (info.width, info.height) == (1080, 1920)
    assert info.duration == pytest.approx(4.0, abs=0.15)
    assert info.fps == pytest.approx(30, abs=0.5)
    assert info.has_audio
    assert dominant_hz(out) == pytest.approx(440, abs=15), "в ролик попал звук верхнего видео"


def test_top_half_shows_top_before_the_switch_and_bottom_after(tmp_path, media):
    bottom, top = media
    out = stage_split(
        bottom, top, tmp_path / "staged.mp4", until=2.0, out_duration=4.0,
        work_dir=tmp_path, cfg=cfg(), preset=None, framing=None, track_out=None, crop=fake_crop,
    )
    red, blue = pixel(out, 1.0, 540, 400), pixel(out, 1.0, 540, 1500)
    assert red[0] > 200 and red[2] < 60, f"сверху до раскрытия не верхнее видео: {red}"
    assert blue[2] > 200 and blue[0] < 60, f"снизу до раскрытия не нижнее видео: {blue}"
    after = pixel(out, 3.0, 540, 400)
    assert after[2] > 200 and after[0] < 60, f"после раскрытия сверху не нижнее видео: {after}"


def test_split_for_the_whole_reel_needs_no_full_frame_pass(tmp_path, media):
    """Верхнее не короче ролика — полнокадровый проход не нужен вовсе."""
    bottom, top = media
    sizes = []

    def counting_crop(src, dst, *, out_width, out_height, **k):
        sizes.append(out_height)
        return fake_crop(src, dst, out_width=out_width, out_height=out_height, **k)

    out = stage_split(
        bottom, top, tmp_path / "staged.mp4", until=4.0, out_duration=4.0,
        work_dir=tmp_path, cfg=cfg(), preset=None, framing=None, track_out=None, crop=counting_crop,
    )
    assert sizes == [960]
    late = pixel(out, 3.5, 540, 400)
    assert late[0] > 200, f"верхнее пропало, хотя его хватает на весь ролик: {late}"


# --------------------------------------------------------------- верхний исходник


async def test_a_job_without_a_top_has_no_top_source(tmp_path):
    from types import SimpleNamespace

    from src.worker.tasks import fetch_top_source

    job = SimpleNamespace(id="j", top_source_key=None)
    assert await fetch_top_source(job, tmp_path / "top.mp4") is None


async def test_the_top_source_is_downloaded_from_storage(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from src.worker import tasks

    got = []

    class FakeStorage:
        async def download(self, key, dst):
            got.append(key)
            Path(dst).write_bytes(b"x")

    monkeypatch.setattr(tasks, "storage", FakeStorage())
    job = SimpleNamespace(id="j", top_source_key="sources/j/top.mp4")
    out = await tasks.fetch_top_source(job, tmp_path / "top.mp4")
    assert out == tmp_path / "top.mp4" and got == ["sources/j/top.mp4"]


async def test_a_purged_top_source_is_a_clear_error(tmp_path, monkeypatch):
    """После уборки по сроку перерендер получает понятный отказ, а не падение."""
    from types import SimpleNamespace

    from src.worker import tasks

    class GoneStorage:
        async def download(self, key, dst):
            raise FileNotFoundError(key)

    monkeypatch.setattr(tasks, "storage", GoneStorage())
    job = SimpleNamespace(id="j", top_source_key="sources/j/top.mp4")
    with pytest.raises(tasks.JobError) as err:
        await tasks.fetch_top_source(job, tmp_path / "top.mp4")
    assert err.value.code == "source_lost"


async def test_a_split_job_whose_top_was_purged_is_a_clear_error(tmp_path):
    """Уборка обнуляет ключ, а лист по-прежнему говорит «поделено»."""
    from types import SimpleNamespace

    from src.worker.tasks import JobError, fetch_top_source

    job = SimpleNamespace(id="j", top_source_key=None, edl={"split_until": 6.1})
    with pytest.raises(JobError) as err:
        await fetch_top_source(job, tmp_path / "top.mp4")
    assert err.value.code == "source_lost"


# --------------------------------------------------------------- что считать верхним видео


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def test_a_picture_is_not_a_top_video(tmp_path):
    """Картинка проходит ffprobe с длиной 0 — и ролик молча вышел бы без верха."""
    from src.pipeline.split import top_video_seconds

    _ff("-f", "lavfi", "-i", "color=c=red:s=640x480", "-frames:v", "1", str(tmp_path / "a.jpg"))
    with pytest.raises(ValueError):
        top_video_seconds(tmp_path / "a.jpg")


def test_audio_with_cover_art_is_not_a_top_video(tmp_path):
    """У mp3 с обложкой есть «видеодорожка» — одна картинка на весь ролик."""
    from src.pipeline.split import top_video_seconds

    _ff("-f", "lavfi", "-i", "color=c=red:s=320x320", "-frames:v", "1", str(tmp_path / "cover.png"))
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-i", str(tmp_path / "cover.png"),
        "-map", "0:a", "-map", "1:v", "-c:a", "libmp3lame", "-c:v", "png",
        "-disposition:v", "attached_pic", str(tmp_path / "song.mp3"))
    with pytest.raises(ValueError):
        top_video_seconds(tmp_path / "song.mp3")


def test_top_length_is_the_video_not_the_audio(tmp_path):
    """Звук длиннее картинки: верх меряется по картинке, иначе он замер бы,
    пока внизу ещё говорят."""
    from src.pipeline.split import top_video_seconds

    _ff("-f", "lavfi", "-i", "testsrc=s=320x240:r=25:d=3", "-f", "lavfi", "-i", "sine=duration=8",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(tmp_path / "v.mp4"))
    assert probe(tmp_path / "v.mp4").duration > 7, "пробник подменился — тест ничего не проверяет"
    assert top_video_seconds(tmp_path / "v.mp4") == pytest.approx(3.0, abs=0.1)


def test_top_length_without_a_stream_duration_comes_from_frames(tmp_path):
    """У mkv/webm длины на дорожке нет — считаем по кадрам."""
    from src.pipeline.split import top_video_seconds

    _ff("-f", "lavfi", "-i", "testsrc=s=320x240:r=25:d=4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(tmp_path / "v.mkv"))
    assert top_video_seconds(tmp_path / "v.mkv") == pytest.approx(4.0, abs=0.1)
