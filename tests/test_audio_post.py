"""Тесты звукового поста (Ф2).

Синтетика везде: голос — синус, музыка — другой синус, эффект генерируется тем
же кодом, что и в проде. Так проверяется именно тракт (нормализация, дакинг,
попадание эффекта в момент), а не звучание конкретной записи.
"""

from __future__ import annotations

import re

import pytest

from src.core.config import settings as default_settings
from src.core.edl import EDL, Broll, BrollAnim, CaptionBlock, CaptionWord, Cut, Word
from src.core.preset import load_preset
from src.core.proc import capture, ffmpeg
from src.pipeline.audio import measure_ebur128
from src.pipeline.audio_post import (
    SFX_PROFILES,
    SFX_SWISH,
    SFX_WHOOSH,
    build_audio,
    duck_ratio,
    plan_sfx,
    synth_sfx,
)

VOICE_HZ = 220
MUSIC_HZ = 3000  # четыре октавы от голоса: полосовой фильтр разведёт их начисто


# ------------------------------------------------------------------ вспомогательное


def band_rms(path, freq: float, start: float, duration: float) -> float:
    """RMS в узкой полосе вокруг частоты, на куске дорожки.

    Полоса нужна, чтобы отделить музыку от голоса: оба синусы, но на разных
    частотах, и в сумме их иначе не разделить. Фильтр каскадом из трёх звеньев:
    одного двухполюсника мало, его склон пропускает голос на −46 дБ, и на этом
    полу тонет весь эффект дакинга.
    """
    band = f"bandpass=f={freq:g}:width_type=h:w={freq * 0.15:g}"
    code, out, err = capture(
        ["ffmpeg", "-hide_banner", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
         "-i", str(path), "-af", f"{band},{band},{band},volumedetect", "-f", "null", "-"]
    )
    assert code == 0
    m = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", out + err)
    assert m, "volumedetect ничего не напечатал"
    return float(m.group(1))


def rms_envelope(path) -> list[tuple[float, float]]:
    """Огибающая дорожки: (время, RMS в дБ) по аудиокадрам, один проход ffmpeg.

    Сотня запусков `volumedetect` со сдвигом дала бы то же самое, но в двадцать
    раз медленнее и грубее по времени.
    """
    envelope = (
        "astats=metadata=1:reset=1,"
        "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-"
    )
    code, out, err = capture(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-af", envelope, "-f", "null", "-"]
    )
    assert code == 0
    env: list[tuple[float, float]] = []
    at: float | None = None
    for line in (out + err).splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            at = float(m.group(1))
            continue
        m = re.search(r"RMS_level=(\S+)", line)
        if m and at is not None:
            try:
                env.append((at, float(m.group(1))))
            except ValueError:  # «-inf» на цифровой тишине
                env.append((at, -999.0))
    return env


def loudest_moment(path) -> float:
    """Момент самого громкого кадра — так ищется удар эффекта."""
    env = rms_envelope(path)
    assert env, "astats не отдал ни одного кадра"
    return max(env, key=lambda p: p[1])[0]


def make_video(path, audio_graph: str, *, duration: float = 6.0):
    """Чёрный квадрат с заданной звуковой дорожкой — минимальный вход этапа."""
    ffmpeg(
        ["-f", "lavfi", "-i", f"color=c=black:s=64x64:r=10:d={duration:g}",
         "-f", "lavfi", "-i", audio_graph,
         "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", str(path)],
        timeout=120,
    )
    return path


def cfg_with(tmp_path, **kw):
    """Настройки с папками ассетов во временной директории (репозиторий не трогаем)."""
    base = dict(sfx_dir=tmp_path / "sfx", music_dir=tmp_path / "music")
    base.update(kw)
    return default_settings.model_copy(update=base)


def simple_edl(*, n_words: int = 12, cuts=None, broll=None) -> EDL:
    ws = [Word(idx=i, text=f"w{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(n_words)]
    return EDL(
        source_duration=ws[-1].end + 0.2,
        words=ws,
        cuts=cuts or [],
        captions=[
            CaptionBlock(words=[CaptionWord(word_idx=i) for i in g])
            for g in [list(range(i, min(i + 3, n_words))) for i in range(0, n_words, 3)]
        ],
        broll=broll or [],
    )


# ------------------------------------------------------------------ синтез эффектов


@pytest.mark.parametrize("kind", sorted(SFX_PROFILES))
def test_synth_sfx_has_predictable_length_and_peak(tmp_path, kind):
    path = synth_sfx(tmp_path / f"{kind}.wav", kind=kind)
    code, out, err = capture(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    )
    assert code == 0
    assert float(out.strip()) == pytest.approx(SFX_PROFILES[kind].duration, abs=0.02)
    # Пик нормирован: только тогда gain_db в пресете значит одно и то же для всех.
    code, out, err = capture(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-af", "volumedetect",
         "-f", "null", "-"]
    )
    peak = float(re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", out + err).group(1))
    assert peak == pytest.approx(-3.0, abs=0.3)


def test_whoosh_sweeps_upward(tmp_path):
    """Смысл whoosh в том, что спектр едет вверх, а не в том, что это шум."""
    path = synth_sfx(tmp_path / "whoosh.wav", kind=SFX_WHOOSH)
    ffmpeg(["-i", str(path), "-af", "lowpass=f=500", str(tmp_path / "low.wav")], timeout=60)
    ffmpeg(["-i", str(path), "-af", "highpass=f=3000", str(tmp_path / "high.wav")], timeout=60)
    lo = loudest_moment(tmp_path / "low.wav")
    hi = loudest_moment(tmp_path / "high.wav")
    assert hi > lo, f"верх должен приходить позже низа, а пришёл на {lo:.2f} и {hi:.2f}"
    # удар эффекта — там, где заявлен профиль: по нему выравнивается вся расстановка
    assert loudest_moment(path) == pytest.approx(SFX_PROFILES[SFX_WHOOSH].peak_at, abs=0.12)


def test_missing_sfx_is_synthesized_on_demand(tmp_path):
    from src.pipeline.audio_post import sfx_path

    cfg = cfg_with(tmp_path)
    assert not (cfg.sfx_dir / f"{SFX_WHOOSH}.flac").exists()
    got = sfx_path(SFX_WHOOSH, cfg=cfg)
    assert got is not None and got.exists()
    # второй вызов берёт готовый файл, а не пересобирает
    assert sfx_path(SFX_WHOOSH, cfg=cfg) == got


def test_unknown_sfx_is_not_invented(tmp_path):
    from src.pipeline.audio_post import sfx_path

    assert sfx_path("nope", cfg=cfg_with(tmp_path)) is None


# ------------------------------------------------------------------ расстановка


def test_plan_sfx_marks_broll_entry():
    edl = simple_edl(broll=[Broll(word_idx=4, duration=1.5, query="cat",
                                  anim=BrollAnim.POP_IN, anchor="center")])
    got = plan_sfx(edl, preset=load_preset("glass"))
    assert [(s.at_word_idx, s.name) for s in got] == [(4, SFX_WHOOSH)]


def test_plan_sfx_marks_only_punch_in():
    """Укрупнение — событие, отъезд — нет: иначе звук комментирует каждый стык."""
    cuts = [
        Cut(src_in=0.0, src_out=2.0, reason="hook", framing="wide"),
        Cut(src_in=2.0, src_out=4.0, reason="keep", framing="tight"),  # ← punch-in
        Cut(src_in=4.0, src_out=6.2, reason="keep", framing="wide"),  # отъезд, молчим
    ]
    got = plan_sfx(simple_edl(cuts=cuts), preset=load_preset("glass"))
    assert [s.name for s in got] == [SFX_SWISH]
    assert got[0].at_word_idx == 4  # первое слово, начинающееся после реза на 2.0 с


def test_plan_sfx_keeps_events_apart():
    """Картинка и рез в одной точке — один звук, а не два наложенных."""
    cuts = [
        Cut(src_in=0.0, src_out=2.0, reason="hook", framing="wide"),
        Cut(src_in=2.0, src_out=6.2, reason="keep", framing="tight"),
    ]
    edl = simple_edl(cuts=cuts, broll=[Broll(word_idx=4, duration=1.5, query="cat",
                                             anim=BrollAnim.POP_IN, anchor="center")])
    got = plan_sfx(edl, preset=load_preset("glass"))
    assert [s.name for s in got] == [SFX_WHOOSH], "картинка важнее реза"


def test_plan_sfx_ignores_cut_out_events():
    edl = simple_edl(
        cuts=[Cut(src_in=0.0, src_out=2.0, reason="hook"),
              Cut(src_in=4.0, src_out=6.2, reason="keep")],
        broll=[Broll(word_idx=6, duration=1.5, query="cat",  # слово 6 = 3.0 с, вырезано
                     anim=BrollAnim.POP_IN, anchor="center")],
    )
    assert plan_sfx(edl, preset=load_preset("glass")) == []


def test_swish_quieter_than_whoosh():
    edl = simple_edl(
        cuts=[Cut(src_in=0.0, src_out=2.0, reason="hook", framing="wide"),
              Cut(src_in=2.0, src_out=6.2, reason="keep", framing="tight")],
    )
    swish = plan_sfx(edl, preset=load_preset("glass"))[0]
    assert swish.gain_db < load_preset("glass").audio.sfx_gain_db


@pytest.mark.parametrize("duck_db,expected", [(-6.0, 1.33), (-12.0, 2.0), (-18.0, 4.0)])
def test_duck_ratio_follows_requested_depth(duck_db, expected):
    assert duck_ratio(duck_db) == pytest.approx(expected, rel=0.02)


def test_duck_ratio_is_clamped():
    assert duck_ratio(0.0) == 1.0  # 1:1 — компрессор, который ничего не делает
    assert duck_ratio(-100.0) <= 20.0  # ffmpeg дальше 20 не умеет


# ------------------------------------------------------------------ сведение


@pytest.fixture(scope="module")
def voice_video(tmp_path_factory):
    """Голос: 3 с тона, потом 3 с тишины — есть где померить дакинг."""
    path = tmp_path_factory.mktemp("audio") / "voice.mp4"
    return make_video(path, f"sine=frequency={VOICE_HZ}:duration=3,apad=whole_dur=6")


@pytest.fixture(scope="module")
def silent_video(tmp_path_factory):
    """Ролик без единого звука: на нём слышно только то, что добавил этап."""
    path = tmp_path_factory.mktemp("audio") / "silent.mp4"
    return make_video(path, "anullsrc=r=48000:cl=stereo:d=6")


@pytest.fixture(scope="module")
def music_file(tmp_path_factory):
    """Подложка короче ролика: заодно проверяется зацикливание."""
    path = tmp_path_factory.mktemp("audio") / "loop.wav"
    ffmpeg(["-f", "lavfi", "-i", f"sine=frequency={MUSIC_HZ}:duration=2.5",
            "-c:a", "pcm_s16le", str(path)], timeout=60)
    return path


def test_loudnorm_hits_preset_target(tmp_path, voice_video):
    preset = load_preset("glass")
    track = build_audio(simple_edl(), voice_video, tmp_path / "a.wav",
                        preset=preset, cfg=cfg_with(tmp_path))
    assert track is not None
    got = measure_ebur128(track.path)
    assert got is not None
    assert got["i"] == pytest.approx(preset.audio.target_lufs, abs=1.5), got


def test_track_length_matches_video(tmp_path, voice_video):
    track = build_audio(simple_edl(), voice_video, tmp_path / "a.wav", cfg=cfg_with(tmp_path))
    assert track is not None
    code, out, _ = capture(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(track.path)])
    assert float(out.strip()) == pytest.approx(6.0, abs=0.1)


def test_video_without_audio_is_skipped(tmp_path):
    path = tmp_path / "mute.mp4"
    ffmpeg(["-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=2",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
           timeout=60)
    assert build_audio(simple_edl(), path, tmp_path / "a.wav", cfg=cfg_with(tmp_path)) is None


def test_music_ducks_under_voice(tmp_path, voice_video, music_file):
    """DoD: музыка не забивает голос. Меряем её же уровень в паузе и под речью."""
    preset = load_preset("glass").model_copy(deep=True)
    preset.audio.music = str(music_file)
    preset.audio.music_gain_db = -10.0  # громче обычного: иначе меряем шум кодека
    preset.audio.duck_db = -12.0
    track = build_audio(simple_edl(), voice_video, tmp_path / "a.wav",
                        preset=preset, cfg=cfg_with(tmp_path))
    assert track is not None and track.music == str(music_file)

    under_voice = band_rms(track.path, MUSIC_HZ, 1.0, 1.5)
    in_pause = band_rms(track.path, MUSIC_HZ, 3.8, 1.5)
    drop = in_pause - under_voice
    assert 4.0 <= drop <= 25.0, (
        f"дакинг вне разумного: под голосом {under_voice:.1f} dB, "
        f"в паузе {in_pause:.1f} dB, разница {drop:.1f} dB"
    )


def test_music_loops_to_cover_the_whole_reel(tmp_path, silent_video, music_file):
    """Трек на 2.5 с обязан покрыть 6 с ролика, а не оборваться на своей длине."""
    preset = load_preset("glass").model_copy(deep=True)
    preset.audio.music = str(music_file)
    preset.audio.music_fade_s = 0.0  # фейды тут только мешали бы мерить
    track = build_audio(simple_edl(), silent_video, tmp_path / "a.wav",
                        preset=preset, cfg=cfg_with(tmp_path))
    assert track is not None
    first = band_rms(track.path, MUSIC_HZ, 0.5, 1.5)
    second = band_rms(track.path, MUSIC_HZ, 4.0, 1.0)  # второй проход трека
    assert abs(second - first) < 2.0, (
        f"музыка не зациклилась: первый проход {first:.1f} dB, второй {second:.1f} dB"
    )


def test_music_fades_in_and_out(tmp_path, silent_video, music_file):
    preset = load_preset("glass").model_copy(deep=True)
    preset.audio.music = str(music_file)
    preset.audio.music_fade_s = 0.5
    track = build_audio(simple_edl(), silent_video, tmp_path / "a.wav",
                        preset=preset, cfg=cfg_with(tmp_path))
    assert track is not None
    middle = band_rms(track.path, MUSIC_HZ, 2.5, 0.5)
    head = band_rms(track.path, MUSIC_HZ, 0.0, 0.1)
    tail = band_rms(track.path, MUSIC_HZ, 5.9, 0.1)
    assert head < middle - 10.0, f"нет фейда на входе: {head:.1f} против {middle:.1f} dB"
    assert tail < middle - 10.0, f"нет фейда на выходе: {tail:.1f} против {middle:.1f} dB"


def test_missing_music_is_not_fatal(tmp_path, voice_video):
    preset = load_preset("glass").model_copy(deep=True)
    preset.audio.music = "no-such-track.mp3"
    track = build_audio(simple_edl(), voice_video, tmp_path / "a.wav",
                        preset=preset, cfg=cfg_with(tmp_path))
    assert track is not None and track.music is None
    got = measure_ebur128(track.path)
    assert got is not None and got["i"] == pytest.approx(preset.audio.target_lufs, abs=1.5)


def test_whoosh_lands_on_the_picture(tmp_path, silent_video):
    """DoD: на входе картинки слышен whoosh — и именно на входе, а не рядом.

    Голос здесь цифровая тишина, поэтому эффект единственный звук в миксе и его
    момент можно померить, а не расслышать.
    """
    edl = simple_edl(broll=[Broll(word_idx=6, duration=1.5, query="cat",
                                  anim=BrollAnim.POP_IN, anchor="center")])
    expected = edl.words[6].start  # без резов выходное время равно исходному
    track = build_audio(edl, silent_video, tmp_path / "a.wav", cfg=cfg_with(tmp_path))
    assert track is not None and track.sfx == 1
    hit = loudest_moment(track.path)
    assert hit == pytest.approx(expected, abs=0.12), (
        f"whoosh ударил на {hit:.2f} c вместо {expected:.2f} c"
    )


def test_sfx_land_in_the_edl(tmp_path, voice_video):
    """Расстановка обязана попасть в EDL: иначе её не видно в артефактах джоба."""
    edl = simple_edl(broll=[Broll(word_idx=4, duration=1.5, query="cat",
                                  anim=BrollAnim.POP_IN, anchor="center")])
    assert edl.sfx == []
    build_audio(edl, voice_video, tmp_path / "a.wav", cfg=cfg_with(tmp_path))
    assert [s.name for s in edl.sfx] == [SFX_WHOOSH]
