"""Тесты звукового поста (Ф2).

Синтетика везде: голос — синус, музыка — другой синус, эффект генерируется тем
же кодом, что и в проде. Так проверяется именно тракт (нормализация, дакинг,
попадание эффекта в момент), а не звучание конкретной записи.
"""

from __future__ import annotations

import math
import re

import pytest

from src.core.config import settings as default_settings
from src.core.edl import EDL, Broll, BrollAnim, CaptionBlock, CaptionWord, Cut, Music, Sfx, Word
from src.core.preset import load_preset
from src.core.proc import capture, ffmpeg
from src.pipeline.audio import measure_ebur128
from src.pipeline.audio_post import (
    SFX_PROFILES,
    SFX_SWISH,
    SFX_TICK,
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
    got = [s for s in plan_sfx(edl, preset=load_preset("glass")) if s.name != SFX_TICK]
    assert [(s.at_word_idx, s.name) for s in got] == [(4, SFX_WHOOSH)]


def test_plan_sfx_marks_only_punch_in():
    """Укрупнение — событие, отъезд — нет: иначе звук комментирует каждый стык."""
    cuts = [
        Cut(src_in=0.0, src_out=2.0, reason="hook", framing="wide"),
        Cut(src_in=2.0, src_out=4.0, reason="keep", framing="tight"),  # ← punch-in
        Cut(src_in=4.0, src_out=6.2, reason="keep", framing="wide"),  # отъезд, молчим
    ]
    got = [s for s in plan_sfx(simple_edl(cuts=cuts), preset=load_preset("glass"))
           if s.name != SFX_TICK]
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
    got = [s for s in plan_sfx(edl, preset=load_preset("glass")) if s.name != SFX_TICK]
    assert [s.name for s in got] == [SFX_WHOOSH], "картинка важнее реза"


def test_plan_sfx_survives_a_reel_without_cuts():
    """Ролик без единого реза — штатный случай (`cut_pauses=False`), не ошибка."""
    got = plan_sfx(simple_edl(), preset=load_preset("glass"))
    assert [s for s in got if s.name != SFX_TICK] == []


def test_plan_sfx_ignores_cut_out_events():
    edl = simple_edl(
        cuts=[Cut(src_in=0.0, src_out=2.0, reason="hook"),
              Cut(src_in=4.0, src_out=6.2, reason="keep")],
        broll=[Broll(word_idx=6, duration=1.5, query="cat",  # слово 6 = 3.0 с, вырезано
                     anim=BrollAnim.POP_IN, anchor="center")],
    )
    assert [s for s in plan_sfx(edl, preset=load_preset("glass")) if s.name != SFX_TICK] == []


def test_swish_quieter_than_whoosh():
    edl = simple_edl(
        cuts=[Cut(src_in=0.0, src_out=2.0, reason="hook", framing="wide"),
              Cut(src_in=2.0, src_out=6.2, reason="keep", framing="tight")],
    )
    swish = [s for s in plan_sfx(edl, preset=load_preset("glass")) if s.name == SFX_SWISH][0]
    assert swish.gain_db < load_preset("glass").audio.sfx_gain_db


def test_duck_ratio_is_clamped():
    """Только границы: саму глубину меряем на собранной дорожке, а не формулой.

    Проверять `duck_ratio` пересказом её же формулы бессмысленно — такой тест
    пройдёт и на сломанном расчёте. Настоящая проверка ниже, на реальном миксе:
    `test_music_ducks_exactly_as_deep_as_the_preset_asks`.
    """
    assert duck_ratio(0.0) == 1.0  # 1:1 — компрессор, который ничего не делает
    assert duck_ratio(-100.0) <= 20.0  # ffmpeg дальше 20 не умеет
    assert duck_ratio(-12.0, target_lufs=-70.0) >= 1.0  # цель ниже порога — не делим на ноль


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


def test_track_covers_video_when_audio_stream_is_short(tmp_path):
    """Дорожка не имеет права кончиться раньше картинки.

    После `concat` звуковой поток регулярно на кадр-другой короче видео, а
    бывает и на секунды (телефон дописал видео после остановки записи). Раньше
    длину задавал только `-t`, который умеет резать и не умеет добавлять, и
    ролик получал тишину в хвосте — без единой ошибки в логе.
    """
    path = tmp_path / "short.mp4"
    ffmpeg(
        ["-f", "lavfi", "-i", "color=c=black:s=64x64:r=25:d=6",
         "-f", "lavfi", "-i", f"sine=frequency={VOICE_HZ}:duration=4",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "96k", str(path)],
        timeout=120,
    )
    track = build_audio(simple_edl(), path, tmp_path / "a.wav", cfg=cfg_with(tmp_path))
    assert track is not None
    code, out, _ = capture(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(track.path)])
    got = float(out.strip())
    assert got >= 6.0 - 0.05, f"дорожка {got:.2f} с при видео 6.00 с"
    # AudioTrack.duration обязан быть замером, а не пересказом просьбы
    assert track.duration == pytest.approx(got, abs=0.05)


def test_broken_music_file_costs_only_the_music(tmp_path, voice_video):
    """Битый файл в assets/music не имеет права уносить с собой весь этап.

    Раньше он валил filter_complex, исключение ловил job.py, и ролик уезжал в
    рендер вообще без звукового поста: без нормализации и без эффектов.
    """
    cfg = cfg_with(tmp_path)
    cfg.music_dir.mkdir(parents=True, exist_ok=True)
    (cfg.music_dir / "broken.mp3").write_bytes(b"\x00not an mp3\xff" * 200)
    preset = load_preset("glass").model_copy(deep=True)
    preset.audio.music = "broken.mp3"
    track = build_audio(simple_edl(), voice_video, tmp_path / "a.wav", preset=preset, cfg=cfg)
    assert track is not None and track.music is None
    got = measure_ebur128(track.path)
    assert got is not None and got["i"] == pytest.approx(preset.audio.target_lufs, abs=1.5)


def test_silent_input_does_not_report_infinite_loudness(tmp_path, silent_video):
    """`lufs_in` уезжает в отчёт джоба, а `-Infinity` — это не JSON."""
    import json

    track = build_audio(simple_edl(), silent_video, tmp_path / "a.wav", cfg=cfg_with(tmp_path))
    assert track is not None
    assert track.lufs_in is None or math.isfinite(track.lufs_in)
    json.loads(json.dumps({"lufs_in": track.lufs_in}))  # без allow_nan «-inf» бы упал


def test_sfx_land_on_the_output_timeline_not_the_source(tmp_path, silent_video):
    """Эффект ставится по времени РЕЗУЛЬТАТА, а не исходника.

    Отличить одно от другого можно только на монтаже с вырезанным куском:
    без резов оба времени совпадают, и ошибка не видна.
    """
    ws = [Word(idx=i, text=f"w{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(12)]
    kept = [0, 1, 2, 3, 8, 9, 10, 11]
    edl = EDL(
        source_duration=6.2,
        words=ws,
        cuts=[Cut(src_in=0.0, src_out=2.0, reason="hook", framing="wide"),
              Cut(src_in=2.0, src_out=4.0, reason="drop_silence"),
              Cut(src_in=4.0, src_out=6.2, reason="keep", framing="tight")],
        captions=[CaptionBlock(words=[CaptionWord(word_idx=i) for i in kept[j:j + 4]])
                  for j in (0, 4)],
        broll=[Broll(word_idx=10, duration=1.5, query="cat",
                     anim=BrollAnim.POP_IN, anchor="center")],
    )
    src_time = ws[10].start  # 5.0 c в исходнике
    out_time = edl.timeline().word_window(ws[10])[0]  # 3.0 c в результате
    assert abs(src_time - out_time) > 1.0, "тест бессмыслен, если времена совпали"

    track = build_audio(edl, silent_video, tmp_path / "a.wav", cfg=cfg_with(tmp_path))
    assert track is not None
    loud = [x.name for x in edl.sfx if x.name != SFX_TICK]
    assert sorted(loud) == sorted([SFX_WHOOSH, SFX_SWISH]), "картинка и укрупнение"
    hit = loudest_moment(track.path)
    assert hit == pytest.approx(out_time, abs=0.12), (
        f"эффект ударил на {hit:.2f} c; выходное время {out_time:.2f}, исходное {src_time:.2f}"
    )


def test_broken_sfx_file_costs_only_the_effect(tmp_path, silent_video):
    """Битый файл в assets/sfx не имеет права уносить с собой весь этап.

    Зеркало `test_broken_music_file_costs_only_the_music`. Сценарий не
    выдуманный: `assets/sfx/README.md` предлагает класть файлы туда руками, а
    `synth_sfx` пишет туда же — убитый на середине воркер оставлял в общей папке
    обрезанный `whoosh.flac`, и КАЖДЫЙ следующий ролик терял loudnorm, музыку и
    эффекты разом (job.py глотал исключение как «звуковой пост пропущен»).
    """
    cfg = cfg_with(tmp_path)
    cfg.sfx_dir.mkdir(parents=True, exist_ok=True)
    broken = cfg.sfx_dir / f"{SFX_WHOOSH}.flac"
    broken.write_bytes(b"not a flac at all")
    edl = simple_edl(broll=[Broll(word_idx=6, duration=1.5, query="cat",
                                  anim=BrollAnim.POP_IN, anchor="center")])
    track = build_audio(edl, silent_video, tmp_path / "a.wav", cfg=cfg)
    assert track is not None, "битый эффект унёс весь звуковой пост"
    # штатный эффект пересобирается поверх мусора, а не пропадает вместе с этапом
    assert [x.name for x in edl.sfx if x.name != SFX_TICK] == [SFX_WHOOSH]
    hit = loudest_moment(track.path)
    assert hit == pytest.approx(edl.words[6].start, abs=0.12)


def test_unreadable_custom_sfx_costs_only_its_own_mark(tmp_path, silent_video):
    """Чужой эффект пересобрать нечем — значит пропускаем метку, а не этап."""
    cfg = cfg_with(tmp_path)
    cfg.sfx_dir.mkdir(parents=True, exist_ok=True)
    (cfg.sfx_dir / "custom.wav").write_bytes(b"not a wav at all")
    edl = simple_edl()
    edl.sfx = [Sfx(at_word_idx=6, name="custom", gain_db=-6.0)]
    track = build_audio(edl, silent_video, tmp_path / "a.wav", cfg=cfg)
    assert track is not None and track.sfx == 0


def test_synth_sfx_leaves_no_half_written_asset(tmp_path):
    """В общую папку эффект попадает целиком или никак: временный файл + rename."""
    from src.pipeline.audio_post import synth_sfx as synth

    dst = tmp_path / "sfx" / f"{SFX_WHOOSH}.flac"
    synth(dst, kind=SFX_WHOOSH)
    leftovers = [p.name for p in dst.parent.iterdir() if p.name != dst.name]
    assert leftovers == [], f"в папке эффектов остался мусор: {leftovers}"


def test_video_without_audio_is_skipped(tmp_path):
    path = tmp_path / "mute.mp4"
    ffmpeg(["-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=2",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
           timeout=60)
    assert build_audio(simple_edl(), path, tmp_path / "a.wav", cfg=cfg_with(tmp_path)) is None


def ducking_depth(tmp_path, voice_video, music_file, *, duck_db, target_lufs=-16.0):
    """Собрать дорожку и померить, НА СКОЛЬКО музыка присела под голосом."""
    preset = load_preset("glass").model_copy(deep=True)
    preset.audio.music = str(music_file)
    preset.audio.music_gain_db = -10.0  # громче обычного: иначе меряем шум кодека
    preset.audio.duck_db = duck_db
    preset.audio.target_lufs = target_lufs
    track = build_audio(simple_edl(), voice_video, tmp_path / "a.wav",
                        preset=preset, cfg=cfg_with(tmp_path))
    assert track is not None and track.music == music_file.name
    under_voice = band_rms(track.path, MUSIC_HZ, 1.0, 1.5)
    in_pause = band_rms(track.path, MUSIC_HZ, 3.8, 1.5)
    return in_pause - under_voice


@pytest.mark.parametrize("duck_db", [-6.0, -12.0, -18.0])
def test_music_ducks_exactly_as_deep_as_the_preset_asks(
    tmp_path, voice_video, music_file, duck_db
):
    """DoD: музыка приседает под голос ровно на `preset.audio.duck_db`.

    Допуск здесь узкий не из вредности: на синтетике тракт попадает в
    запрошенное с точностью до полудецибела. Прежние «4..25 дБ» приняли бы и
    вдвое слабее, и вдвое сильнее нужного — то есть сломанный расчёт ratio.
    """
    drop = ducking_depth(tmp_path, voice_video, music_file, duck_db=duck_db)
    assert abs(drop - abs(duck_db)) < 1.5, (
        f"запрошено {abs(duck_db):.0f} дБ, получено {drop:.1f} дБ"
    )


def test_ducking_depth_survives_a_change_of_target_loudness(
    tmp_path, voice_video, music_file
):
    """`duck_db` и `target_lufs` — из одного пресета, и не спорят друг с другом.

    Раньше запас над порогом сайдчейна был зашит числом (24 дБ = −16 минус −40),
    и на пресете с целью −20 LUFS запрошенные −12 дБ давали фактические −10.
    """
    drop = ducking_depth(tmp_path, voice_video, music_file, duck_db=-12.0, target_lufs=-20.0)
    assert abs(drop - 12.0) < 1.5, f"при цели −20 LUFS дакинг вышел {drop:.1f} дБ вместо 12"


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
    assert track is not None
    assert [x.name for x in edl.sfx if x.name != SFX_TICK] == [SFX_WHOOSH]
    hit = loudest_moment(track.path)
    assert hit == pytest.approx(expected, abs=0.12), (
        f"whoosh ударил на {hit:.2f} c вместо {expected:.2f} c"
    )


def edl_with_a_pause_after_the_punch_in() -> EDL:
    """Монтаж, где рез и первое слово после него разнесены на полсекунды.

    Ровно этим он и отличается от `simple_edl`: там слова стоят через 0.5 с и рез
    всегда совпадает со стартом слова, поэтому разница между «якорь на резе» и
    «якорь на слове» тождественно равна нулю и ошибку не видно.
    """
    starts = [0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ws = [Word(idx=i, text=f"w{i}", start=s, end=s + 0.4) for i, s in enumerate(starts)]
    return EDL(
        source_duration=6.0,
        words=ws,
        cuts=[Cut(src_in=0.0, src_out=2.0, reason="hook", framing="wide"),
              Cut(src_in=2.0, src_out=6.0, reason="keep", framing="tight")],  # ← punch-in
        captions=[CaptionBlock(words=[CaptionWord(word_idx=i) for i in g])
                  for g in ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])],
    )


def test_swish_lands_on_the_cut_not_on_the_first_word(tmp_path, silent_video):
    """Укрупнение слышно ровно там, где оно видно.

    Крупность рисует `crop.framing_spans` по накопленным длительностям кусков,
    то есть меняется на ГРАНИЦЕ РЕЗА. Эффект, привязанный к первому слову внутри
    куска, стабильно опаздывал на всю паузу между резом и речью.
    """
    edl = edl_with_a_pause_after_the_punch_in()
    tl = edl.timeline()
    cut_at = tl.map(2.0)  # ← сюда же смотрит framing_spans
    word_at = tl.word_window(edl.words[4])[0]
    assert word_at - cut_at > 0.4, "тест бессмыслен, если рез и слово совпали"

    track = build_audio(edl, silent_video, tmp_path / "a.wav", cfg=cfg_with(tmp_path))
    assert track is not None
    assert [x.name for x in edl.sfx if x.name != SFX_TICK] == [SFX_SWISH]
    hit = loudest_moment(track.path)
    assert hit == pytest.approx(cut_at, abs=0.05), (
        f"swish ударил на {hit:.2f} c; рез на {cut_at:.2f}, первое слово на {word_at:.2f}"
    )


def test_sfx_stay_on_the_source_timeline_when_cuts_are_not_applied(tmp_path, silent_video):
    """`cut_pauses=False`: видео скопировано, а не нарезано — звук живёт в исходном.

    В этом режиме джоб отдаёт рендеру `apply_cuts=False`, картинка и субтитры
    остаются на исходном таймлайне, и звук обязан остаться там же. Считать
    эффекты по нарезанному таймлайну значило бы сдвинуть их все на суммарную
    длину вырезанного.
    """
    ws = [Word(idx=i, text=f"w{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(12)]
    kept = [0, 1, 2, 3, 8, 9, 10, 11]
    edl = EDL(
        source_duration=6.2,
        words=ws,
        cuts=[Cut(src_in=0.0, src_out=2.0, reason="hook", framing="wide"),
              Cut(src_in=2.0, src_out=4.0, reason="drop_silence"),
              Cut(src_in=4.0, src_out=6.2, reason="keep", framing="tight")],
        captions=[CaptionBlock(words=[CaptionWord(word_idx=i) for i in kept[j:j + 4]])
                  for j in (0, 4)],
        broll=[Broll(word_idx=10, duration=1.5, query="cat",
                     anim=BrollAnim.POP_IN, anchor="center")],
    )
    track = build_audio(edl, silent_video, tmp_path / "a.wav",
                        apply_cuts=False, cfg=cfg_with(tmp_path))
    assert track is not None
    # резов в картинке нет — подчёркивать нечего, остаётся одна картинка
    assert [s.name for s in edl.sfx if s.name != SFX_TICK] == [SFX_WHOOSH]
    hit = loudest_moment(track.path)
    assert hit == pytest.approx(ws[10].start, abs=0.12), (
        f"эффект ударил на {hit:.2f} c вместо исходных {ws[10].start:.2f}"
    )


def test_whoosh_on_the_very_first_word_still_lands_on_the_picture(tmp_path, silent_video):
    """Картинка на первом слове — штатный хук, а не редкость.

    Эффект стартует раньше события на свой пик; у нулевого события старт
    упирается в ноль, и без подрезки головы удар уезжал ВПЕРЁД на весь `peak_at`
    — то есть звучал уже после картинки.
    """
    edl = simple_edl(broll=[Broll(word_idx=0, duration=1.5, query="cat",
                                  anim=BrollAnim.POP_IN, anchor="center")])
    track = build_audio(edl, silent_video, tmp_path / "a.wav", cfg=cfg_with(tmp_path))
    assert track is not None
    assert [x.name for x in edl.sfx if x.name != SFX_TICK] == [SFX_WHOOSH]
    hit = loudest_moment(track.path)
    assert hit == pytest.approx(0.0, abs=0.12), (
        f"whoosh ударил на {hit:.2f} c, а картинка входит на 0.00 "
        f"(пик профиля {SFX_PROFILES[SFX_WHOOSH].peak_at:.2f} с)"
    )


def test_unresolved_music_leaves_no_trace_in_the_edl(tmp_path, voice_video):
    """Два артефакта одного джоба не имеют права противоречить друг другу.

    Отчёт джоба пишет `music=None`, а `edl.json` утверждал, что ролик сведён с
    подложкой, — потому что незарезолвленный трек оставался в EDL.
    """
    cfg = cfg_with(tmp_path)
    cfg.music_dir.mkdir(parents=True, exist_ok=True)
    edl = simple_edl()
    edl.music = Music(track="ghost.mp3")
    track = build_audio(edl, voice_video, tmp_path / "a.wav", cfg=cfg)
    assert track is not None and track.music is None
    assert edl.music is None


def test_music_reaches_the_edl_as_an_asset_name(tmp_path, voice_video, music_file):
    """`EDL.music.track` — имя файла в assets/music, а не путь этой машины.

    Абсолютный путь в `edl.json` — машинно-локальный артефакт: тот же EDL на
    другой машине молча остался бы без музыки, а путь ещё и утекал в отчёт.
    """
    preset = load_preset("glass").model_copy(deep=True)
    preset.audio.music = str(music_file)  # абсолютный путь в пресете — так можно
    edl = simple_edl()
    track = build_audio(edl, voice_video, tmp_path / "a.wav",
                        preset=preset, cfg=cfg_with(tmp_path))
    assert track is not None and track.music == music_file.name
    assert edl.music is not None and edl.music.track == music_file.name


def test_sfx_land_in_the_edl(tmp_path, voice_video):
    """Расстановка обязана попасть в EDL: иначе её не видно в артефактах джоба."""
    edl = simple_edl(broll=[Broll(word_idx=4, duration=1.5, query="cat",
                                  anim=BrollAnim.POP_IN, anchor="center")])
    assert edl.sfx == []
    build_audio(edl, voice_video, tmp_path / "a.wav", cfg=cfg_with(tmp_path))
    assert [s.name for s in edl.sfx if s.name != SFX_TICK] == [SFX_WHOOSH]
    assert any(s.name == SFX_TICK for s in edl.sfx), "щелчки субтитров тоже обязаны попасть в лист"


# ------------------------------------------------------------------ щелчок субтитра


def test_every_caption_block_gets_a_tick():
    """Звук на появлении субтитра — третий вид акцента.

    Якорь — ПЕРВОЕ НЕВЫРЕЗАННОЕ слово блока, тот же примитив, которым считается
    начало блока в `render_plan.build_plan`. Брать `block.words[0]` наивно
    нельзя: блок с резом внутри разъехался бы с картинкой.
    """
    edl = simple_edl()
    ticks = [s for s in plan_sfx(edl, preset=load_preset("glass")) if s.name == SFX_TICK]
    assert [s.at_word_idx for s in ticks] == [b.words[0].word_idx for b in edl.captions]


def test_the_tick_is_the_quietest_of_the_three():
    """Щелчок звучит на каждом блоке, а картинка — пару раз за ролик. Громкость
    редкого события на частом превращается в стрекот."""
    glass = load_preset("glass")
    tick = [s for s in plan_sfx(simple_edl(), preset=glass) if s.name == SFX_TICK][0]
    assert tick.gain_db == glass.audio.caption_gain_db
    assert tick.gain_db < glass.audio.sfx_gain_db - 6, "щелчок не тише акцентов"


def test_the_tick_yields_to_a_picture():
    """Приоритет: картинка > рез > щелчок. Блок, который и так подчёркнут
    вставкой, второго звука не получает."""
    edl = simple_edl(broll=[Broll(word_idx=3, duration=1.5, query="cat",
                                  anim=BrollAnim.POP_IN, anchor="center")])
    got = plan_sfx(edl, preset=load_preset("glass"))
    assert [s.name for s in got if s.at_word_idx == 3] == [SFX_WHOOSH]


def test_ticks_keep_their_own_distance():
    """Порог частоты у щелчка СВОЙ и больше общего.

    Общий `SFX_MIN_GAP_S = 1.0` про другое — «две метки слились в кашу». Блок
    субтитра живёт 2-3 секунды, и порог в секунду не проредил бы ничего.
    """
    from src.pipeline.audio_post import CAPTION_SFX_MIN_GAP_S

    ws = [Word(idx=i, text=f"w{i}", start=i * 0.3, end=i * 0.3 + 0.25) for i in range(12)]
    edl = EDL(
        source_duration=ws[-1].end + 0.2,
        words=ws,
        cuts=[],
        captions=[CaptionBlock(words=[CaptionWord(word_idx=i) for i in g])
                  for g in [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]],
    )
    ticks = [s for s in plan_sfx(edl, preset=load_preset("glass")) if s.name == SFX_TICK]
    times = [ws[s.at_word_idx].start for s in ticks]
    assert len(times) < len(edl.captions), "частые блоки не проредились"
    assert all(b - a >= CAPTION_SFX_MIN_GAP_S for a, b in zip(times, times[1:]))


def test_a_block_cut_out_of_the_reel_gets_no_tick():
    edl = simple_edl(cuts=[Cut(src_in=0.0, src_out=1.4, reason="hook"),
                           Cut(src_in=3.0, src_out=6.2, reason="keep")])
    ticks = [s for s in plan_sfx(edl, preset=load_preset("glass")) if s.name == SFX_TICK]
    kept = {w.idx for c in edl.kept_cuts for w in edl.words if c.src_in <= w.start < c.src_out}
    assert all(s.at_word_idx in kept for s in ticks), "щелчок на вырезанном слове"


def test_the_tick_lands_before_the_first_word(tmp_path, monkeypatch):
    """Блок начинается НА первом слове, то есть на атаке слога — худшее место
    для разборчивости. Щелчок сдвигается в тишину перед словом."""
    from src.pipeline.audio_post import CAPTION_SFX_LEAD_S, SFX_PROFILES, sfx_marks

    edl = simple_edl()
    ticks = [s for s in plan_sfx(edl, preset=load_preset("glass")) if s.name == SFX_TICK]
    marks = [m for m in sfx_marks(edl, ticks) if m.path.stem == SFX_TICK]
    assert marks, "метки щелчка не собрались"
    # Второй блок: первый упёрся бы в ноль и дал бы head_trim.
    second = marks[1]
    want = edl.words[3].start - CAPTION_SFX_LEAD_S - SFX_PROFILES[SFX_TICK].peak_at
    assert second.at == pytest.approx(want, abs=0.002)
