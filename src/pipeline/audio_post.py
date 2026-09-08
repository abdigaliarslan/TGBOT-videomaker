"""Ф2: звуковой пост между нарезкой и рендером. Отдельная дорожка, целиком ffmpeg.

До этой фазы в пайплайне было восемь этапов и ни одного про звук: ролик отдавался
с той громкостью, с какой человек его записал, без подложки и без акцентов.

Что делает этап:

1. `loudnorm` голоса на `preset.audio.target_lufs` — речь ровная по громкости;
2. подложка из `assets/music/` (`preset.audio.music`), зациклённая под длину
   ролика, с фейдами `preset.audio.music_fade_s` на входе и выходе;
3. дакинг подложки под голос через `sidechaincompress` на `preset.audio.duck_db`;
4. `whoosh` на входе картинки и `swish` на резе с укрупнением — из `assets/sfx/`.

Результат кладётся в `RenderPlan.audio`; оба рендерера (Remotion и ASS) глушат
звук видео и берут эту дорожку.

Расстановку эффектов делает Python по УЖЕ ГОТОВОМУ монтажу (§0: LLM не решает
того, что однозначно выводится из резов и картинок).

⚠ Музыки в репозитории нет и не будет: чужой трек нельзя положить в продукт, не
разобравшись с лицензией. Свой файл кладут в `assets/music/` (там же README) и
включают строкой `"music": "<имя файла>"` в секции `audio` пресета. Пока файла
нет, этап аккуратно пропускает подложку и дакинг, оставляя loudnorm и эффекты.
Эффекты, наоборот, синтезируются процедурно (розовый шум → полосы → огибающая),
поэтому с лицензией у них чисто, а в репозитории они лежат обычными ассетами.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.core.config import Settings, settings as default_settings
from src.core.edl import EDL, Music, Sfx, Word
from src.core.preset import Preset, load_preset
from src.core.proc import capture, ffmpeg
from src.pipeline.audio import loudnorm_filter, measure_loudness

log = logging.getLogger(__name__)

# Всё сводится к одному формату: иначе amix спорит о раскладке каналов, а
# loudnorm в динамическом режиме отдаёт 192 кГц, которых не ждёт ни один рендерер.
SAMPLE_RATE = 48_000
AFORMAT = f"aformat=sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts=stereo"

SFX_WHOOSH = "whoosh"
SFX_SWISH = "swish"
MUSIC_SUFFIXES = (".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac", ".aac")
SFX_SUFFIXES = (".flac", ".wav", ".mp3", ".m4a", ".ogg")
# Эффекты лежат в репозитории во flac, а не в wav: сжатие тут ни при чём —
# `*.wav` перечислен в .gitignore (туда сыплются временные дорожки джобов), и
# wav-эффект просто не доехал бы до чужой машины. Flac побитово тот же звук.
SFX_ASSET_SUFFIX = ".flac"

# Рез тише картинки: укрупнение — рядовое событие монтажа, картинка — редкое.
SWISH_TRIM_DB = 4.0
# Ближе этого две метки сливаются в кашу; whoosh важнее, swish уступает.
SFX_MIN_GAP_S = 1.0

# Дакинг. sidechaincompress не умеет «убавь ровно на N дБ»: он давит на
# over_db * (1 - 1/ratio), где over_db — насколько ключ выше порога. Порог
# −40 dBFS, голос после loudnorm живёт около −16 LUFS, то есть примерно на 24 дБ
# выше порога — отсюда и считается ratio под нужные duck_db.
DUCK_THRESHOLD_DB = -40.0
DUCK_HEADROOM_DB = 24.0
DUCK_ATTACK_MS = 20  # успеть убрать музыку к первому слогу
DUCK_RELEASE_MS = 350  # вернуть между фразами, но не дёргать на каждой паузе
# Потолок микса. Голос нормализован по TP −1.5 dBTP, музыка и эффекты
# добавляются сверху — без потолка сумма изредка вылезает за 0 dBFS и щёлкает.
LIMITER_CEILING = 0.97
LIMITER_ATTACK_MS = 1  # лукахед лимитера = задержка всей дорожки, 1 мс не слышно


# --------------------------------------------------------------------------- эффекты


@dataclass(frozen=True, slots=True)
class Band:
    """Полоса шума со своей огибающей.

    Огибающая собирается двумя `afade` (подъём, спад) вместо `volume` с
    выражением: afade считается по сэмплам, а `volume=eval=frame` ступеньками по
    аудиокадру (~21 мс), и на шуме эти ступеньки слышны как треск.
    """

    freq: float
    start: float
    rise: float
    fall: float

    @property
    def peak(self) -> float:
        return self.start + self.rise

    @property
    def end(self) -> float:
        return self.peak + self.fall


@dataclass(frozen=True, slots=True)
class SfxProfile:
    """Рецепт эффекта: полосы поднимаются по очереди снизу вверх.

    Это и есть whoosh: спектральный центр едет вверх, амплитуда идёт горбом.
    Никакой записи не нужно, лицензия чистая, звук настоящий.
    """

    bands: tuple[Band, ...]
    seeds: tuple[int, int]  # два независимых шума → стерео без искусственной ширины
    color: str = "pink"

    @property
    def duration(self) -> float:
        return max(b.end for b in self.bands)

    @property
    def peak_at(self) -> float:
        """Момент удара внутри файла: по нему эффект выравнивается на событие."""
        return max(self.bands, key=lambda b: b.freq).peak


SFX_PROFILES: dict[str, SfxProfile] = {
    SFX_WHOOSH: SfxProfile(
        bands=(
            Band(320, 0.00, 0.26, 0.30),
            Band(800, 0.04, 0.26, 0.28),
            Band(2000, 0.08, 0.26, 0.24),
            Band(4500, 0.12, 0.25, 0.20),
        ),
        seeds=(1301, 4703),
    ),
    SFX_SWISH: SfxProfile(
        bands=(
            Band(350, 0.00, 0.13, 0.16),
            Band(1100, 0.02, 0.13, 0.14),
            Band(2600, 0.05, 0.12, 0.12),
        ),
        seeds=(2609, 8117),
    ),
}


def _peak_dbfs(path: Path) -> float | None:
    """Пик дорожки в dBFS через volumedetect. None, если замерить не удалось."""
    code, out, err = capture(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-af", "volumedetect",
         "-f", "null", "-"]
    )
    if code != 0:
        return None
    m = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", out + err)
    return float(m.group(1)) if m else None


def _synth_channel(profile: SfxProfile, stream: int, out_label: str) -> list[str]:
    """Цепочка filter_complex для одного канала эффекта."""
    parts: list[str] = []
    labels = [f"b{stream}_{i}" for i in range(len(profile.bands))]
    parts.append(
        f"[{stream}:a]asplit={len(profile.bands)}"
        + "".join(f"[n{stream}_{i}]" for i in range(len(profile.bands)))
    )
    for i, band in enumerate(profile.bands):
        parts.append(
            f"[n{stream}_{i}]bandpass=f={band.freq:g}:width_type=o:w=1.6,"
            f"afade=t=in:st={band.start:g}:d={band.rise:g}:curve=qsin,"
            f"afade=t=out:st={band.peak:g}:d={band.fall:g}:curve=qsin[{labels[i]}]"
        )
    parts.append(
        "".join(f"[{x}]" for x in labels)
        + f"amix=inputs={len(labels)}:normalize=0[{out_label}]"
    )
    return parts


def synth_sfx(dst: str | Path, *, kind: str = SFX_WHOOSH, peak_dbfs: float = -3.0) -> Path:
    """Синтез эффекта: шум → полосы → огибающая → нормировка по пику.

    Два прохода, потому что уровень суммы полос заранее неизвестен, а эффект с
    предсказуемым пиком нужен: только тогда `gain_db` в пресете означает одно и
    то же для всех эффектов.
    """
    profile = SFX_PROFILES.get(kind)
    if profile is None:
        raise ValueError(f"нет профиля эффекта {kind!r}; есть: {', '.join(SFX_PROFILES)}")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    inputs: list[str] = []
    parts: list[str] = []
    for stream, seed in enumerate(profile.seeds):
        source = (
            f"anoisesrc=d={profile.duration:g}:c={profile.color}"
            f":r={SAMPLE_RATE}:a=0.8:seed={seed}"
        )
        inputs += ["-f", "lavfi", "-i", source]
        parts += _synth_channel(profile, stream, "L" if stream == 0 else "R")
    parts.append(f"[L][R]amerge=inputs=2,{AFORMAT}[out]")

    codec = "flac" if dst.suffix.lower() == ".flac" else "pcm_s16le"
    with tempfile.TemporaryDirectory(prefix="reelbot-sfx-") as tmp:
        raw = Path(tmp) / "raw.wav"
        ffmpeg([*inputs, "-filter_complex", ";".join(parts), "-map", "[out]",
                "-c:a", "pcm_s16le", str(raw)], timeout=120)
        peak = _peak_dbfs(raw)
        gain = 0.0 if peak is None else peak_dbfs - peak
        # sample_fmt задаём явно: flac по умолчанию берёт разрядность из фильтра
        # (float32) и файл выходит больше исходного wav при том же звуке.
        ffmpeg(["-i", str(raw), "-af", f"volume={gain:.2f}dB", "-c:a", codec,
                "-sample_fmt", "s16", str(dst)], timeout=120)
    log.info("эффект %s собран: %s (%.2f с)", kind, dst, profile.duration)
    return dst


def sfx_path(name: str, *, cfg: Settings | None = None) -> Path | None:
    """Путь к эффекту. Штатный эффект синтезируется, если файла ещё нет."""
    cfg = cfg or default_settings
    found = _find_asset(name, cfg.sfx_dir, SFX_SUFFIXES)
    if found is not None:
        return found
    if name not in SFX_PROFILES:
        log.warning("эффект %r не найден в %s", name, cfg.sfx_dir)
        return None
    try:
        return synth_sfx(cfg.sfx_dir / f"{name}{SFX_ASSET_SUFFIX}", kind=name)
    except Exception as e:  # noqa: BLE001 — звук не повод ронять ролик
        log.warning("не удалось синтезировать эффект %r: %s", name, e)
        return None


# --------------------------------------------------------------------------- ассеты


def _find_asset(name: str, root: Path, suffixes: tuple[str, ...]) -> Path | None:
    """Имя файла (с расширением или без) или путь → существующий файл."""
    if not name:
        return None
    candidate = Path(name)
    roots = [candidate] if candidate.is_absolute() or candidate.parent != Path(".") else []
    roots.append(root / name)
    for base in roots:
        if base.suffix and base.exists():
            return base
        for suf in suffixes:
            probe_path = base.with_suffix(suf) if base.suffix else Path(f"{base}{suf}")
            if probe_path.exists():
                return probe_path
    return None


def resolve_music(
    edl: EDL, *, preset: Preset, cfg: Settings | None = None
) -> tuple[Path, Music] | None:
    """Подложка для ролика: сначала EDL, потом пресет. None — играем без музыки."""
    cfg = cfg or default_settings
    spec = edl.music
    if spec is None and preset.audio.music:
        spec = Music(
            track=preset.audio.music,
            gain_db=preset.audio.music_gain_db,
            duck_db=preset.audio.duck_db,
        )
    if spec is None:
        return None
    path = _find_asset(spec.track, cfg.music_dir, MUSIC_SUFFIXES)
    if path is None:
        log.warning(
            "подложка %r не найдена в %s — ролик соберётся без музыки", spec.track, cfg.music_dir
        )
        return None
    return path, spec


# --------------------------------------------------------------------------- расстановка


def _first_word_in(edl: EDL, start: float, end: float) -> Word | None:
    """Первое слово внутри куска исходника. None — в куске слов нет.

    Верхняя граница обязательна: без неё в немом куске нашлось бы слово из
    следующего, и звук реза уехал бы на несколько секунд вперёд.
    """
    for w in edl.words:
        if w.start >= start - 1e-6:
            return w if w.start <= end + 1e-6 else None
    return None


def plan_sfx(edl: EDL, *, preset: Preset) -> list[Sfx]:
    """Куда ставить звуковые акценты. Чистая функция от готового монтажа.

    Акцента заслуживают два события: вход картинки и рез с укрупнением
    (wide → tight). Оба — смена того, что зритель видит; звук её только
    подтверждает. Реза «в тот же план» и конца фразы это не касается: whoosh на
    каждом стыке превращается в стрекот.
    """
    tl = edl.timeline()
    by_idx = {w.idx: w for w in edl.words}
    gain = preset.audio.sfx_gain_db
    accepted: list[tuple[float, Sfx]] = []

    for b in edl.broll:
        w = by_idx.get(b.word_idx)
        win = tl.word_window(w) if w is not None else None
        if win is None:
            continue
        accepted.append((win[0], Sfx(at_word_idx=b.word_idx, name=SFX_WHOOSH, gain_db=gain)))
    accepted.sort(key=lambda item: item[0])

    kept = sorted(edl.kept_cuts, key=lambda c: c.src_in)
    for prev, cur in zip(kept, kept[1:]):
        if not (prev.framing == "wide" and cur.framing == "tight"):
            continue  # укрупнение, а не любой рез: пулбэк звуком не подчёркиваем
        w = _first_word_in(edl, cur.src_in, cur.src_out)
        win = tl.word_window(w) if w is not None else None
        if win is None or w is None:
            continue
        if any(abs(win[0] - t) < SFX_MIN_GAP_S for t, _ in accepted):
            continue
        accepted.append(
            (win[0], Sfx(at_word_idx=w.idx, name=SFX_SWISH, gain_db=gain - SWISH_TRIM_DB))
        )
        accepted.sort(key=lambda item: item[0])

    return [s for _, s in accepted]


def sfx_marks(
    edl: EDL, sfx: list[Sfx], *, cfg: Settings | None = None
) -> list[tuple[float, Path, float]]:
    """`Sfx` → (момент на выходном таймлайне, файл, усиление).

    Эффект стартует раньше события ровно на свой пик: whoosh должен ударить в
    кадр, а не начаться на нём.
    """
    cfg = cfg or default_settings
    tl = edl.timeline()
    by_idx = {w.idx: w for w in edl.words}
    marks: list[tuple[float, Path, float]] = []
    for s in sfx:
        w = by_idx.get(s.at_word_idx)
        win = tl.word_window(w) if w is not None else None
        if win is None:
            continue
        path = sfx_path(s.name, cfg=cfg)
        if path is None:
            continue
        lead = SFX_PROFILES[s.name].peak_at if s.name in SFX_PROFILES else 0.0
        marks.append((max(0.0, win[0] - lead), path, s.gain_db))
    marks.sort(key=lambda m: m[0])
    return marks


# --------------------------------------------------------------------------- сведение


def duck_ratio(duck_db: float) -> float:
    """Коэффициент компрессии, дающий примерно `duck_db` подавления музыки.

    Именно «примерно»: компрессор давит по огибающей, поэтому на тихой фразе
    музыка приседает меньше, чем на громкой. Это не баг, а то, ради чего берут
    компрессор, а не выключатель: дакинг фиксированной глубины слышно как
    дыхание фона на каждой паузе между словами.
    """
    want = min(abs(duck_db), DUCK_HEADROOM_DB - 2.0)
    return max(1.0, min(20.0, 1.0 / (1.0 - want / DUCK_HEADROOM_DB)))


@dataclass(slots=True)
class AudioTrack:
    path: Path
    duration: float
    music: str | None = None
    sfx: int = 0
    lufs_in: float | None = None
    target_lufs: float = -16.0


def build_audio(
    edl: EDL,
    video: str | Path,
    out: str | Path,
    *,
    preset: Preset | None = None,
    duration: float | None = None,
    cfg: Settings | None = None,
) -> AudioTrack | None:
    """Нарезанное видео + EDL → готовая звуковая дорожка. None, если звука нет.

    Вход — именно `cut.mp4`: дорожка обязана лежать на выходном таймлайне, том
    же, по которому построен `RenderPlan`. Кроп длительность не меняет, поэтому
    считать звук до него можно и нужно (кроп его всё равно не трогает).
    """
    cfg = cfg or default_settings
    preset = preset or load_preset(edl.preset)
    video, out = Path(video), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from src.pipeline.probe import probe  # локально: probe тянет ffprobe, а он нужен не всем

    info = probe(video)
    if not info.has_audio:
        log.warning("в %s нет звуковой дорожки — звуковой пост пропущен", video.name)
        return None
    total = duration or info.duration
    if total <= 0:
        return None

    if not edl.sfx:
        edl.sfx = plan_sfx(edl, preset=preset)
    marks = sfx_marks(edl, edl.sfx, cfg=cfg)
    music = resolve_music(edl, preset=preset, cfg=cfg)
    if music is not None and edl.music is None:
        edl.music = music[1]  # чтобы edl.json показывал, с чем ролик сведён

    target = preset.audio.target_lufs
    measured = measure_loudness(video, target_lufs=target)
    inputs: list[str] = ["-i", str(video)]
    parts: list[str] = [f"[0:a]{loudnorm_filter(measured, target_lufs=target)},{AFORMAT}[voice]"]
    mix: list[str] = ["[voice]"]
    stream = 1

    if music is not None:
        path, spec = music
        fade = max(0.0, min(preset.audio.music_fade_s, total / 2.0))
        # -stream_loop вместо aloop: демуксер повторяет файл сам, без буфера в
        # памяти на всю длину трека.
        inputs += ["-stream_loop", "-1", "-i", str(path)]
        chain = [AFORMAT, f"atrim=0:{total:.3f}", "asetpts=N/SR/TB", f"volume={spec.gain_db:g}dB"]
        if fade > 0:
            chain.append(f"afade=t=in:st=0:d={fade:g}")
            chain.append(f"afade=t=out:st={max(0.0, total - fade):.3f}:d={fade:g}")
        parts.append(f"[{stream}:a]" + ",".join(chain) + "[music]")
        parts.append("[voice]asplit=2[voice_mix][voice_key]")
        parts.append(
            f"[music][voice_key]sidechaincompress"
            f"=threshold={10 ** (DUCK_THRESHOLD_DB / 20):.4f}"
            f":ratio={duck_ratio(spec.duck_db):.2f}"
            f":attack={DUCK_ATTACK_MS}:release={DUCK_RELEASE_MS}:level_sc=1[music_duck]"
        )
        mix = ["[voice_mix]", "[music_duck]"]
        stream += 1

    for i, (at, path, gain_db) in enumerate(marks):
        inputs += ["-i", str(path)]
        delay_ms = round(at * 1000)
        parts.append(
            f"[{stream}:a]{AFORMAT},volume={gain_db:g}dB,adelay={delay_ms}:all=1[sfx{i}]"
        )
        mix.append(f"[sfx{i}]")
        stream += 1

    tail = mix[0]
    if len(mix) > 1:
        # normalize=0: amix иначе делит на число входов, и голос проседает от
        # самого факта появления музыки. duration=first — длина по голосу.
        parts.append("".join(mix) + f"amix=inputs={len(mix)}:normalize=0:duration=first[mixed]")
        tail = "[mixed]"
    # level=0 обязателен: с включённым автоуровнем alimiter тянет дорожку обратно
    # к 0 dBFS и стирает всё, что только что сделал loudnorm.
    parts.append(
        f"{tail}alimiter=limit={LIMITER_CEILING}:attack={LIMITER_ATTACK_MS}:level=0[out]"
    )

    ffmpeg(
        [*inputs, "-filter_complex", ";".join(parts), "-map", "[out]",
         "-t", f"{total:.3f}", "-c:a", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", str(out)],
        timeout=3600,
    )
    track = AudioTrack(
        path=out,
        duration=total,
        music=music[1].track if music else None,
        sfx=len(marks),
        lufs_in=(measured or {}).get("input_i"),
        target_lufs=target,
    )
    log.info(
        "звук сведён: %s (%.2f с), вход %.1f LUFS → цель %.1f LUFS, музыка=%s, эффектов=%d",
        out.name, total, track.lufs_in if track.lufs_in is not None else float("nan"),
        target, track.music or "нет", track.sfx,
    )
    return track


if __name__ == "__main__":  # python -m src.pipeline.audio_post cut.mp4 edl.json audio.wav
    import json
    import sys

    from src.core.logging import setup_logging

    setup_logging(default_settings.log_level)
    if len(sys.argv) < 4:
        raise SystemExit("usage: python -m src.pipeline.audio_post <video> <edl.json> <out.wav>")
    edl_obj = EDL(**json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
    print(build_audio(edl_obj, sys.argv[1], sys.argv[3]))
