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
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.core.config import Settings, settings as default_settings
from src.core.edl import EDL, Music, Sfx, Timeline, Word
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
# −40 dBFS, а голос после loudnorm живёт ровно на `preset.audio.target_lufs` —
# значит запас над порогом считается из пресета, а не берётся константой. Раньше
# тут стояли зашитые 24 дБ (то есть молчаливое «target_lufs всегда −16»), и на
# пресете с целью −20 запрошенные −12 дБ давали фактические −10.
DUCK_THRESHOLD_DB = -40.0
# Ниже этого запаса компрессор уже нечем давить: ratio улетает в потолок.
DUCK_MIN_HEADROOM_DB = 6.0
DUCK_ATTACK_MS = 20  # успеть убрать музыку к первому слогу
DUCK_RELEASE_MS = 350  # вернуть между фразами, но не дёргать на каждой паузе
# Потолок микса. Голос нормализован по TP −1.5 dBTP, музыка и эффекты
# добавляются сверху — без потолка сумма изредка вылезает за 0 dBFS и щёлкает.
# Потолок тот же −1.5 dB, а не «чуть ниже нуля»: прежние 0.97 (−0.26 dB) сводили
# на нет тот самый запас, ради которого loudnorm держит TP, и ролик уезжал в
# соцсеть с пиком выше −1 dBTP — там его дожмёт уже чужой кодек.
LIMITER_CEILING = 0.8414  # 10^(-1.5/20)
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
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-vn", "-af", "volumedetect",
         "-f", "null", "-"]
    )
    if code != 0:
        return None
    m = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", out + err)
    return float(m.group(1)) if m else None


def _audio_duration(path: str | Path) -> float | None:
    """Длительность звука файла в секундах. None — потока нет или он не читается.

    Спрашиваем и поток, и контейнер: у ogg/opus длительность потока сплошь и
    рядом «N/A», и проверка «читается ли файл» по одному только потоку забраковала
    бы совершенно живую подложку.
    """
    code, out, _ = capture(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type,duration:format=duration",
         "-of", "default=nw=1", str(path)]
    )
    if code != 0 or "codec_type=audio" not in out:
        return None
    for line in out.splitlines():
        if not line.startswith("duration="):
            continue
        try:
            value = float(line.split("=", 1)[1])
        except ValueError:
            continue
        if value > 0:
            return value
    return None


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
    # Пишем рядом с целью и переименовываем: assets/sfx — общая папка, куда
    # смотрят все джобы. Пиши ffmpeg прямо в `dst`, и убитый на середине воркер
    # (kill, полный диск, гонка двух воркеров) оставил бы там обрезанный файл
    # НАВСЕГДА — а он валит весь звуковой пост, а не одну метку.
    staged = dst.with_suffix(f".{os.getpid()}.tmp{dst.suffix or '.wav'}")
    try:
        with tempfile.TemporaryDirectory(prefix="reelbot-sfx-") as tmp:
            raw = Path(tmp) / "raw.wav"
            ffmpeg([*inputs, "-filter_complex", ";".join(parts), "-map", "[out]",
                    "-c:a", "pcm_s16le", str(raw)], timeout=120)
            peak = _peak_dbfs(raw)
            gain = 0.0 if peak is None else peak_dbfs - peak
            # sample_fmt задаём явно: flac по умолчанию берёт разрядность из фильтра
            # (float32) и файл выходит больше исходного wav при том же звуке.
            ffmpeg(["-i", str(raw), "-af", f"volume={gain:.2f}dB", "-c:a", codec,
                    "-sample_fmt", "s16", str(staged)], timeout=120)
        os.replace(staged, dst)
    finally:
        staged.unlink(missing_ok=True)
    log.info("эффект %s собран: %s (%.2f с)", kind, dst, profile.duration)
    return dst


def sfx_path(name: str, *, cfg: Settings | None = None) -> Path | None:
    """Путь к ЧИТАЕМОМУ эффекту. Штатный эффект синтезируется, если файла нет.

    Найденный файл проверяется тем же `_audio_duration`, что и подложка, и по той
    же причине: `assets/sfx` — папка, куда кладут руками (см. её README), и туда
    же пишет `synth_sfx`. Битый файл в ней раньше валил `filter_complex`, а
    вместе с ним весь этап: ролик уезжал в рендер без loudnorm, музыки и
    эффектов. Одна метка эффекта не стоит всего звукового поста.
    """
    cfg = cfg or default_settings
    found = _find_asset(name, cfg.sfx_dir, SFX_SUFFIXES)
    if found is not None and _audio_duration(found) is not None:
        return found
    if found is not None:
        log.warning("эффект %s не читается — пробуем пересобрать", found)
    if name not in SFX_PROFILES:
        log.warning("эффект %r не найден (или не читается) в %s", name, cfg.sfx_dir)
        return None
    try:
        made = synth_sfx(cfg.sfx_dir / f"{name}{SFX_ASSET_SUFFIX}", kind=name)
    except Exception as e:  # noqa: BLE001 — звук не повод ронять ролик
        log.warning("не удалось синтезировать эффект %r: %s", name, e)
        return None
    if _audio_duration(made) is None:
        log.warning("собранный эффект %s не читается — метка пропущена", made)
        return None
    return made


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
    # Файл в assets/music кладёт человек, и он может положить что угодно: обрезанный
    # mp3, переименованный zip, пустышку. Раньше такой файл валил весь filter_complex,
    # а вместе с ним и loudnorm с эффектами — ролик уезжал в рендер с сырым звуком.
    # Дешевле проверить один раз ffprobe, чем терять весь этап из-за подложки.
    if _audio_duration(path) is None:
        log.warning("подложка %s не читается — ролик соберётся без музыки", path)
        return None
    # В EDL уезжает ИМЯ ассета, а не строка спецификации: `_find_asset` принимает
    # и абсолютный путь, а `EDL.music.track` по контракту (edl.py) — имя файла в
    # assets/music. Абсолютный путь в edl.json — это машинно-локальный артефакт:
    # на другой машине тот же EDL молча остался бы без музыки, а сам путь ещё и
    # утекал в отчёт джоба.
    if spec.track != path.name:
        log.info("подложка %s → в EDL пишем имя %s", spec.track, path.name)
        spec = spec.model_copy(update={"track": path.name})
    return path, spec


# --------------------------------------------------------------------------- расстановка


def _timeline(edl: EDL, *, apply_cuts: bool) -> Timeline:
    """Таймлайн, на котором живёт дорожка. Тот же переключатель, что в плане.

    `apply_cuts=False` (`cut_pauses=False` в джобе) значит, что видео не резали,
    а скопировали: картинка и субтитры остаются на исходном таймлайне, и звук
    обязан остаться там же. `render_plan.build_plan` делает ровно это.
    """
    return edl.timeline() if apply_cuts else Timeline([], edl.source_duration)


def _first_word_in(edl: EDL, start: float, end: float) -> Word | None:
    """Первое слово внутри куска исходника. None — в куске слов нет.

    Принадлежность считается по середине слова — так же, как её считает
    `kept_word_indices`: слово, начавшееся до реза и попавшее в кусок серединой,
    остаётся в ролике, и пропускать его нельзя.

    Верхняя граница обязательна: без неё в немом куске нашлось бы слово из
    следующего, и звук реза уехал бы на несколько секунд вперёд.
    """
    for w in edl.words:
        if start - 1e-6 <= w.mid <= end + 1e-6:
            return w
    return None


def _punch_ins(edl: EDL, tl: Timeline) -> list[tuple[int, float]]:
    """Резы с укрупнением: (слово-ссылка, момент РЕЗА на выходном таймлайне).

    Якорь — именно рез, а не первое слово после него. Крупность меняет
    `crop.framing_spans` по накопленным длительностям кусков, то есть ровно в
    `tl.map(cut.src_in)`; эффект, подчёркивающий рез, обязан ударить туда же.
    Привязка к слову опаздывала на всю паузу между резом и первым словом.
    """
    out: list[tuple[int, float]] = []
    kept = sorted(edl.kept_cuts, key=lambda c: c.src_in)
    for prev, cur in zip(kept, kept[1:]):
        if not (prev.framing == "wide" and cur.framing == "tight"):
            continue  # укрупнение, а не любой рез: пулбэк звуком не подчёркиваем
        w = _first_word_in(edl, cur.src_in, cur.src_out)
        at = tl.map(cur.src_in, clamp=False)
        if w is None or at is None:
            continue
        out.append((w.idx, at))
    return out


def plan_sfx(edl: EDL, *, preset: Preset, apply_cuts: bool = True) -> list[Sfx]:
    """Куда ставить звуковые акценты. Чистая функция от готового монтажа.

    Акцента заслуживают два события: вход картинки и рез с укрупнением
    (wide → tight). Оба — смена того, что зритель видит; звук её только
    подтверждает. Реза «в тот же план» и конца фразы это не касается: whoosh на
    каждом стыке превращается в стрекот.

    При `apply_cuts=False` резов в картинке нет вообще (кроп ведёт весь ролик
    одной крупностью — `crop.framing_spans`), поэтому и подчёркивать нечего:
    остаются только картинки.
    """
    tl = _timeline(edl, apply_cuts=apply_cuts)
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

    if apply_cuts:
        for w_idx, at in _punch_ins(edl, tl):
            if any(abs(at - t) < SFX_MIN_GAP_S for t, _ in accepted):
                continue
            accepted.append(
                (at, Sfx(at_word_idx=w_idx, name=SFX_SWISH, gain_db=gain - SWISH_TRIM_DB))
            )
            accepted.sort(key=lambda item: item[0])

    return [s for _, s in accepted]


@dataclass(frozen=True, slots=True)
class SfxMark:
    """Метка эффекта на выходном таймлайне, готовая к сборке фильтра."""

    at: float  # когда начать ФАЙЛ
    path: Path
    gain_db: float
    head_trim: float = 0.0  # сколько срезать с начала файла, чтобы удар попал в кадр


def sfx_marks(
    edl: EDL, sfx: list[Sfx], *, cfg: Settings | None = None, apply_cuts: bool = True
) -> list[SfxMark]:
    """`Sfx` → метки на выходном таймлайне.

    Эффект стартует раньше события ровно на свой пик: whoosh должен ударить в
    кадр, а не начаться на нём. Событие ближе к нулю, чем пик (картинка на первом
    слове — штатный хук), — срезаем голову файла на недостающее: иначе старт
    упирался бы в ноль, а удар уезжал вперёд на всю разницу.

    Swish привязан к резу, а не к слову: `at_word_idx` в контракте `Sfx` — это
    ссылка на событие, а точное время реза берётся из таймлайна (`_punch_ins`).
    """
    cfg = cfg or default_settings
    tl = _timeline(edl, apply_cuts=apply_cuts)
    by_idx = {w.idx: w for w in edl.words}
    punch_at = dict(_punch_ins(edl, tl)) if apply_cuts else {}
    marks: list[SfxMark] = []
    for s in sfx:
        w = by_idx.get(s.at_word_idx)
        win = tl.word_window(w) if w is not None else None
        if win is None:
            continue
        at = punch_at.get(s.at_word_idx, win[0]) if s.name == SFX_SWISH else win[0]
        path = sfx_path(s.name, cfg=cfg)
        if path is None:
            continue
        lead = SFX_PROFILES[s.name].peak_at if s.name in SFX_PROFILES else 0.0
        start = at - lead
        marks.append(
            SfxMark(
                at=max(0.0, start),
                path=path,
                gain_db=s.gain_db,
                head_trim=max(0.0, -start),
            )
        )
    marks.sort(key=lambda m: m.at)
    return marks


# --------------------------------------------------------------------------- сведение


def duck_ratio(duck_db: float, *, target_lufs: float = -16.0) -> float:
    """Коэффициент компрессии, дающий примерно `duck_db` подавления музыки.

    Запас считается от цели громкости голоса: после loudnorm речь живёт на
    `target_lufs`, порог сайдчейна стоит на `DUCK_THRESHOLD_DB`, разница между
    ними — это те самые over_db, из которых компрессор делает подавление. Оба
    числа приходят из пресета, поэтому и глубина следует за пресетом: смените
    `target_lufs` на −20, и `duck_db` продолжит означать ровно то, что написано.

    Именно «примерно»: компрессор давит по огибающей, поэтому на тихой фразе
    музыка приседает меньше, чем на громкой. Это не баг, а то, ради чего берут
    компрессор, а не выключатель: дакинг фиксированной глубины слышно как
    дыхание фона на каждой паузе между словами.
    """
    headroom = max(DUCK_MIN_HEADROOM_DB, target_lufs - DUCK_THRESHOLD_DB)
    want = min(abs(duck_db), headroom - 2.0)
    return max(1.0, min(20.0, 1.0 / (1.0 - want / headroom)))


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
    apply_cuts: bool = True,
    sfx: bool = True,
    cfg: Settings | None = None,
) -> AudioTrack | None:
    """Нарезанное видео + EDL → готовая звуковая дорожка. None, если звука нет.

    Вход — именно `cut.mp4`: дорожка обязана лежать на выходном таймлайне, том
    же, по которому построен `RenderPlan`. Кроп длительность не меняет, поэтому
    считать звук до него можно и нужно (кроп его всё равно не трогает).

    `apply_cuts` — тот же переключатель, что у `build_plan`: `False` значит, что
    паузы не вырезали (джоб просто скопировал исходник), и звук остаётся на
    исходном таймлайне вместе с картинкой и субтитрами. Передавать сюда надо то
    же самое, что и в план, иначе эффекты уедут на суммарную длину вырезанного.

    Побочный эффект: заполняет `edl.sfx`, если он пуст, и проставляет
    `edl.music` по факту резолва (в том числе сбрасывает в None, когда подложка
    не нашлась) — чтобы `edl.json` описывал то, что реально в миксе. Повторный
    вызов на том же EDL с другим пресетом переиспользует уже сохранённую
    расстановку эффектов.
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
    # Длину берём по максимуму из файла и плана, а не «как получится».
    # `info.duration` — длина контейнера, и она бывает короче того, что рисует
    # рендерер: план строит длительность из EDL (`Timeline.duration`), а звуковой
    # поток внутри mp4 нередко на кадр-другой короче видеопотока. Дорожка короче
    # ролика — это тишина в хвосте, которую никто не заметит на приёмке и заметит
    # зритель.
    # ⚠ Обратная сторона: дорожка бывает на десяток миллисекунд ДЛИННЕЕ
    # видеопотока. Remotion это переживает (длина задана `durationInFrames`), а
    # `subs_ass.burn` мультиплексирует `-map 0:v -map 1:a` без `-shortest` — и
    # хвост уезжает в отданный mp4: контейнер длиннее картинки. Лечится в
    # `subs_ass` одним флагом (чужой лейн), укорачивать звук под это нельзя.
    tl = _timeline(edl, apply_cuts=apply_cuts)
    total = max(info.duration, tl.duration)
    if total <= 0:
        return None

    # `sfx=False` — пользователь снял галочку «звуковые эффекты». Громкость и
    # подложка при этом остаются: выключаются только акценты. EDL тоже очищается,
    # чтобы сохранённый лист описывал реальный микс, а не отвергнутые эффекты.
    if not sfx:
        edl.sfx = []
    elif not edl.sfx:
        edl.sfx = plan_sfx(edl, preset=preset, apply_cuts=apply_cuts)
    marks = sfx_marks(edl, edl.sfx, cfg=cfg, apply_cuts=apply_cuts)
    music = resolve_music(edl, preset=preset, cfg=cfg)
    # edl.json — артефакт джоба, и он обязан описывать РЕАЛЬНЫЙ микс. Раньше
    # незарезолвленная подложка оставалась в EDL, и два артефакта одного джоба
    # противоречили друг другу: отчёт писал music=None, а edl.json — имя трека.
    edl.music = music[1] if music is not None else None

    target = preset.audio.target_lufs
    measured = measure_loudness(video, target_lufs=target)
    inputs: list[str] = ["-i", str(video)]
    # `apad` + `atrim` фиксируют длину голоса на `total` ДО микса. Без них длину
    # задавал только `-t`, а он умеет резать, но не умеет добавлять: если звуковой
    # поток входа кончался раньше видео (обычное дело после concat), ролик получал
    # дорожку короче себя. Плюс `amix=duration=first` меряет длину микса по первому
    # входу — то есть ровно по этому голосу.
    parts: list[str] = [
        f"[0:a]{loudnorm_filter(measured, target_lufs=target)},{AFORMAT},"
        f"atrim=0:{total:.3f},asetpts=N/SR/TB,apad=whole_dur={total:.3f}[voice]"
    ]
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
            f":ratio={duck_ratio(spec.duck_db, target_lufs=target):.2f}"
            f":attack={DUCK_ATTACK_MS}:release={DUCK_RELEASE_MS}:level_sc=1[music_duck]"
        )
        mix = ["[voice_mix]", "[music_duck]"]
        stream += 1

    for i, m in enumerate(marks):
        inputs += ["-i", str(m.path)]
        chain = [AFORMAT]
        if m.head_trim > 1e-3:
            # Событие ближе к началу ролика, чем пик эффекта: старт упёрся в
            # ноль, и без подрезки головы удар прозвучал бы ПОСЛЕ картинки.
            chain += [f"atrim=start={m.head_trim:.3f}", "asetpts=N/SR/TB"]
        chain.append(f"volume={m.gain_db:g}dB")
        chain.append(f"adelay={round(m.at * 1000)}:all=1")
        parts.append(f"[{stream}:a]" + ",".join(chain) + f"[sfx{i}]")
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
    # Длину читаем с готового файла, а не повторяем то, что просили: рассинхрон
    # звука с видео — самый дорогой дефект этапа (субтитры уезжают от речи), и
    # узнавать о нём надо здесь, а не по жалобе на ролик.
    real = _audio_duration(out)
    if real is not None and abs(real - total) > 0.05:
        log.warning(
            "дорожка %s вышла %.3f с вместо %.3f — звук разъедется с видео",
            out.name, real, total,
        )
    lufs_in = (measured or {}).get("input_i")
    # На немом входе loudnorm печатает «-inf»: в отчёте джоба это превращается в
    # `-Infinity`, которого нет в JSON и который ломает любого читателя отчёта.
    if lufs_in is not None and not math.isfinite(lufs_in):
        lufs_in = None
    track = AudioTrack(
        path=out,
        duration=real if real is not None else total,
        music=music[1].track if music else None,
        sfx=len(marks),
        lufs_in=lufs_in,
        target_lufs=target,
    )
    log.info(
        "звук сведён: %s (%.2f с), вход %.1f LUFS → цель %.1f LUFS, музыка=%s, эффектов=%d",
        out.name, track.duration, lufs_in if lufs_in is not None else float("nan"),
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
