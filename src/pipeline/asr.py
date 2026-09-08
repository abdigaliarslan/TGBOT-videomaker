"""M1: ASR с пословным выравниванием → list[Word].

Бэкенд выбирается один раз и абстрагирован, потому что стек Whisper ломается
на мажорных апдейтах (§5 M1 ⚠), а версии зафиксированы в pyproject.toml.

- `whisperx`     — форсированное выравнивание wav2vec2: границы слов считаются
                   отдельной моделью по звуку, а не берутся у Whisper. Требует
                   torch, но CPU ему достаточно — замеры в `pick_backend`.
                   Это рабочий бэкенд по умолчанию.
- `faster_whisper` — CTranslate2, работает на CPU, тайминги слов берутся из
                   cross-attention самой модели. Втрое легче по памяти, но
                   промахивается по границам слов вчетверо сильнее (216 мс
                   против 46 на фикстуре, замеры в `pick_backend`). Фолбэк.

CLI: python -m src.pipeline.asr <file> > words.json
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.core.config import settings
from src.core.edl import Word
from src.pipeline.audio import ASR_SAMPLE_RATE, extract_wav

log = logging.getLogger(__name__)

Backend = Literal["whisperx", "faster_whisper"]

# Загрузка весов с CDN умеет вставать намертво: соединение живо, данные не идут.
# Без таймаута это вешает задачу целиком, и пользователь смотрит на замерший
# прогресс-бар. Пусть лучше упадёт — сработает откат на другой бэкенд.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")

# Модели тяжёлые (сотни МБ) — грузим один раз на процесс.
_MODEL_CACHE: dict[tuple, Any] = {}


@dataclass(slots=True)
class ASRResult:
    words: list[Word]
    language: str
    backend: str
    model: str
    duration: float = 0.0
    dropped_segments: int = 0  # отброшенные галлюцинации
    raw_text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "backend": self.backend,
            "model": self.model,
            "duration": self.duration,
            "dropped_segments": self.dropped_segments,
            "text": self.raw_text,
            "words": [w.model_dump() for w in self.words],
        }

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


# ------------------------------------------------------------------ выбор бэкенда


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 — отсутствие CUDA не должно ронять пайплайн
        return False


def _has_whisperx() -> bool:
    from importlib.util import find_spec

    return find_spec("whisperx") is not None


def pick_backend(requested: str = "auto") -> Backend:
    if requested == "whisperx":
        if not _has_whisperx():
            raise RuntimeError("asr_backend=whisperx, но пакет whisperx не установлен")
        return "whisperx"
    if requested == "faster_whisper":
        return "faster_whisper"
    # auto: whisperx, если установлен — даже на CPU.
    #
    # Все числа ниже сняты при ASR_TRIM_WORD_SILENCE=1, то есть на той самой
    # конфигурации, которая уезжает в прод; колонка «без обрезки» оставлена
    # рядом, чтобы сравнение не смешивало два разных изменения.
    #
    # Точность границ фраз на фикстуре tests/fixtures/speech.mp4 (15.5 с,
    # английский синтез, эталон размечен при её генерации; tests/asr_accuracy.py),
    # среднее |Δ| / максимум:
    #
    #   бэкенд/модель          без обрезки      с обрезкой (дефолт)
    #   faster_whisper/small   234 / 829 мс      216 / 829 мс
    #   faster_whisper/medium  185 / 669 мс      119 / 272 мс
    #   whisperx/small          46 / 101 мс       46 / 101 мс
    #   whisperx/medium         46 / 101 мс       46 / 101 мс
    #
    # То же на РАБОЧЕМ языке (samples/phase-0-source.mp4, русский, 20 с; эталон —
    # возобновление речи после паузы по энергии, `asr_accuracy.py --media`).
    # Отклонение НАЧАЛА первого слова после паузы:
    #
    #   бэкенд/модель          без обрезки      с обрезкой (дефолт)
    #   faster_whisper/small   277 / 949 мс       88 / 264 мс  (+1 пауза без слова)
    #   faster_whisper/medium  149 / 349 мс      104 / 349 мс
    #   whisperx/small          25 /  53 мс       25 /  53 мс
    #   whisperx/medium         25 /  53 мс       25 /  53 мс
    #
    # Разрыв на поставляемых настройках — 4.7× на фикстуре (216 против 46) и
    # 3.5× на русском (88 против 25), а не «пятикратный/четырнадцатикратный»:
    # прежние числа сравнивали whisperx с обрезкой против faster_whisper без неё
    # и считались измерителем, который брал БЛИЖАЙШЕЕ слово вместо того, которое
    # стоит после паузы. Сверх среднего у faster_whisper/small есть отказ, какого
    # у whisperx нет вовсе: одно слово он растянул через паузу целиком, и точки
    # эталона там просто не к чему приложить.
    #
    # Отдельно про faster_whisper/medium — это честный конкурент (104/349 мс на
    # русском, 119/272 на фикстуре, 2.0 ГБ, без torch и без модели выравнивания).
    # Выбран всё-таки whisperx/small: вчетверо точнее (25/53) при сопоставимом
    # времени, ценой лишнего гигабайта памяти и разовой закачки весов выравнивания.
    #
    # Границы слов у whisperx считает wav2vec2 по звуку, а не Whisper по своему
    # кросс-аттеншену, поэтому НА ОДНОМ И ТОМ ЖЕ ТЕКСТЕ размер модели Whisper на
    # них не влияет: на английской фикстуре small и medium дали побуквенно
    # одинаковые списки слов (29 слов, те же тайминги до миллисекунды). На
    # русском medium меняет сам текст — 38 слов против 39 («насчет того» против
    # «о том», «в этой мире» против «в этом мире»), и вместе с текстом едет
    # сегментация; там, где слова совпали, тайминги снова совпадают точно.
    # Отсюда whisper_model="small": medium вдвое дороже, попадание в речь не
    # улучшает (25/53 мс у обоих), а текст на этом ролике портит.
    #
    # Цена — на том же 20-секундном ролике (12 ядер, без GPU, int8, 4 потока;
    # «тёплый» = модели уже загружены прогревом воркера, машина под нагрузкой,
    # разброс по времени до двух раз, память от загрузки почти не зависит):
    #
    #   бэкенд/модель          холодный   тёплый   пик RSS
    #   faster_whisper/small      7.2 с     3.7 с    1.1 ГБ
    #   faster_whisper/medium    16.3 с    12.3 с    2.0 ГБ
    #   whisperx/small           21.6 с    11.3 с    3.0 ГБ
    #   whisperx/medium          31.4 с    16.3 с    3.7 ГБ
    if _has_whisperx():
        return "whisperx"
    return "faster_whisper"


def _device_and_compute() -> tuple[str, str]:
    device = settings.whisper_device
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    compute = settings.whisper_compute_type
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


# ------------------------------------------------------------------ нормализация


def _clean(text: str) -> str:
    return text.strip()


def _finalize(raw: list[tuple[str, float, float, float]], *, duration: float) -> list[Word]:
    """Из сырых (text, start, end, prob) в list[Word]: чистка, монотонность, индексы.

    Индекс равен позиции в списке — на этот инвариант опирается весь EDL (§3).
    """
    words: list[Word] = []
    prev_end = 0.0
    for text, start, end, prob in raw:
        t = _clean(text)
        if not t:
            continue
        s = max(0.0, float(start))
        e = float(end)
        if e <= s:
            e = s + 0.06
        # ASR иногда отдаёт перекрывающиеся границы — выпрямляем, иначе субтитры дрожат.
        if s < prev_end:
            s = prev_end
            if e <= s:
                e = s + 0.06
        if duration > 0:
            s = min(s, duration)
            e = min(e, duration)
            if e <= s:
                continue
        words.append(
            Word(
                idx=len(words),
                text=t,
                start=round(s, 3),
                end=round(e, 3),
                prob=round(float(prob), 3),
            )
        )
        prev_end = e
    return words


# ------------------------------------------------- обрезка тишины на краях слова
#
# Оба бэкенда растягивают ПОСЛЕДНЕЕ слово перед паузой на саму паузу: на эталонном
# ролике whisperx отдаёт «других» 10.80..13.00, хотя речь там кончается на 11.17
# (замер энергии, tests/asr_accuracy.py --media). Для подсветки это ещё терпимо
# (слово просто висит), а вот для нарезки пауз — нет: director.build_cuts считает
# паузу как `w.start - prev.end`, и проглоченная словом тишина в неё не попадает.
# Пауза остаётся в ролике, причём внутри «ядра» куска, которое нельзя трогать.
#
# Поэтому границы слова подрезаются по энергии сигнала. Только внутрь: время речи
# мы не выдумываем, а отдаём назад то, что бэкенд прихватил из тишины.
#
# ⚠ Проверять обрезку метрикой «отклонение конца слова от энергии» нельзя: это
# тот же RMS по тому же wav, и она обязана улучшиться (537 → 14 мс на русском —
# согласие двух замеров энергии, а не независимая точность). Независимые
# подтверждения два: на фикстуре с РАЗМЕЧЕННЫМИ границами faster_whisper/medium
# идёт 185 → 119 мс, /small 234 → 216 (whisperx там не трогается вовсе — обрезать
# нечего); и нарезка пауз: build_cuts на том же ролике вырезает 0.98 с без
# обрезки против 3.44 с с ней — те самые паузы, спрятанные внутрь слов.

_ENV_HOP = 0.010  # шаг огибающей, с
_ENV_WIN = 0.020  # окно RMS, с
# Короче 150 мс — не пауза, а придыхание или атака согласной; трогать её значит
# резать речь ради статистики.
_TRIM_MIN = 0.15
_TRIM_MIN_WORD = 0.06  # столько слово живёт даже после обрезки
# Порог «речь/тишина» ставится между шумовым полом и уровнем речи самой записи:
# абсолютных дБ тут быть не может, громкость исходника не нормирована.
_TRIM_FLOOR_Q, _TRIM_SPEECH_Q = 10.0, 90.0
# Доля пути от шумового пола к уровню речи. Это ПОДГОНКА ПОД ОДИН РОЛИК, и знать
# об этом важнее, чем само значение: свип по samples/phase-0-source.mp4 даёт
# 0.2 → 273 мс, 0.35 → 132, 0.5 → 14, 0.65 → 54 (отклонение конца речи). Разброс
# в 20 раз от шага 0.15, n=1 — и мерено энергией по тому же wav, по которому
# считает сама обрезка, то есть замер согласованный, а не независимый.
# Второй ролик (фикстура, эталон размечен независимо) оптимума не подтверждает и
# не опровергает: там порог уходит на −66 дБ при шумовом поле −120, обрезка почти
# не срабатывает, и метрика от 0.2 до 0.8 только улучшается (218 → 168 мс на
# faster_whisper/small). Менять значение без третьего ролика с настоящим шумовым
# полом смысла нет; считать его найденным оптимумом — тоже.
_TRIM_MIX = 0.50
# Потолок в абсолютных дБ: на записи с громким фоном относительный порог поднялся
# бы внутрь речи. Выше него тишины не бывает — лучше не обрезать вовсе.
_TRIM_MAX_DB = -40.0
# Меньший разброс уровней — на записи нет тишины, отделять нечего.
_TRIM_MIN_RANGE_DB = 12.0
# Одиночный щелчок не должен держать границу: речью считается серия кадров.
_TRIM_RUN = 3
_TRIM_RUN_MIN = 3


def _envelope(wav: Path) -> tuple[np.ndarray, float] | None:
    """RMS-огибающая моно-wav в дБ и шаг кадра. None — если файл не PCM.

    Читается stdlib-модулем `wave`: ASR всё равно работает с wav, который сам же
    и извлёк (моно 16 кГц PCM16), а лишний запуск ffmpeg ради замера энергии —
    это ещё один процесс на каждый ролик.
    """
    # Разбор целиком внутри try: обещание «битый файл — просто без обрезки»
    # держится только так. У усечённого wav (файл дописан не до конца) нечётное
    # число байт, и np.frombuffer роняет ValueError уже ПОСЛЕ распознавания —
    # то есть задача падала бы на ровном месте, с готовым транскриптом в руках.
    try:
        with wave.open(str(wav), "rb") as f:
            if f.getsampwidth() != 2:
                return None
            sr, channels = f.getframerate(), f.getnchannels()
            raw = f.readframes(f.getnframes())
        samples = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            samples = (
                samples[: len(samples) // channels * channels].reshape(-1, channels).mean(axis=1)
            )
    except Exception as e:  # noqa: BLE001 — не PCM, битый файл: обрезки просто не будет
        log.debug("огибающая недоступна (%s): %s", type(e).__name__, e)
        return None

    hop, win = int(sr * _ENV_HOP), int(sr * _ENV_WIN)
    if hop <= 0 or len(samples) < win:
        return None
    # Скользящее окно через кумулятивную сумму квадратов: матрица кадров на
    # трёхминутном ролике весила бы под полсотни мегабайт, а нужен один RMS.
    squares = np.concatenate(([0.0], np.cumsum(samples.astype(np.float64) ** 2)))
    starts = np.arange(1 + (len(samples) - win) // hop) * hop
    rms = np.sqrt((squares[starts + win] - squares[starts]) / win + 1e-12)
    return 20.0 * np.log10(rms + 1e-12), hop / sr


def _loud_mask(db: np.ndarray, threshold: float) -> np.ndarray:
    """Кадры, попавшие в серию из `_TRIM_RUN`, где громких не меньше `_TRIM_RUN_MIN`."""
    loud = (db > threshold).astype(np.float32)
    if len(loud) < _TRIM_RUN:
        return loud > 0
    kernel = np.ones(_TRIM_RUN, dtype=np.float32)
    return np.convolve(loud, kernel, mode="same") >= _TRIM_RUN_MIN


def trim_word_silence(words: list[Word], wav: Path) -> int:
    """Подрезать края слов до фактической речи. Правит список на месте.

    Возвращает число изменённых слов. Границы только сдвигаются ВНУТРЬ слова,
    поэтому монотонность (end[i] <= start[i+1]) сохраняется по построению.
    """
    env = _envelope(wav)
    if env is None or not words:
        return 0
    db, hop = env
    floor, speech = np.percentile(db, [_TRIM_FLOOR_Q, _TRIM_SPEECH_Q])
    if speech - floor < _TRIM_MIN_RANGE_DB:
        # Ровный уровень: либо сплошная речь без пауз, либо запись без динамики.
        # Порога, отделяющего тишину, здесь просто нет — не выдумываем его.
        log.debug("обрезка тишины пропущена: разброс уровней %.1f дБ", speech - floor)
        return 0
    threshold = min(float(floor + _TRIM_MIX * (speech - floor)), _TRIM_MAX_DB)
    mask = _loud_mask(db, threshold)

    changed = 0
    for w in words:
        i0 = max(0, round(w.start / hop))
        # Правая граница исключающая: кадр, начинающийся ровно на конце слова,
        # принадлежит уже следующему — иначе его речь удержала бы границу на месте.
        i1 = min(len(mask), round(w.end / hop))
        if i1 - i0 < 2:
            continue
        loud = np.nonzero(mask[i0:i1])[0]
        if len(loud) == 0:
            # Слово целиком под порогом — скорее ошибка порога, чем слова.
            continue
        # Время считается по НОМЕРУ кадра, а не от начала слова: слово начинается
        # между кадрами, и отсчёт от него сдвинул бы обе границы на полкадра.
        start = float(i0 + loud[0]) * hop
        end = float(i0 + loud[-1] + 1) * hop
        new_start = start if start - w.start >= _TRIM_MIN else w.start
        new_end = end if w.end - end >= _TRIM_MIN else w.end
        new_end = max(new_end, new_start + _TRIM_MIN_WORD)
        new_start = min(new_start, new_end - _TRIM_MIN_WORD)
        # Наружу не выпускаем: обрезка не имеет права ДОБАВИТЬ слову времени.
        new_start, new_end = max(new_start, w.start), min(new_end, w.end)
        if new_end - new_start < _TRIM_MIN_WORD:
            continue
        if new_start != w.start or new_end != w.end:
            w.start, w.end = round(new_start, 3), round(new_end, 3)
            changed += 1
    return changed


# ------------------------------------------------------------------ бэкенды


def _transcribe_faster_whisper(wav: Path, *, language: str | None, model_name: str) -> ASRResult:
    from faster_whisper import WhisperModel

    device, compute = _device_and_compute()
    key = ("fw", model_name, device, compute, settings.whisper_cpu_threads)
    model = _MODEL_CACHE.get(key)
    if model is None:
        log.info("загружаю faster-whisper %s (%s/%s)", model_name, device, compute)
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute,
            cpu_threads=settings.whisper_cpu_threads,
        )
        _MODEL_CACHE[key] = model

    segments, info = model.transcribe(
        str(wav),
        language=language or None,
        word_timestamps=True,
        beam_size=settings.whisper_beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        # Whisper без этого зацикливается и дописывает несказанное.
        condition_on_previous_text=False,
    )

    raw: list[tuple[str, float, float, float]] = []
    texts: list[str] = []
    dropped = 0
    for seg in segments:
        seg_words = list(getattr(seg, "words", None) or [])
        if not seg_words:
            continue
        probs = [float(getattr(w, "probability", 1.0) or 0.0) for w in seg_words]
        avg = sum(probs) / len(probs)
        # Галлюцинации на музыке/шуме: низкая средняя уверенность слов.
        if avg < settings.whisper_min_confidence:
            dropped += 1
            log.debug("отброшен сегмент (conf=%.2f): %r", avg, seg.text)
            continue
        texts.append(seg.text)
        for w in seg_words:
            raw.append(
                (w.word, w.start, w.end, float(getattr(w, "probability", 1.0) or 1.0))
            )

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    return ASRResult(
        words=_finalize(raw, duration=duration),
        language=getattr(info, "language", language or "") or "",
        backend="faster_whisper",
        model=model_name,
        duration=duration,
        dropped_segments=dropped,
        raw_text=" ".join(t.strip() for t in texts).strip(),
        meta={"device": device, "compute_type": compute},
    )


def _limit_torch_threads() -> None:
    """Ограничить потоки torch тем же числом, что и ctranslate2.

    `whisperx.load_model(threads=...)` держит только распознавание; выравнивание
    считает torch, а он по умолчанию берёт все ядра — тот самый пик тока, из-за
    которого гаснет ноутбук (см. _ffmpeg_cmd в core/proc.py). Замерено: на 20 с
    ролике 4 потока против «всех» дают 22.2 с против 21.3 с (разница в пределах
    разброса), зато средняя загрузка падает с 345% до 256%.
    """
    try:
        import torch

        torch.set_num_threads(max(1, settings.whisper_cpu_threads))
    except Exception as e:  # noqa: BLE001 — ограничение потоков не условие работы
        log.debug("не удалось ограничить потоки torch: %s", e)


def _whisperx_model(model_name: str, language: str | None):
    """Модель whisperx из кеша. Ключ включает язык: он влияет на загрузку."""
    import whisperx  # type: ignore[import-not-found]

    device, compute = _device_and_compute()
    key = ("wx", model_name, device, compute, language or "")
    model = _MODEL_CACHE.get(key)
    if model is None:
        log.info("загружаю whisperx %s (%s/%s)", model_name, device, compute)
        # threads обязателен: без него whisperx берёт все ядра, а это тот самый
        # пик тока, из-за которого ноутбук гаснет (см. _ffmpeg_cmd в core/proc.py).
        model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute,
            language=language or None,
            threads=settings.whisper_cpu_threads,
        )
        _MODEL_CACHE[key] = model
    return model


def _transcribe_whisperx(wav: Path, *, language: str | None, model_name: str) -> ASRResult:
    """whisperx с откатом на faster_whisper на ЛЮБОМ его отказе.

    Раньше откат стоял только вокруг модели выравнивания, а рядом лежат ещё три
    места, где whisperx падает штатно: загрузка весов (сеть, HF_HUB_DOWNLOAD_TIMEOUT),
    нехватка памяти на torch (3 ГБ против 1 ГБ у faster_whisper) и битый кеш.
    Пайплайн вызывает transcribe() без обработчика — любой из них ронял задачу,
    хотя рабочий бэкенд лежит рядом и даёт хоть и более грубые, но границы.
    """
    try:
        return _whisperx_impl(wav, language=language, model_name=model_name)
    except Exception as e:  # noqa: BLE001 — падение бэкенда не должно ронять задачу
        log.warning(
            "whisperx не справился (%s: %s) — считаю на faster-whisper, "
            "границы слов будут грубее",
            type(e).__name__, e,
        )
        return _transcribe_faster_whisper(wav, language=language, model_name=model_name)


def _whisperx_impl(wav: Path, *, language: str | None, model_name: str) -> ASRResult:
    import whisperx  # type: ignore[import-not-found]

    device, compute = _device_and_compute()
    _limit_torch_threads()
    model = _whisperx_model(model_name, language)

    audio = whisperx.load_audio(str(wav))
    result = model.transcribe(audio, batch_size=8, language=language or None)
    lang = result.get("language", language or "")

    align_key = ("wx-align", lang, device)
    aligned_model = _MODEL_CACHE.get(align_key)
    if aligned_model is None:
        try:
            aligned_model = whisperx.load_align_model(language_code=lang, device=device)
        except Exception as e:  # noqa: BLE001 — сеть, отсутствие модели, что угодно
            # Выравнивание — улучшение точности, а не условие работы. Ролик
            # важнее лишних 200 мс на границах слов.
            log.warning(
                "модель выравнивания для языка %r недоступна (%s) — "
                "считаю на faster-whisper, границы слов будут грубее",
                lang, e,
            )
            return _transcribe_faster_whisper(wav, language=lang, model_name=model_name)
        _MODEL_CACHE[align_key] = aligned_model
    align_model, metadata = aligned_model

    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, device, return_char_alignments=False
    )

    raw: list[tuple[str, float, float, float]] = []
    dropped = 0
    unaligned = 0
    for seg in aligned.get("segments", []):
        seg_words = seg.get("words") or []
        usable = [w for w in seg_words if w.get("start") is not None and w.get("end") is not None]
        # Слово без границ выравнивание оставляет, когда в нём нет ни одного
        # символа из словаря wav2vec2 (цифры, латиница в русском тексте): в
        # alignment.py оно так и уходит с start=end=None, а интерполяция чинит
        # только предложения, где уцелело хоть одно слово. Такое слово ПРОПАДАЕТ
        # из субтитров — зритель это видит, поэтому оно должно быть в логе.
        for w in seg_words:
            if w.get("start") is None or w.get("end") is None:
                unaligned += 1
                log.warning("слово без границ после выравнивания, выброшено: %r", w.get("word"))
        if not usable:
            continue
        probs = [float(w.get("score", 1.0) or 0.0) for w in usable]
        if sum(probs) / len(probs) < settings.whisper_min_confidence:
            dropped += 1
            continue
        for w in usable:
            raw.append((w["word"], w["start"], w["end"], float(w.get("score", 1.0) or 1.0)))

    return ASRResult(
        words=_finalize(raw, duration=len(audio) / ASR_SAMPLE_RATE),
        language=lang,
        backend="whisperx",
        model=model_name,
        duration=len(audio) / ASR_SAMPLE_RATE,
        dropped_segments=dropped,
        raw_text=" ".join(s.get("text", "").strip() for s in aligned.get("segments", [])).strip(),
        meta={"device": device, "compute_type": compute, "unaligned_words": unaligned},
    )


# ------------------------------------------------------------------ публичный вход


def transcribe(
    src: str | Path,
    *,
    language: str | None = None,
    model: str | None = None,
    backend: str | None = None,
) -> ASRResult:
    """Видео или wav → пословный транскрипт.

    Не-wav вход сначала прогоняется через ffmpeg (моно 16 кГц).
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)

    lang = settings.whisper_language if language is None else language
    model_name = model or settings.whisper_model
    chosen = pick_backend(backend or settings.asr_backend)

    tmp: Path | None = None
    wav = src
    if src.suffix.lower() != ".wav":
        tmp = Path(tempfile.mkdtemp(prefix="reelbot-asr-")) / "audio.wav"
        wav = extract_wav(src, tmp)
    try:
        fn = _transcribe_whisperx if chosen == "whisperx" else _transcribe_faster_whisper
        result = fn(wav, language=lang or None, model_name=model_name)
        if settings.asr_trim_word_silence:
            trimmed = trim_word_silence(result.words, wav)
            result.meta["trimmed_words"] = trimmed
            if trimmed:
                log.info("обрезано тишины на краях слов: %d слов из %d", trimmed, len(result.words))
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
            tmp.parent.rmdir()

    log.info(
        "ASR: %d слов, язык=%s, бэкенд=%s/%s, отброшено сегментов=%d",
        len(result.words),
        result.language,
        result.backend,
        result.model,
        result.dropped_segments,
    )
    return result


def prewarm(language: str | None = None, model: str | None = None) -> None:
    """Скачать и загрузить модели заранее — на старте воркера, а не в задаче.

    Модель выравнивания зависит от ЯЗЫКА и качается при первом ролике на этом
    языке: для русского это ~1.2 ГБ. Если делать это внутри задачи, пользователь
    смотрит на замерший прогресс-бар, а зависшая закачка (такое бывает) вешает
    задачу целиком. То же правило, что и для Chromium в рендере.
    """
    lang = (settings.whisper_language if language is None else language) or ""
    model_name = model or settings.whisper_model
    chosen = pick_backend(settings.asr_backend)
    log.info("прогрев ASR: бэкенд=%s модель=%s язык=%s", chosen, model_name, lang or "auto")

    if chosen == "faster_whisper":
        _transcribe_faster_whisper  # noqa: B018 — модель грузится при первом вызове
        from faster_whisper import WhisperModel

        device, compute = _device_and_compute()
        key = ("fw", model_name, device, compute, settings.whisper_cpu_threads)
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = WhisperModel(
                model_name, device=device, compute_type=compute,
                cpu_threads=settings.whisper_cpu_threads,
            )
        return

    import whisperx  # type: ignore[import-not-found]

    device, _compute = _device_and_compute()
    _limit_torch_threads()
    _whisperx_model(model_name, lang or None)

    if not lang:
        # Язык определяется по звуку — модель выравнивания заранее не выбрать.
        log.info("прогрев: язык не задан, модель выравнивания подтянется на первом ролике")
        return
    align_key = ("wx-align", lang, device)
    if align_key not in _MODEL_CACHE:
        _MODEL_CACHE[align_key] = whisperx.load_align_model(language_code=lang, device=device)
    log.info("прогрев ASR завершён")


def words_from_json(payload: dict | list) -> list[Word]:
    """Обратная загрузка words.json (для перерендера из сохранённого EDL, §0.4)."""
    items = payload["words"] if isinstance(payload, dict) else payload
    return [Word(**w) for w in items]


if __name__ == "__main__":  # python -m src.pipeline.asr file.mp4 > words.json
    import sys

    from src.core.logging import setup_logging

    setup_logging(settings.log_level)
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m src.pipeline.asr <file> > words.json")
    res = transcribe(sys.argv[1])
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
