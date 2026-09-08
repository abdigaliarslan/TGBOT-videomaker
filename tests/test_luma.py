"""Замер яркости фона под субтитром (Ф9).

Синтетика везде: клипы с заведомо известной яркостью генерирует ffmpeg, поэтому
ожидания здесь — не «похоже на правду», а числа, которые обязаны сойтись. Четыре
ловушки, ради которых тесты и написаны: порядок каналов, матрица декодирования
(bt709 по тегу потока, а не bt601 «как обычно»), область замера (прямоугольник
панели, а не весь кадр) и таймлайн (выходной, а не исходника).

Серые и чёрно-белые клипы ни первую, ни вторую ловушку не ловят: там R=G=B, и
порядок каналов с матрицей на результат не влияют. Поэтому есть отдельный
цветной клип через настоящий декодер — `test_colour_clip_decodes_by_stream_tag`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.edl import EDL, CaptionBlock, CaptionWord, Cut, Word
from src.core.geometry import Rect
from src.core.preset import load_preset
from src.core.proc import ffmpeg
from src.pipeline.luma import LumaReport, frame_luma, measure_bg_luma

W, H = 1080, 1920
FPS = 24
PRESET = load_preset("glass")

# Допуск на весь тракт «RGB → yuv420p limited → h264 → BGR». Округление
# диапазона и хроматическая субдискретизация дают единицы кода из 255.
TOL = 0.02


# ------------------------------------------------------------------ вспомогательное


def wcag_luminance(rgb: tuple[int, int, int]) -> float:
    """Ожидание, посчитанное по букве WCAG: разгамить каналы, потом свернуть.

    Формула написана здесь заново нарочно: тест обязан знать правильный ответ
    сам, а не спрашивать его у проверяемого модуля.
    """
    def linear(code: int) -> float:
        c = code / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def solid(color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Кадр одного цвета. `color` — в порядке RGB, как отдаёт ffmpeg (rgb24)."""
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def clip_split_in_time(path, first: str, second: str, *, half: float = 1.0) -> None:
    """Клип, у которого первая половина одного цвета, вторая — другого."""
    ffmpeg([
        "-f", "lavfi", "-i", f"color=c={first}:s={W}x{H}:r={FPS}:d={half:g}",
        "-f", "lavfi", "-i", f"color=c={second}:s={W}x{H}:r={FPS}:d={half:g}",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
        "-pix_fmt", "yuv420p", str(path),
    ])


def clip_colour(path, colour: str, *, duration: float = 1.0) -> None:
    """Одноцветный клип с ЯВНЫМ тегом bt709 — так помечает свой выход кроп.

    Без тега спор о матрице не поставить: декодер вправе гадать по разрешению.
    """
    ffmpeg([
        "-f", "lavfi", "-i", f"color=c={colour}:s={W}x{H}:r={FPS}:d={duration:g}",
        "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-color_range", "tv",
        "-vf", "scale=out_color_matrix=bt709", str(path),
    ])


def clip_split_in_space(path, top: str, bottom: str, *, duration: float = 1.0) -> None:
    """Клип, у которого верх кадра одного цвета, низ — другого."""
    ffmpeg([
        "-f", "lavfi", "-i", f"color=c={top}:s={W}x{H // 2}:r={FPS}:d={duration:g}",
        "-f", "lavfi", "-i", f"color=c={bottom}:s={W}x{H // 2}:r={FPS}:d={duration:g}",
        "-filter_complex", "[0:v][1:v]vstack=inputs=2",
        "-pix_fmt", "yuv420p", str(path),
    ])


def make_edl(spans: list[tuple[float, float]], *, positions: list[str] | None = None) -> EDL:
    """EDL из одного слова на блок: `spans` — окна слов в ВРЕМЕНИ ИСХОДНИКА."""
    words = [
        Word(idx=i, text="тест", start=a, end=b) for i, (a, b) in enumerate(spans)
    ]
    pos = positions or ["bottom"] * len(spans)
    return EDL(
        source_duration=spans[-1][1],
        words=words,
        cuts=[Cut(src_in=spans[0][0], src_out=spans[-1][1], reason="keep")],
        captions=[
            CaptionBlock(words=[CaptionWord(word_idx=i)], position=p)
            for i, p in enumerate(pos)
        ],
    )


def measure(path, edl: EDL) -> LumaReport:
    return measure_bg_luma(edl, path, preset=PRESET)


# ------------------------------------------------------------------ люма кадра


def test_frame_luma_uses_rgb_order_and_rec709():
    """Порядок каналов — RGB. Перепутанный не падает, а тихо врёт."""
    rect = Rect(0, 0, 8, 8)
    assert frame_luma(solid((255, 0, 0)), rect) == pytest.approx(0.2126, abs=1e-6)  # красный
    assert frame_luma(solid((0, 255, 0)), rect) == pytest.approx(0.7152, abs=1e-6)  # зелёный
    assert frame_luma(solid((0, 0, 255)), rect) == pytest.approx(0.0722, abs=1e-6)  # синий
    assert frame_luma(solid((255, 255, 255)), rect) == pytest.approx(1.0, abs=1e-6)
    assert frame_luma(solid((0, 0, 0)), rect) == pytest.approx(0.0, abs=1e-6)


def test_frame_luma_is_linear_light_not_srgb_code():
    """Единица — относительная яркость по WCAG: средний серый даёт 0.216, не 0.5.

    Число потребляет `contrastRatio` в Captions.tsx, а формула контраста по WCAG
    определена только на линейном свете.
    """
    rect = Rect(0, 0, 8, 8)
    assert frame_luma(solid((128, 128, 128)), rect) == pytest.approx(
        wcag_luminance((128, 128, 128)), abs=1e-6
    )
    assert frame_luma(solid((128, 128, 128)), rect) == pytest.approx(0.2159, abs=1e-3)


def test_frame_luma_counts_only_the_rect():
    """Половина кадра белая — среднее по прямоугольнику зависит от того, где он."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[50:, :] = 255
    assert frame_luma(frame, Rect(0, 0, 100, 50)) == pytest.approx(0.0, abs=1e-6)
    assert frame_luma(frame, Rect(0, 50, 100, 50)) == pytest.approx(1.0, abs=1e-6)
    assert frame_luma(frame, Rect(0, 0, 100, 100)) == pytest.approx(0.5, abs=1e-6)


def test_frame_luma_survives_rect_outside_frame():
    """Блок у кромки после округления вылезает за кадр — это не потеря замера."""
    frame = solid((255, 255, 255), (100, 100))
    assert frame_luma(frame, Rect(-20, 90, 200, 30)) == pytest.approx(1.0, abs=1e-6)
    assert frame_luma(frame, Rect(200, 200, 50, 50)) is None
    assert frame_luma(frame, Rect(10, 10, 0.2, 0.2)) is not None  # округляется вверх до пикселя
    assert frame_luma(frame, Rect(10, 10, 0, 0)) is None


# ------------------------------------------------------------------ по видео


def test_white_clip_reads_one(tmp_path):
    path = tmp_path / "white.mp4"
    clip_split_in_time(path, "white", "white")
    edl = make_edl([(0.0, 0.9), (1.1, 1.9)])
    rep = measure(path, edl)
    assert rep.measured == 2
    for block in edl.captions:
        assert block.bg_luma == pytest.approx(1.0, abs=TOL)


def test_black_clip_reads_zero(tmp_path):
    path = tmp_path / "black.mp4"
    clip_split_in_time(path, "black", "black")
    edl = make_edl([(0.0, 0.9), (1.1, 1.9)])
    rep = measure(path, edl)
    assert rep.measured == 2
    for block in edl.captions:
        assert block.bg_luma == pytest.approx(0.0, abs=TOL)


def test_each_block_gets_its_own_frame(tmp_path):
    """Первая половина чёрная, вторая белая — блоки обязаны разойтись."""
    path = tmp_path / "half.mp4"
    clip_split_in_time(path, "black", "white")
    edl = make_edl([(0.1, 0.8), (1.2, 1.9)])
    measure(path, edl)
    assert edl.captions[0].bg_luma == pytest.approx(0.0, abs=TOL)
    assert edl.captions[1].bg_luma == pytest.approx(1.0, abs=TOL)


def test_position_decides_what_is_measured(tmp_path):
    """Верх кадра чёрный, низ белый: замер идёт по caption_rect, а не по кадру."""
    path = tmp_path / "vstack.mp4"
    clip_split_in_space(path, "black", "white")
    edl = make_edl([(0.1, 0.4), (0.5, 0.9)], positions=["top", "bottom"])
    measure(path, edl)
    assert edl.captions[0].bg_luma == pytest.approx(0.0, abs=TOL)
    assert edl.captions[1].bg_luma == pytest.approx(1.0, abs=TOL)


def test_measures_output_timeline_not_source(tmp_path):
    """Видео уже нарезано: время блока — выходное, а не таймкод исходника.

    Слова стоят на 10-й секунде исходника, а клип длится две. По исходному
    времени оба блока упёрлись бы в конец файла и оба стали бы белыми.
    """
    path = tmp_path / "cut.mp4"
    clip_split_in_time(path, "black", "white")
    edl = make_edl([(10.1, 10.8), (11.2, 11.9)])
    edl.source_duration = 20.0
    measure(path, edl)
    assert edl.captions[0].bg_luma == pytest.approx(0.0, abs=TOL)
    assert edl.captions[1].bg_luma == pytest.approx(1.0, abs=TOL)


def test_grey_clip_is_a_fifth_not_a_half(tmp_path):
    """Средний серый — 0.216, а не 0.5: меряем линейный свет, а не sRGB-код."""
    path = tmp_path / "grey.mp4"
    clip_split_in_time(path, "gray", "gray")
    edl = make_edl([(0.2, 1.8)])
    measure(path, edl)
    assert edl.captions[0].bg_luma == pytest.approx(wcag_luminance((128, 128, 128)), abs=0.02)


def test_colour_clip_decodes_by_stream_tag(tmp_path):
    """Цветной кадр через настоящий декодер: и порядок каналов, и матрица.

    Клип помечен bt709, и рендерер (Chromium, как и ffmpeg) раскладывает его по
    тегу. `cv2.VideoCapture` тег игнорирует и считает всегда по bt601 — на этом
    цвете он дал бы 0.196 вместо 0.241, втрое дальше допуска. Перепутанный
    порядок каналов даёт 0.105. На белом, чёрном и сером ни то, ни другое не
    отличить: там R=G=B.
    """
    path = tmp_path / "orange.mp4"
    clip_colour(path, "0xFF4000")
    edl = make_edl([(0.2, 0.8)])
    measure(path, edl)
    assert edl.captions[0].bg_luma == pytest.approx(wcag_luminance((0xFF, 0x40, 0x00)), abs=0.02)


# ------------------------------------------------------------------ деградация


def test_missing_video_leaves_none(tmp_path):
    edl = make_edl([(0.0, 1.0)])
    rep = measure(tmp_path / "нет-такого.mp4", edl)
    assert rep.measured == 0
    assert edl.captions[0].bg_luma is None


def test_broken_video_leaves_none(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video at all")
    edl = make_edl([(0.0, 1.0)])
    rep = measure(path, edl)
    assert rep.measured == 0
    assert edl.captions[0].bg_luma is None


def test_old_value_is_dropped_not_kept(tmp_path):
    """EDL приходит извне (`process_from_edl`) и может нести число прошлого прогона.

    Снятое с другого кропа и другого пресета, оно не «замер похуже», а чужое
    число, которому рендерер поверит как измерению.
    """
    edl = make_edl([(0.0, 1.0)])
    edl.captions[0].bg_luma = 0.91
    rep = measure(tmp_path / "нет-такого.mp4", edl)
    assert rep.measured == 0
    assert edl.captions[0].bg_luma is None


def test_block_past_end_of_file_stays_none(tmp_path):
    """Длительность плана берётся из таймлайна EDL, а файл может кончиться раньше.

    Тогда блок остаётся без замера, а не получает последний кадр файла.
    """
    path = tmp_path / "half.mp4"
    clip_split_in_time(path, "black", "white")  # две секунды
    edl = make_edl([(0.1, 0.5), (3.0, 3.8)])
    edl.source_duration = 4.0
    edl.cuts[0].src_out = 4.0
    rep = measure(path, edl)
    assert rep.measured == 1
    assert edl.captions[0].bg_luma == pytest.approx(0.0, abs=TOL)
    assert edl.captions[1].bg_luma is None


def test_no_captions_is_not_an_error(tmp_path):
    path = tmp_path / "white.mp4"
    clip_split_in_time(path, "white", "white")
    edl = make_edl([(0.0, 1.9)])
    edl.captions = []
    rep = measure(path, edl)
    assert rep.blocks == 0 and rep.measured == 0 and rep.mean is None


def test_measurement_distributes_frames_by_block(tmp_path):
    """Кадров ровно по три на блок — распакованный кадр и есть цена этапа.

    Секундомер здесь не проверяется нарочно: на двухсекундном клипе почти всё
    время съедают два запуска процессов (ffprobe и ffmpeg), а не работа, и на
    загруженной машине такой порог ловит соседей по CPU, а не регрессию. Число
    из DoD («меньше секунды») снимается на 18-секундном эталоне прогоном
    пайплайна; верхняя граница тут — только сторож от зависания.
    """
    path = tmp_path / "white.mp4"
    clip_split_in_time(path, "white", "black")
    edl = make_edl([(0.1, 0.5), (0.6, 1.0), (1.1, 1.5), (1.6, 1.9)])
    rep = measure(path, edl)
    assert rep.measured == 4
    assert rep.frames == 4 * 3  # три кадра на блок, ни одним больше
    assert rep.seconds < 10.0
