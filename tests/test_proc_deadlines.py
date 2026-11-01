"""У каждого внешнего процесса обязан быть крайний срок.

Зависший ffmpeg или ffprobe — не теория: битый контейнер, сетевой диск,
подвисшее устройство. Процесс без срока держит задачу вечно, а `job_timeout`
воркера тут бессилен: arq отмеряет его через `asyncio.wait_for`, а тот не
умеет прерывать чужой процесс. Такая задача и становится той самой «висящей
в RUNNING», которую потом освобождает сторож, — то есть пользователь
дожидается не ролика, а возврата на баланс.

Проверка — настоящая: в PATH подкладывается зависающий двойник, и вызов
обязан вернуться сам.
"""

from __future__ import annotations

import subprocess
import time

import pytest

# `exec`, а не просто `sleep`: иначе оболочка остаётся отдельным процессом, и
# убитым оказывается она, а не тот, кто держит открытым stdout. Настоящий ffmpeg
# детей не плодит, так что двойник обязан вести себя так же.
HANG = "#!/bin/sh\nexec sleep 30\n"


@pytest.fixture
def hanging(tmp_path, monkeypatch):
    """Подкладывает в PATH зависающие ffmpeg и ffprobe."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        exe = bindir / name
        exe.write_text(HANG, encoding="utf-8")
        exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ['PATH']}")
    media = tmp_path / "src.mp4"
    media.write_bytes(b"\0" * 1024)
    return media


def test_probe_gives_up_on_a_hanging_ffprobe(hanging, monkeypatch):
    """ffprobe зовут из бота, из приложения и из прогона. Без срока зависший
    вызов съедает поток там, куда пользователь смотрит прямо сейчас."""
    from src.pipeline import probe as probe_mod

    monkeypatch.setattr(probe_mod, "PROBE_TIMEOUT", 0.5)
    t = time.perf_counter()
    with pytest.raises(subprocess.TimeoutExpired):
        probe_mod.probe(hanging)
    assert time.perf_counter() - t < 5


def test_a_file_that_cannot_be_read_in_time_is_not_playable(hanging, monkeypatch):
    """`is_playable` — вопрос «цел ли файл», и зависший ffprobe на него уже
    ответил. Пробрасывать наружу тайм-аут значило бы обойти ветку с внятной
    ошибкой про оборванную нарезку."""
    from src.pipeline import probe as probe_mod

    monkeypatch.setattr(probe_mod, "PROBE_TIMEOUT", 0.5)
    assert probe_mod.is_playable(hanging) is False


def test_the_voice_for_the_corrector_is_not_worth_a_hung_job(hanging, monkeypatch):
    """Голос для корректора — необязательный шаг: не вышло, правим по тексту.
    Поэтому срок есть, а его истечение — предупреждение, не падение задачи."""
    from src.pipeline import director

    monkeypatch.setattr(director, "VOICE_CLIP_TIMEOUT", 0.5)
    t = time.perf_counter()
    assert director.voice_clip(hanging) is None
    assert time.perf_counter() - t < 5


def test_the_luma_measurement_gives_up_instead_of_hanging(hanging, monkeypatch, tmp_path):
    """Замер яркости читает кадры потоком, поэтому срок стоит сторожем на весь
    проход: `read` на зависшем процессе не возвращается никогда."""
    from src.pipeline import luma

    monkeypatch.setattr(luma, "DECODE_TIMEOUT", 0.5)
    t = time.perf_counter()
    frames = list(luma._decode(hanging, [0, 1], 16, 16, luma.default_settings))
    assert frames == []
    assert time.perf_counter() - t < 5


def test_every_outside_process_in_the_pipeline_has_a_deadline():
    """Сторож от возврата старой беды: новый вызов без срока виден сразу.

    Текстовая проверка намеренно: срок — свойство строки вызова, и ловить его
    на живом ffmpeg значит гонять минуты ради одного ключевого слова.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src"
    # `ffmpeg()` и `affmpeg()` запускают процесс всегда. `run`, `arun`, `capture`
    # и `Popen` — только когда в вызове назван бинарник: именем `run` зовут и
    # местные корутины, и упоминают его в комментариях.
    always = ("ffmpeg(", "affmpeg(")
    binaries = ('"ffmpeg"', '"ffprobe"', '"node"', "_ffmpeg_cmd")
    naked: list[str] = []
    for path in sorted(root.rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(r"(?<!def )(?<![\w.])(a?ffmpeg|a?run|capture)\(|Popen\(", src):
            # Вызов бывает многострочным: берём его целиком по балансу скобок.
            depth, end = 0, m.end() - 1
            for i in range(m.end() - 1, len(src)):
                depth += (src[i] == "(") - (src[i] == ")")
                if depth == 0:
                    end = i
                    break
            call = src[m.start(): end + 1]
            # Потоковый запуск срок внутри вызова не несёт — у него будильник
            # (`core.proc.deadline`) сразу за `Popen`.
            guarded = "Popen(" in call and "deadline(" in src[end: end + 300]
            spawns = (
                "Popen(" in call
                or call.startswith(always)
                or any(b in call for b in binaries)
            )
            if not spawns or "timeout" in call or guarded:
                continue
            naked.append(f"{path.relative_to(root)}: {call.splitlines()[0]}")
    assert not naked, "внешний процесс без крайнего срока:\n" + "\n".join(naked)
