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


@pytest.fixture
def hanging_node(tmp_path, monkeypatch):
    """Зависающий `node`, который успевает сказать, под каким номером он запущен."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "node"
    exe.write_text('#!/bin/sh\necho $$ > "$NODE_PIDFILE"\nexec sleep 30\n', encoding="utf-8")
    exe.chmod(0o755)
    pidfile = tmp_path / "node.pid"
    monkeypatch.setenv("NODE_PIDFILE", str(pidfile))
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ['PATH']}")
    return pidfile


async def _started(pidfile) -> int:
    import asyncio

    for _ in range(200):
        if pidfile.exists() and pidfile.read_text().strip():
            return int(pidfile.read_text())
        await asyncio.sleep(0.05)
    pytest.fail("двойник node не запустился")


def _alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_a_cancelled_render_takes_its_node_down_with_it(hanging_node):
    """Отмена задачи обязана снять и рендер.

    Выкладка перезапускает воркер посреди ролика, arq отменяет задачу — это
    `CancelledError`, а убивался Node только по истечении срока. Он оставался
    рендерить в никуда и делил ядра со следующей попыткой той же задачи.
    """
    import asyncio

    from src.pipeline import render

    task = asyncio.create_task(render._run_node_with_progress("{}", timeout=60, on_progress=None))
    pid = await _started(hanging_node)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not _alive(pid), "рендер пережил отмену своей задачи"


async def test_a_render_past_its_deadline_is_killed(hanging_node):
    """Прежнее поведение на месте: срок вышел — процесса нет."""
    import asyncio

    from src.pipeline import render

    with pytest.raises(asyncio.TimeoutError):
        await render._run_node_with_progress("{}", timeout=0.5, on_progress=None)
    assert not _alive(await _started(hanging_node)), "рендер пережил свой срок"


def test_the_render_deadline_fires_before_the_job_deadline():
    """Срок рендера обязан быть меньше срока задачи: при равных первым всегда
    срабатывал срок задачи, то есть отмена, и ветка «рендер не уложился» была
    недостижима."""
    from src.pipeline.render import RENDER_TIMEOUT
    from src.worker.tasks import JOB_TIMEOUT

    assert RENDER_TIMEOUT < JOB_TIMEOUT


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
