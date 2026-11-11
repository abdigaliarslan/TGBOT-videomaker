"""Мост Python → рендерер.

Два бэкенда за одним интерфейсом:

- `remotion` — нормальный рендер (M5): React-анимации, пружины, glass-панель;
- `ass` — быстрый путь (M3): ffmpeg + libass, без панели и b-roll, зато без Node.

Взаимодействие с Node — через subprocess с JSON на stdin и путём к файлу на
stdout. Никакого HTTP-сервиса между ними (§1).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from src.core.config import Settings, settings as default_settings
from src.core.proc import ProcError, arun, run
from src.core.render_plan import RenderPlan

log = logging.getLogger(__name__)

RENDER_DIR = Path(__file__).resolve().parents[2] / "render"
RENDER_CLI = RENDER_DIR / "render.mjs"


class RenderError(RuntimeError):
    pass


@dataclass(slots=True)
class RenderRequest:
    plan: RenderPlan
    out: Path
    assets_dir: Path
    job_key: str
    crf: int = 18
    concurrency: int | None = None
    x264_preset: str = "medium"
    gl: str | None = None


def _stage_asset(src: str | Path, assets_dir: Path) -> str:
    """Кладёт файл в директорию джоба и возвращает имя относительно неё.

    Node монтирует директорию в бандл одним симлинком, поэтому все ассеты
    должны лежать в ней, а в плане быть относительными именами.
    """
    src = Path(src)
    assets_dir.mkdir(parents=True, exist_ok=True)
    if src.parent.resolve() == assets_dir.resolve():
        return src.name
    dst = assets_dir / src.name
    if not dst.exists():
        try:
            os.link(src, dst)  # жёсткая ссылка: без копирования гигабайтов
        except OSError:
            shutil.copy2(src, dst)
    return dst.name


def prepare_assets(
    plan: RenderPlan, assets_dir: Path, *, cfg: Settings | None = None
) -> RenderPlan:
    """Переносит видео, звук, картинки и шрифт в директорию джоба."""
    cfg = cfg or default_settings
    assets_dir.mkdir(parents=True, exist_ok=True)

    plan.video = _stage_asset(plan.video, assets_dir)
    if plan.audio:
        plan.audio = _stage_asset(plan.audio, assets_dir)
    for item in plan.broll:
        if item.asset:
            item.asset = _stage_asset(item.asset, assets_dir)

    font_file = (plan.preset.get("font") or {}).get("file")
    if font_file:
        candidate = cfg.fonts_dir / font_file
        if candidate.exists():
            plan.meta["font"] = _stage_asset(candidate, assets_dir)
        else:
            log.warning("шрифт %s не найден в %s — Chromium возьмёт системный", font_file, cfg.fonts_dir)
    return plan


def _request_payload(req: RenderRequest) -> str:
    payload = {
        "plan": {**req.plan.to_dict(), "fontUrl": req.plan.meta.get("font")},
        "assetsDir": str(req.assets_dir.resolve()),
        "out": str(req.out.resolve()),
        "jobKey": req.job_key,
        "crf": req.crf,
        "x264Preset": req.x264_preset,
    }
    if req.concurrency:
        payload["concurrency"] = req.concurrency
    if req.gl:
        payload["gl"] = req.gl
    return json.dumps(payload, ensure_ascii=False)


def ensure_browser() -> None:
    """Скачать Chromium заранее — на старте воркера, а не на каждый рендер (M5 ⚠)."""
    try:
        run(["node", str(RENDER_CLI), "--ensure-browser"], timeout=1800)
        log.info("браузер рендера готов")
    except ProcError as e:
        raise RenderError(f"не удалось подготовить браузер: {e}") from e


def ensure_bundle() -> None:
    try:
        run(["node", str(RENDER_CLI), "--bundle"], timeout=1800)
    except ProcError as e:
        raise RenderError(f"не удалось собрать бандл рендера: {e}") from e


def render_remotion(
    plan: RenderPlan,
    out: str | Path,
    *,
    assets_dir: str | Path,
    job_key: str = "job",
    cfg: Settings | None = None,
    crf: int | None = None,
) -> Path:
    cfg = cfg or default_settings
    crf = cfg.render_crf if crf is None else crf
    if not RENDER_CLI.exists():
        raise RenderError(f"нет {RENDER_CLI}: проект рендера не установлен")
    if not (RENDER_DIR / "node_modules").exists():
        raise RenderError(f"в {RENDER_DIR} не выполнен npm install")

    assets_dir = Path(assets_dir)
    plan = prepare_assets(plan, assets_dir, cfg=cfg)
    req = RenderRequest(
        plan=plan,
        out=Path(out),
        assets_dir=assets_dir,
        job_key=job_key,
        crf=crf,
        concurrency=cfg.render_concurrency,
        x264_preset=cfg.render_x264_preset,
        gl=cfg.render_gl or None,
    )
    try:
        stdout = run(["node", str(RENDER_CLI)], stdin=_request_payload(req), timeout=10_800)
    except ProcError as e:
        raise RenderError(f"Remotion не отрендерил: {e}") from e

    produced = Path(stdout.strip().splitlines()[-1]) if stdout.strip() else req.out
    if not produced.exists():
        raise RenderError(f"рендер завершился, но файла нет: {produced}")
    return produced


_PROGRESS = re.compile(r"\[render\] progress (\d+)")

# Срок самому рендеру. Обязан быть МЕНЬШЕ `job_timeout` воркера
# (`worker.tasks.JOB_TIMEOUT`, три часа): тот отсчитывается от начала всей задачи,
# и при равных сроках первым всегда срабатывал он — то есть отменой, а не этим
# сроком. Трёхминутный ролик рендерится около получаса, запас четырёхкратный.
RENDER_TIMEOUT = 2 * 3600

# Рендер — это Node и порождённые им процессы Chromium. На POSIX он уходит в свою
# группу, чтобы снять можно было всех разом: `proc.kill()` убивает один Node, а
# браузер остался бы сиротой. В Windows групп нет — там убивается сам процесс.
_OWN_GROUP: dict[str, bool] = {"start_new_session": True} if os.name == "posix" else {}


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Снять рендер вместе с его потомками. Процесса уже нет — не ошибка."""
    with suppress(ProcessLookupError, PermissionError, OSError):
        if _OWN_GROUP:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()


async def _run_node_with_progress(
    payload: str, *, timeout: float, on_progress: Callable[[float], Awaitable[None] | None] | None
) -> str:
    """Запуск рендера с чтением stderr по строкам, а не одним куском в конце.

    `arun` отдаёт вывод только после выхода процесса, а рендер — самый долгий
    этап: на 30-секундном ролике он шёл четыре минуты, и всё это время полоса
    прогресса стояла на месте. Пользователь видел «завис на 97 %».
    """
    cmd = ["node", str(RENDER_CLI)]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **_OWN_GROUP,
    )
    tail: list[str] = []

    async def pump_stderr() -> None:
        assert proc.stderr is not None
        async for raw in proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            tail.append(line)
            del tail[:-200]  # в ошибку уходит хвост, а не весь лог рендера
            m = _PROGRESS.search(line)
            if m and on_progress is not None:
                try:
                    res = on_progress(min(1.0, int(m.group(1)) / 100))
                    if hasattr(res, "__await__"):
                        await res  # type: ignore[misc]
                except Exception as e:  # noqa: BLE001 — прогресс не должен ронять рендер
                    log.warning("прогресс рендера не записан: %s", e)

    async def run() -> bytes:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(payload.encode())
        await proc.stdin.drain()
        proc.stdin.close()
        out, _ = await asyncio.gather(proc.stdout.read(), pump_stderr())
        await proc.wait()
        return out

    try:
        out = await asyncio.wait_for(run(), timeout=timeout)
    finally:
        # Убиваем на ЛЮБОМ выходе, пока процесс жив, а не только по сроку. Отмена
        # задачи (выкладка перезапустила воркер, вышел `job_timeout` arq) — это
        # `CancelledError`, и раньше он проходил мимо: Node с Chromium оставались
        # рендерить в никуда и делили ядра со следующей попыткой той же задачи.
        if proc.returncode is None:
            _kill_tree(proc)
            with suppress(Exception):
                await proc.wait()
    if proc.returncode != 0:
        raise ProcError(cmd, proc.returncode or -1, "\n".join(tail))
    return out.decode("utf-8", "replace")


async def arender_remotion(
    plan: RenderPlan,
    out: str | Path,
    *,
    assets_dir: str | Path,
    job_key: str = "job",
    cfg: Settings | None = None,
    crf: int | None = None,
    on_progress: Callable[[float], Awaitable[None] | None] | None = None,
) -> Path:
    """Асинхронный вариант: воркер не должен блокировать loop на рендере.

    `on_progress(доля 0..1)` вызывается по мере рендера кадров.
    """
    cfg = cfg or default_settings
    crf = cfg.render_crf if crf is None else crf
    assets_dir = Path(assets_dir)
    plan = prepare_assets(plan, assets_dir, cfg=cfg)
    req = RenderRequest(
        plan=plan, out=Path(out), assets_dir=assets_dir, job_key=job_key,
        crf=crf, concurrency=cfg.render_concurrency,
        x264_preset=cfg.render_x264_preset, gl=cfg.render_gl or None,
    )
    try:
        stdout = await _run_node_with_progress(
            _request_payload(req), timeout=RENDER_TIMEOUT, on_progress=on_progress
        )
    except ProcError as e:
        raise RenderError(f"Remotion не отрендерил: {e}") from e
    except asyncio.TimeoutError as e:
        raise RenderError(f"Remotion не уложился в {RENDER_TIMEOUT // 60} мин") from e
    produced = Path(stdout.strip().splitlines()[-1]) if stdout.strip() else req.out
    if not produced.exists():
        raise RenderError(f"рендер завершился, но файла нет: {produced}")
    return produced


def render_ass(
    plan: RenderPlan,
    out: str | Path,
    *,
    work_dir: str | Path,
    cfg: Settings | None = None,
) -> Path:
    from src.pipeline.subs_ass import burn, write_ass

    cfg = cfg or default_settings
    work_dir = Path(work_dir)
    ass_path = write_ass(plan, work_dir / "captions.ass")
    return burn(plan, ass_path, out, fonts_dir=cfg.fonts_dir)


def render(
    plan: RenderPlan,
    out: str | Path,
    *,
    work_dir: str | Path,
    backend: str | None = None,
    job_key: str = "job",
    cfg: Settings | None = None,
) -> Path:
    """Единый вход. Бэкенд берётся из конфига, если не указан явно."""
    cfg = cfg or default_settings
    chosen = backend or cfg.render_backend
    if chosen == "ass":
        return render_ass(plan, out, work_dir=work_dir, cfg=cfg)
    return render_remotion(plan, out, assets_dir=work_dir, job_key=job_key, cfg=cfg)


if __name__ == "__main__":  # python -m src.pipeline.render plan.json out.mp4
    import sys

    from src.core.logging import setup_logging

    setup_logging(default_settings.log_level)
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    dest = Path(sys.argv[2])
    print(render(RenderPlan.from_dict(data), dest, work_dir=dest.parent / "work"))
