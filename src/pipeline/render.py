"""Мост Python → рендерер.

Два бэкенда за одним интерфейсом:

- `remotion` — нормальный рендер (M5): React-анимации, пружины, glass-панель;
- `ass` — быстрый путь (M3): ffmpeg + libass, без панели и b-roll, зато без Node.

Взаимодействие с Node — через subprocess с JSON на stdin и путём к файлу на
stdout. Никакого HTTP-сервиса между ними (§1).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

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
    crf: int = 18,
) -> Path:
    cfg = cfg or default_settings
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
    )
    try:
        stdout = run(["node", str(RENDER_CLI)], stdin=_request_payload(req), timeout=10_800)
    except ProcError as e:
        raise RenderError(f"Remotion не отрендерил: {e}") from e

    produced = Path(stdout.strip().splitlines()[-1]) if stdout.strip() else req.out
    if not produced.exists():
        raise RenderError(f"рендер завершился, но файла нет: {produced}")
    return produced


async def arender_remotion(
    plan: RenderPlan,
    out: str | Path,
    *,
    assets_dir: str | Path,
    job_key: str = "job",
    cfg: Settings | None = None,
    crf: int = 18,
) -> Path:
    """Асинхронный вариант: воркер не должен блокировать loop на рендере."""
    cfg = cfg or default_settings
    assets_dir = Path(assets_dir)
    plan = prepare_assets(plan, assets_dir, cfg=cfg)
    req = RenderRequest(
        plan=plan, out=Path(out), assets_dir=assets_dir, job_key=job_key,
        crf=crf, concurrency=cfg.render_concurrency,
    )
    try:
        stdout = await arun(["node", str(RENDER_CLI)], stdin=_request_payload(req), timeout=10_800)
    except ProcError as e:
        raise RenderError(f"Remotion не отрендерил: {e}") from e
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
