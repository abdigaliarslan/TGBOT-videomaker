"""Прогон пайплайна на файле с выгрузкой всех артефактов (замер базы, Фаза 0).

Не тест, а инструмент: воркер удаляет рабочую папку и отдаёт только ролик, а для
сравнения фаз нужны ещё EDL, слова с таймингами, план рендера и тайминги этапов.
Здесь тот же `process_video`, но артефакты складываются рядом с выходом под
именами, на которые ссылается samples/README.md.

    python tests/run_pipeline.py samples/phase-0-source.mp4 --out samples \
        --prefix phase-0 --label baseline
    python tests/run_pipeline.py video.mp4 --out /tmp/run --preset bold --no-llm

Прогон недетерминированный: EDL приходит от LLM. Поэтому повтор даёт похожий, но
не тот же ролик — базовые артефакты хранятся в репозитории, а не пересобираются.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import settings  # noqa: E402
from src.core.edl import EDL  # noqa: E402
from src.core.logging import setup_logging  # noqa: E402
from src.llm.client import LLMClient  # noqa: E402
from src.pipeline.job import process_from_edl, process_video  # noqa: E402

# Что вынимаем из рабочей папки. Имя в выгрузке — <основа>-<label>.json: ролик
# именуется фазой (phase-0.mp4), а артефакты — тем, чем они являются для сравнения
# (edl-baseline.json), поэтому у них разные ключи в командной строке.
ARTIFACTS = ("edl", "words", "plan")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


async def run(args: argparse.Namespace) -> int:
    src = Path(args.video)
    if not src.exists():
        print(f"нет файла: {src}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    work = Path(args.work) if args.work else out_dir / f".work-{args.prefix}"
    if work.exists():
        shutil.rmtree(work)

    fixed_edl: EDL | None = None
    if args.edl:
        raw = json.loads(Path(args.edl).read_text(encoding="utf-8"))
        # Поля, которых больше нет в контракте (camera), молча выбрасываем:
        # эталонный EDL снят до их удаления, а сравнивать фазы надо на нём же.
        known = set(EDL.model_fields)
        fixed_edl = EDL(**{k: v for k, v in raw.items() if k in known})

    llm = None if args.no_llm or fixed_edl is not None else LLMClient()
    print(
        f"вход: {src}  "
        f"{'EDL из файла: ' + args.edl if fixed_edl else 'llm=' + ('выкл' if llm is None else str(llm.ready))}  "
        f"asr={settings.asr_backend}/{settings.whisper_model}  "
        f"рендер={args.backend or settings.render_backend}  "
        f"пресет={args.preset or settings.default_preset}  b-roll={settings.broll_enabled}",
        flush=True,
    )

    t0 = time.perf_counter()
    if fixed_edl is not None:
        res = await process_from_edl(
            src, work, fixed_edl, preset=args.preset, backend=args.backend
        )
    else:
        res = await process_video(
            src, work, preset=args.preset, llm=llm, backend=args.backend, keep_work=True
        )
    wall = round(time.perf_counter() - t0, 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{args.prefix}.mp4"
    shutil.copy2(res.output, video)
    for base in ARTIFACTS:
        artifact = work / f"{base}.json"
        if artifact.exists():
            shutil.copy2(artifact, out_dir / f"{base}-{args.label}.json")

    summary = res.summary()
    summary["wall_seconds"] = wall
    summary["source"] = {"path": str(src.resolve()), "sha256": sha256(src)}
    summary["output"] = str(video.resolve())
    (out_dir / f"{args.prefix}-run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)

    t = summary["timings"]
    print(f"готово за {wall} c: {video}")
    print("  этапы: " + ", ".join(f"{k} {v}" for k, v in t.items()))
    print(
        f"  слов {summary['words']}, блоков {summary['blocks']}, "
        f"вырезано кусков {summary['cuts_dropped']}, b-roll {summary['broll']}, "
        f"выход {summary['out_duration']} c"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="прогон пайплайна с выгрузкой артефактов")
    ap.add_argument("video", help="исходное видео")
    ap.add_argument("--out", default="samples", help="куда сложить артефакты (по умолч. samples)")
    ap.add_argument("--prefix", default="phase-0", help="имя ролика и сводки (по умолч. phase-0)")
    ap.add_argument("--label", default="baseline", help="суффикс edl/words/plan (по умолч. baseline)")
    ap.add_argument("--preset", default=None, help="пресет субтитров")
    ap.add_argument("--backend", default=None, choices=["ass", "remotion"], help="рендерер")
    ap.add_argument("--no-llm", action="store_true", help="без LLM: детерминированный режиссёр")
    ap.add_argument(
        "--edl", default=None,
        help="готовый EDL: без ASR и LLM, тот же монтаж — так сравниваются фазы",
    )
    ap.add_argument("--work", default=None, help="рабочая папка (по умолч. <out>/.work-<prefix>)")
    ap.add_argument("--keep-work", action="store_true", help="не удалять рабочую папку")
    ap.add_argument("--log", default="INFO", help="уровень логов")
    args = ap.parse_args()

    setup_logging(args.log)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
