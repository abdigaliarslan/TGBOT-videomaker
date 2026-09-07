"""Проверка таймингов words.json: монотонность, индексы, паузы (замер базы, Фаза 0).

Не тест, а инструмент: печатает, насколько тайминги ASR пригодны для пословной
анимации. Если w[i].end > w[i+1].start, субтитр подсвечивает два слова разом или
уезжает на слово вперёд — глазом это видно как дрожание, а числом не измерено.
Главный инвариант (и главное число DoD) — количество немонотонных пар.

Вывод машиночитаемый (--json), пороги зашиты константами и не зависят от данных,
поэтому отчёты разных прогонов сравнимы между собой на следующих фазах.

    python tests/words_monotonic.py data/work/<job>/words.json
    python tests/words_monotonic.py words.json --json /tmp/words_report.json --limit 20
    python tests/words_monotonic.py words.json --strict   # код возврата 1 при нарушениях

Коды возврата: 0 — чисто (или нарушения есть, но без --strict), 1 — --strict и есть
нарушения инвариантов 1-4, 2 — файл не читается или не похож на words.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.edl import Word  # noqa: E402
from src.pipeline.asr import words_from_json  # noqa: E402

# Тайминги в words.json округлены до миллисекунды (см. asr._finalize), поэтому
# расхождение мельче половины миллисекунды — это шум float, а не болезнь данных.
# Порог вынесен в отчёт, чтобы числа разных прогонов оставались сопоставимыми.
TOL_MS = 0.5

# Слово короче 40 мс физически не произносится: это либо артефакт выравнивания,
# либо слово, разорванное пополам. Подсветить его в кадре всё равно нельзя.
SHORT_WORD_MS = 40.0
# Слово длиннее 3 с — почти наверняка склейка со следующей паузой: ASR растянул
# границу до конца сегмента. Для пословной анимации это мёртвое время.
LONG_WORD_S = 3.0

# Инварианты, нарушение которых ломает пайплайн (а не просто портит вид).
FATAL_CHECKS = ("monotonic", "starts_nondecreasing", "positive_duration", "idx_sequential")


@dataclass(slots=True)
class Check:
    """Один инвариант: счётчик нарушений плюс все примеры для последующего среза."""

    key: str
    title: str
    sort_key: str = "delta_ms"
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def violations(self) -> int:
        return len(self.examples)

    def worst(self, limit: int) -> list[dict[str, Any]]:
        """Самые грубые нарушения — по величине отклонения, при равенстве по порядку."""
        ranked = sorted(self.examples, key=lambda e: (-float(e.get(self.sort_key, 0)), e["i"]))
        return ranked[: max(0, limit)]

    def to_dict(self, limit: int) -> dict[str, Any]:
        return {"title": self.title, "violations": self.violations, "worst": self.worst(limit)}


def _ms(value: float) -> float:
    """Округление до десятой миллисекунды. -0.0 приводим к 0.0: иначе в отчёте «-0 мс»."""
    r = round(value, 1)
    return r + 0.0 if r == 0 else r


def _brief(w: Word) -> dict[str, Any]:
    return {"idx": w.idx, "text": w.text, "start": w.start, "end": w.end}


def _fmt_word(w: Word) -> str:
    return f"#{w.idx} {w.text!r} {w.start:.3f}..{w.end:.3f}"


def _percentile(values: list[float], q: float) -> float:
    """Линейная интерполяция по отсортированному списку (как numpy.percentile)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def load_words(path: Path) -> tuple[list[Word], dict[str, Any]]:
    """words.json → слова и метаданные прогона.

    Поддержаны оба вида: полный {"words": [...], "backend": ...} из ASRResult.to_dict()
    и голый список слов — второй кладут в отчёты вручную, и запрещать его незачем.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "words" not in payload:
            # Частый случай — подсунули эталон фразовых границ (speech.truth.json)
            # или EDL; лучше сказать это прямо, чем уронить KeyError.
            raise ValueError(f"в объекте нет ключа 'words' (есть: {sorted(payload)[:8]})")
        # source_duration — чтобы инструмент принимал и edl.json: там те же слова,
        # а во время отладки под рукой чаще оказывается именно он.
        meta = {
            k: payload[k]
            for k in ("language", "backend", "model", "duration", "source_duration",
                      "dropped_segments")
            if k in payload
        }
    elif isinstance(payload, list):
        meta = {}
    else:
        raise ValueError(f"ожидался объект или список, получен {type(payload).__name__}")
    return words_from_json(payload), meta


def analyze(words: list[Word], meta: dict[str, Any], *, limit: int) -> dict[str, Any]:
    """Все инварианты за один проход. Возвращает готовый к сериализации отчёт."""
    checks = {
        "monotonic": Check("monotonic", "монотонность: end[i] <= start[i+1]"),
        "starts_nondecreasing": Check("starts_nondecreasing", "неубывание стартов"),
        "positive_duration": Check("positive_duration", "положительная длительность"),
        "idx_sequential": Check("idx_sequential", "idx == позиция в списке", sort_key="delta_idx"),
        "short_words": Check("short_words", f"слишком короткие (< {SHORT_WORD_MS:.0f} мс)"),
        "long_words": Check("long_words", f"слишком длинные (> {LONG_WORD_S:.1f} с)"),
        "negative_time": Check("negative_time", "отрицательные тайминги"),
    }

    for i, w in enumerate(words):
        if w.idx != i:
            checks["idx_sequential"].examples.append(
                {"i": i, "word": _brief(w), "expected": i, "got": w.idx,
                 "delta_idx": abs(w.idx - i)}
            )
        dur_ms = (w.end - w.start) * 1000.0
        # Пороги сравниваем с допуском: 13.950 - 13.910 в double даёт 39.99 мс, и без
        # него слово ровно на границе то попадает в отчёт, то нет — числа перестают
        # быть сравнимыми между прогонами.
        if dur_ms <= TOL_MS:
            # Вырожденное слово: подсвечивать нечего, а если end < start — ещё и
            # ломается любая раскладка по таймлайну.
            checks["positive_duration"].examples.append(
                {"i": i, "word": _brief(w), "duration_ms": _ms(dur_ms), "delta_ms": _ms(-dur_ms)}
            )
        elif dur_ms < SHORT_WORD_MS - TOL_MS:
            checks["short_words"].examples.append(
                {"i": i, "word": _brief(w), "duration_ms": _ms(dur_ms),
                 "delta_ms": _ms(SHORT_WORD_MS - dur_ms)}
            )
        elif dur_ms > LONG_WORD_S * 1000.0 + TOL_MS:
            checks["long_words"].examples.append(
                {"i": i, "word": _brief(w), "duration_ms": _ms(dur_ms),
                 "delta_ms": _ms(dur_ms - LONG_WORD_S * 1000.0)}
            )
        if w.start < 0 or w.end < 0:
            checks["negative_time"].examples.append(
                {"i": i, "word": _brief(w), "delta_ms": _ms(-min(w.start, w.end) * 1000.0)}
            )

    gaps_ms: list[float] = []
    for i, (a, b) in enumerate(zip(words, words[1:])):
        overlap_ms = (a.end - b.start) * 1000.0
        gaps_ms.append(_ms(-overlap_ms))
        if overlap_ms > TOL_MS:
            # Соседи наложились: подсветка одного слова захватывает следующее.
            checks["monotonic"].examples.append(
                {"i": i, "prev": _brief(a), "next": _brief(b), "delta_ms": _ms(overlap_ms)}
            )
        back_ms = (a.start - b.start) * 1000.0
        if back_ms > TOL_MS:
            # Отдельная болезнь: не перекрытие соседей, а настоящая перестановка
            # слов — тогда врёт и порядок субтитров, и ссылки режиссёра по word_idx.
            checks["starts_nondecreasing"].examples.append(
                {"i": i, "prev": _brief(a), "next": _brief(b), "delta_ms": _ms(back_ms)}
            )

    first_start = words[0].start if words else 0.0
    last_end = max((w.end for w in words), default=0.0)
    span_s = max(0.0, last_end - first_start)
    speech_s = sum(max(0.0, w.end - w.start) for w in words)
    source_s = float(meta.get("duration") or meta.get("source_duration") or 0.0)

    max_gap = max(gaps_ms) if gaps_ms else 0.0
    gap_at = gaps_ms.index(max_gap) if gaps_ms else -1
    gaps = {
        "count": len(gaps_ms),
        "median_ms": _ms(statistics.median(gaps_ms)) if gaps_ms else 0.0,
        "p90_ms": _ms(_percentile(gaps_ms, 0.90)) if gaps_ms else 0.0,
        "max_ms": _ms(max_gap),
        "min_ms": _ms(min(gaps_ms)) if gaps_ms else 0.0,
        "negative": sum(1 for g in gaps_ms if g < -TOL_MS),
        "max_at": (
            {"i": gap_at, "prev": _brief(words[gap_at]), "next": _brief(words[gap_at + 1])}
            if gap_at >= 0
            else None
        ),
    }

    fatal = sum(checks[k].violations for k in FATAL_CHECKS)
    return {
        "words": len(words),
        # Главное число DoD — вынесено наверх, чтобы фазы сравнивались одним grep.
        "non_monotonic_pairs": checks["monotonic"].violations,
        "fatal_violations": fatal,
        "ok": fatal == 0,
        "source": meta,
        "summary": {
            "words": len(words),
            "first_start": round(first_start, 3),
            "last_end": round(last_end, 3),
            "covered_s": round(span_s, 3),
            "speech_s": round(speech_s, 3),
            "silence_s": round(max(0.0, span_s - speech_s), 3),
            "speech_ratio_of_span": round(speech_s / span_s, 4) if span_s > 0 else None,
            "speech_ratio_of_source": round(speech_s / source_s, 4) if source_s > 0 else None,
            "source_duration": round(source_s, 3) if source_s > 0 else None,
        },
        "gaps": gaps,
        "checks": {k: c.to_dict(limit) for k, c in checks.items()},
        "thresholds": {
            "tolerance_ms": TOL_MS,
            "short_word_ms": SHORT_WORD_MS,
            "long_word_s": LONG_WORD_S,
            "fatal_checks": list(FATAL_CHECKS),
        },
    }


def _print_report(path: Path, report: dict[str, Any], *, limit: int) -> None:
    s = report["summary"]
    src = report["source"]
    print(f"файл: {path}")
    origin = "/".join(str(src[k]) for k in ("backend", "model", "language") if src.get(k))
    bits = [f"прогон: {origin}"] if origin else []
    if s["source_duration"]:
        bits.append(f"исходник: {s['source_duration']:.3f} с")
    if bits:
        print("   ".join(bits))
    print(
        f"слов: {s['words']}   покрыто: {s['first_start']:.3f}..{s['last_end']:.3f} "
        f"({s['covered_s']:.3f} с)   речь: {s['speech_s']:.3f} с"
    )
    ratio_span = s["speech_ratio_of_span"]
    ratio_src = s["speech_ratio_of_source"]
    parts = []
    if ratio_span is not None:
        parts.append(f"{ratio_span * 100:.1f}% интервала слов")
    if ratio_src is not None:
        parts.append(f"{ratio_src * 100:.1f}% исходника")
    if parts:
        print("доля времени под речью: " + ", ".join(parts))

    g = report["gaps"]
    print(
        f"паузы между словами ({g['count']}): медиана {g['median_ms']:.0f} мс, "
        f"p90 {g['p90_ms']:.0f} мс, максимум {g['max_ms']:.0f} мс, минимум {g['min_ms']:.0f} мс"
    )
    print()

    for key, chk in report["checks"].items():
        mark = "ok " if chk["violations"] == 0 else "!! "
        fatal = " (критично)" if key in FATAL_CHECKS and chk["violations"] else ""
        print(f"  [{mark}] {chk['title']}: {chk['violations']}{fatal}")
        for ex in chk["worst"]:
            if "prev" in ex:
                a, b = ex["prev"], ex["next"]
                print(
                    f"        #{a['idx']} {a['text']!r} {a['start']:.3f}..{a['end']:.3f} -> "
                    f"#{b['idx']} {b['text']!r} {b['start']:.3f}..{b['end']:.3f}"
                    f"   на {ex['delta_ms']:.1f} мс"
                )
            elif key == "idx_sequential":
                w = ex["word"]
                print(f"        позиция {ex['i']}: idx={ex['got']} {w['text']!r} (ждали {ex['i']})")
            else:
                w = ex["word"]
                tail = f"   длительность {ex['duration_ms']:.1f} мс" if "duration_ms" in ex else ""
                print(
                    f"        #{w['idx']} {w['text']!r} {w['start']:.3f}..{w['end']:.3f}{tail}"
                )
        if chk["violations"] > limit:
            print(f"        ... ещё {chk['violations'] - limit} (показаны худшие {limit})")
    print()
    print(f"немонотонных пар: {report['non_monotonic_pairs']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Проверка инвариантов таймингов words.json")
    ap.add_argument("path", type=Path, help="путь к words.json")
    ap.add_argument("--json", dest="json_out", type=Path, help="куда записать полный отчёт")
    ap.add_argument("--limit", type=int, default=10, help="сколько худших примеров показывать")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="код возврата 1, если нарушены инварианты 1-4 (для CI)",
    )
    args = ap.parse_args()

    path: Path = args.path
    if not path.exists():
        print(f"файл не найден: {path}", file=sys.stderr)
        return 2
    try:
        words, meta = load_words(path)
    except Exception as e:  # noqa: BLE001 — любой разбор сломанного файла это диагноз
        print(f"не похоже на words.json ({type(e).__name__}): {e}", file=sys.stderr)
        return 2

    report = analyze(words, meta, limit=args.limit)
    _print_report(path, report, limit=args.limit)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"отчёт: {args.json_out}")

    if args.strict and report["fatal_violations"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
