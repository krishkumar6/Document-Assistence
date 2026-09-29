"""Compare the chunk configs on a labelled question set.

Retrieval metrics (free, offline):
    python -m eval.compare_chunks
End-to-end with Claude (answer correctness + citation page accuracy; calls the API):
    python -m eval.compare_chunks --llm

A retrieved chunk counts as relevant if it comes from the gold file and
contains text from the gold page.
"""

import argparse
import json
import re
import statistics
import time
import unicodedata
from pathlib import Path

from app import store
from app.config import CHUNK_CONFIGS, DOCS_DIR
from app.ingest import ingest_paths

HERE = Path(__file__).resolve().parent


def relevant(hit: store.Hit, q: dict) -> bool:
    return hit.source == q["file"] and any(s.page == q["page"] for s in hit.spans)


def _norm(s: str) -> str:
    # Models emit narrow no-break spaces ("18 months") and Unicode hyphens
    # ("three‑week"); fold them so grading compares content, not typography.
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[‐-―-]", " ", s)
    return re.sub(r"\s+", " ", s)


def graded(answer: str, expect: list[list[str]]) -> bool:
    """Every group must match; any alternative within a group is fine."""
    text = _norm(answer)
    # Whole-token match, so "24 hours" does not satisfy an expected "4 hours".
    return all(
        any(re.search(rf"(?<![\w.]){re.escape(_norm(alt))}(?!\w)", text) for alt in group) for group in expect
    )


def _answer_with_patience(answer_question, question, hits, attempts: int = 6):
    """Free tiers cap tokens per minute; wait out the window instead of failing the run."""
    from app.answer import LLMError

    for attempt in range(attempts):
        try:
            return answer_question(question, hits)
        except LLMError as e:
            if e.status != 429 or attempt == attempts - 1:
                raise
            print(f"  rate limited, waiting 30s ({attempt + 1}/{attempts - 1})")
            time.sleep(30)


def run(k: int, use_llm: bool, reingest: bool) -> dict:
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    report = {}

    for name, cfg in CHUNK_CONFIGS.items():
        t0 = time.perf_counter()
        if reingest:
            ingest_paths(pdfs, [cfg])
        ingest_s = time.perf_counter() - t0

        rows, latencies = [], []
        for q in questions:
            t = time.perf_counter()
            hits = store.search(cfg, q["question"], k)
            latencies.append(time.perf_counter() - t)
            ranks = [i for i, h in enumerate(hits, 1) if relevant(h, q)]
            rows.append({
                "question": q["question"],
                "hit@1": bool(ranks) and ranks[0] == 1,
                f"hit@{k}": bool(ranks),
                "rr": 1 / ranks[0] if ranks else 0.0,
                # Share of retrieved text that is actually relevant: large chunks
                # find the page more easily but drag in more unrelated text.
                "precision": sum(len(h.text) for h in hits if relevant(h, q)) / max(1, sum(len(h.text) for h in hits)),
                "context_chars": sum(len(h.text) for h in hits),
            })

        if use_llm:
            from app.answer import answer_question

            for q, row in zip(questions, rows):
                ans = _answer_with_patience(answer_question, q["question"], store.search(cfg, q["question"], k))
                row["answer"] = ans.answer
                row["correct"] = graded(ans.answer, q["expect"])
                row["citation_ok"] = any(c.file == q["file"] and q["page"] in c.pages for c in ans.citations)
                row["input_tokens"] = ans.input_tokens
                row["output_tokens"] = ans.output_tokens

        n = len(rows)
        summary = {
            "chunk_size": cfg.size,
            "overlap": cfg.overlap,
            "chunks_in_index": store.stats(cfg)["chunks"],
            "ingest_s": round(ingest_s, 2) if reingest else None,
            "hit@1": sum(r["hit@1"] for r in rows) / n,
            f"hit@{k}": sum(r[f"hit@{k}"] for r in rows) / n,
            "mrr": statistics.mean(r["rr"] for r in rows),
            "context_precision": statistics.mean(r["precision"] for r in rows),
            "avg_context_chars": statistics.mean(r["context_chars"] for r in rows),
            "avg_search_ms": 1000 * statistics.mean(latencies),
        }
        if use_llm:
            summary["answer_accuracy"] = sum(r["correct"] for r in rows) / n
            summary["citation_accuracy"] = sum(r["citation_ok"] for r in rows) / n
            summary["avg_input_tokens"] = statistics.mean(r["input_tokens"] for r in rows)
        report[name] = {"summary": summary, "rows": rows}
    return report


def print_table(report: dict) -> None:
    names = list(report)
    metrics = list(report[names[0]]["summary"])
    print(f"\n| metric | " + " | ".join(f"{n} ({report[n]['summary']['chunk_size']} chars)" for n in names) + " |")
    print("|---|" + "---|" * len(names))
    for m in metrics:
        if m in ("chunk_size",):
            continue
        cells = []
        for n in names:
            v = report[n]["summary"][m]
            cells.append("-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"| {m} | " + " | ".join(cells) + " |")

    print("\nPer-question misses (gold page not in top-k):")
    for n in names:
        misses = [r["question"] for r in report[n]["rows"] if not r["rr"]]
        print(f"  {n}: {misses or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--llm", action="store_true", help="also generate answers with Claude (costs API tokens)")
    parser.add_argument("--no-reingest", action="store_true")
    args = parser.parse_args()

    report = run(args.k, args.llm, not args.no_reingest)
    print_table(report)
    out = HERE / ("results_llm.json" if args.llm else "results.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfull results -> {out}")


if __name__ == "__main__":
    main()
