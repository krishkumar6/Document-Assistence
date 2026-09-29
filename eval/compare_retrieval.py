"""Compare retrieval strategies on the labelled questions (free: no LLM calls).

    python -m eval.compare_retrieval            # both chunk configs
    python -m eval.compare_retrieval --config small

Strategies: vector, bm25, hybrid (RRF fusion), vector + rerank, hybrid + rerank.
A retrieved chunk counts as relevant if it is from the gold file and contains
text from the gold page (same rule as eval/compare_chunks.py). Metrics are
reported overall and per question kind ("standard" = the original questions,
"exact-term" = codes/numbers, "paraphrase" = reworded, few shared words).
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from app import retrieval
from app.config import CHUNK_CONFIGS
from eval.compare_chunks import relevant

HERE = Path(__file__).resolve().parent
STRATEGIES = [
    ("vector", "vector", False),
    ("bm25", "bm25", False),
    ("hybrid", "hybrid", False),
    ("vector+rerank", "vector", True),
    ("hybrid+rerank", "hybrid", True),
]
DEPTH = 10  # rank depth for MRR


def evaluate(cfg, questions, mode, rerank):
    rows, times = [], []
    for q in questions:
        t = time.perf_counter()
        hits = retrieval.search(cfg, q["question"], DEPTH, mode, rerank)
        times.append(1000 * (time.perf_counter() - t))
        ranks = [i for i, h in enumerate(hits, 1) if relevant(h, q)]
        first = ranks[0] if ranks else None
        rows.append({"question": q["question"], "kind": q.get("kind", "standard"), "rank": first})
    return rows, times


def summarize(rows):
    n = len(rows)
    return {
        "hit@1": sum(r["rank"] == 1 for r in rows) / n,
        "hit@3": sum(r["rank"] is not None and r["rank"] <= 3 for r in rows) / n,
        "mrr@10": statistics.mean(1 / r["rank"] if r["rank"] else 0 for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CHUNK_CONFIGS), action="append")
    args = parser.parse_args()
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    kinds = sorted({q.get("kind", "standard") for q in questions})

    # Warm up (model loading and index building shouldn't count as query latency).
    for cfg in CHUNK_CONFIGS.values():
        retrieval.search(cfg, "warm up", 3, "hybrid", True)

    report = {}
    for name in args.config or CHUNK_CONFIGS:
        cfg = CHUNK_CONFIGS[name]
        counts = ", ".join(f"{sum(q.get('kind', 'standard') == k for q in questions)} {k}" for k in kinds)
        print(f"\n### {name} chunks ({cfg.size} chars), {len(questions)} questions ({counts})\n")
        print("| strategy | hit@1 | hit@3 | MRR@10 | " + " | ".join(f"hit@1 {k}" for k in kinds) + " | ms/query |")
        print("|---|---|---|---|" + "---|" * len(kinds) + "---|")
        report[name] = {}
        for label, mode, rerank in STRATEGIES:
            rows, times = evaluate(cfg, questions, mode, rerank)
            s = summarize(rows)
            by_kind = {k: summarize([r for r in rows if r["kind"] == k])["hit@1"] for k in kinds}
            ms = statistics.median(times)
            report[name][label] = {**s, "hit@1_by_kind": by_kind, "median_ms": ms, "rows": rows}
            print(f"| {label} | {s['hit@1']:.0%} | {s['hit@3']:.0%} | {s['mrr@10']:.3f} | "
                  + " | ".join(f"{by_kind[k]:.0%}" for k in kinds) + f" | {ms:.0f} |")

        misses = {label: [r["question"] for r in v["rows"] if r["rank"] != 1] for label, v in report[name].items()}
        print("\nNot ranked first:")
        for label, qs in misses.items():
            print(f"  {label}: {qs or 'none'}")

    out = HERE / "results_retrieval.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfull results -> {out}")


if __name__ == "__main__":
    main()
