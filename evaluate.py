"""Compare every retrieval mode on the labeled questions in data/eval.json."""

import json
import time
from pathlib import Path

from hybrid_rag.metrics import hit_at, reciprocal_rank
from hybrid_rag.pipeline import MODES, Pipeline
from hybrid_rag.retrieval import load_corpus

DATA = Path(__file__).parent / "data"
TOP_K = 10


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def print_row(label: str, rows: list[dict], latency_ms: float | None = None) -> None:
    cells = [
        label,
        f"{mean([r['hit1'] for r in rows]):.2f}",
        f"{mean([r['hit3'] for r in rows]):.2f}",
        f"{mean([r['rr'] for r in rows]):.2f}",
    ]
    if latency_ms is not None:
        cells.append(f"{latency_ms:.0f}")
    print("| " + " | ".join(cells) + " |")


def main() -> None:
    queries = json.loads((DATA / "eval.json").read_text(encoding="utf-8"))
    pipeline = Pipeline.build(load_corpus(DATA / "corpus.json"))
    results: dict[str, list[dict]] = {}
    latency: dict[str, float] = {}

    for mode in MODES:
        results[mode] = []
        started = time.perf_counter()
        for query in queries:
            ranked = [d.id for d, _ in pipeline.search(query["question"], mode=mode, k=TOP_K)]
            relevant = set(query["relevant"])
            results[mode].append({
                "type": query["type"],
                "hit1": hit_at(ranked, relevant, 1),
                "hit3": hit_at(ranked, relevant, 3),
                "rr": reciprocal_rank(ranked, relevant),
            })
        latency[mode] = (time.perf_counter() - started) * 1000 / len(queries)

    print(f"All {len(queries)} questions\n")
    print("| Mode | Hit@1 | Hit@3 | MRR | ms/query |")
    print("|---|---|---|---|---|")
    for mode in MODES:
        print_row(mode, results[mode], latency[mode])

    for query_type in sorted({q["type"] for q in queries}):
        count = sum(q["type"] == query_type for q in queries)
        print(f"\n{query_type} questions ({count})\n")
        print("| Mode | Hit@1 | Hit@3 | MRR |")
        print("|---|---|---|---|")
        for mode in MODES:
            print_row(mode, [r for r in results[mode] if r["type"] == query_type])


if __name__ == "__main__":
    main()
