"""Retrieval metrics. A query counts as answered by any doc in its relevant set."""

from __future__ import annotations


def first_relevant_rank(ranked_ids: list[str], relevant: set[str]) -> int | None:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return rank
    return None


def hit_at(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    rank = first_relevant_rank(ranked_ids, relevant)
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    rank = first_relevant_rank(ranked_ids, relevant)
    return 0.0 if rank is None else 1 / rank
