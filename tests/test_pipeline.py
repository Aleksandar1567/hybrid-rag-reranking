"""Fast tests that use fakes in place of the downloaded models."""

import pytest

from hybrid_rag.metrics import hit_at, reciprocal_rank
from hybrid_rag.pipeline import Pipeline
from hybrid_rag.retrieval import BM25Retriever, Doc, reciprocal_rank_fusion

DOCS = [
    Doc("a", "Error E4012", "TLS handshake failure on the gateway"),
    Doc("b", "Battery", "The gateway keeps running after power loss"),
    Doc("c", "Heat", "The gateway shuts down when it overheats"),
]


class FixedRetriever:
    def __init__(self, ids):
        self.ids = ids

    def search(self, query, k):
        return [(doc_id, 1.0) for doc_id in self.ids[:k]]


class PrefersReranker:
    """Gives the top score to the text containing a chosen word."""

    def __init__(self, word):
        self.word = word

    def score(self, query, texts):
        return [1.0 if self.word in text else 0.0 for text in texts]


def make_pipeline(keyword_ids, vector_ids, word="Heat"):
    return Pipeline(
        DOCS, FixedRetriever(keyword_ids), FixedRetriever(vector_ids), PrefersReranker(word)
    )


def test_rrf_prefers_doc_ranked_well_by_both_lists():
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]], k=1)
    assert fused[0][0] == "b"  # 1/3 + 1/3 beats 1/2 from a single list
    assert {doc_id for doc_id, _ in fused} == {"a", "b", "c"}


def test_rrf_score_formula():
    assert reciprocal_rank_fusion([["a"]], k=60) == [("a", 1 / 61)]


def test_bm25_finds_exact_code_and_skips_non_matches():
    hits = BM25Retriever(DOCS).search("E4012", k=3)
    assert [doc_id for doc_id, _ in hits] == ["a"]


def test_bm25_rarer_term_outweighs_common_term():
    hits = BM25Retriever(DOCS).search("gateway overheats", k=3)
    assert hits[0][0] == "c"


def test_hybrid_returns_docs_found_by_only_one_retriever():
    pipeline = make_pipeline(["a"], ["b"])
    ids = [doc.id for doc, _ in pipeline.search("q", mode="hybrid", k=2)]
    assert set(ids) == {"a", "b"}


def test_reranker_can_promote_last_candidate_to_first():
    pipeline = make_pipeline(["a", "b", "c"], ["a", "b", "c"])
    assert pipeline.search("q", mode="hybrid", k=1)[0][0].id == "a"
    assert pipeline.search("q", mode="hybrid+rerank", k=1)[0][0].id == "c"


def test_empty_keyword_results_still_work():
    pipeline = make_pipeline([], ["b"])
    assert pipeline.search("q", mode="hybrid+rerank", k=1)[0][0].id == "b"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        make_pipeline(["a"], ["a"]).search("q", mode="nope")


def test_metrics():
    assert hit_at(["x", "a"], {"a"}, 1) == 0.0
    assert hit_at(["x", "a"], {"a"}, 2) == 1.0
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["x"], {"a"}) == 0.0
