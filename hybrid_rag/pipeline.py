"""Wires retrievers, fusion and the reranker into one search() call."""

from __future__ import annotations

from typing import Protocol

from .retrieval import BM25Retriever, Doc, VectorRetriever, reciprocal_rank_fusion

MODES = ("bm25", "vector", "hybrid", "hybrid+rerank")


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Reads the query and a document together, so it is accurate but slow.

    That cost is why it only sees the few candidates that retrieval returned.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        from sentence_transformers import CrossEncoder  # slow import, keep it lazy

        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, text) for text in texts])]


class Pipeline:
    def __init__(
        self,
        docs: list[Doc],
        keyword: Retriever,
        vector: Retriever,
        reranker: Reranker,
        candidates: int = 20,
    ):
        self.docs = {d.id: d for d in docs}
        self.keyword = keyword
        self.vector = vector
        self.reranker = reranker
        self.candidates = candidates

    @classmethod
    def build(cls, docs: list[Doc], candidates: int = 20) -> "Pipeline":
        return cls(
            docs,
            BM25Retriever(docs),
            VectorRetriever(docs),
            CrossEncoderReranker(),
            candidates,
        )

    def _hybrid(self, query: str) -> list[tuple[str, float]]:
        keyword_ids = [d for d, _ in self.keyword.search(query, self.candidates)]
        vector_ids = [d for d, _ in self.vector.search(query, self.candidates)]
        return reciprocal_rank_fusion([keyword_ids, vector_ids])

    def search(self, query: str, mode: str = "hybrid+rerank", k: int = 5) -> list[tuple[Doc, float]]:
        if mode == "bm25":
            hits = self.keyword.search(query, k)
        elif mode == "vector":
            hits = self.vector.search(query, k)
        elif mode == "hybrid":
            hits = self._hybrid(query)[:k]
        elif mode == "hybrid+rerank":
            candidate_ids = [d for d, _ in self._hybrid(query)[: self.candidates]]
            scores = self.reranker.score(
                query, [self.docs[d].full_text for d in candidate_ids]
            )
            hits = sorted(zip(candidate_ids, scores), key=lambda h: h[1], reverse=True)[:k]
        else:
            raise ValueError(f"Unknown mode {mode!r}, expected one of {MODES}")
        return [(self.docs[doc_id], score) for doc_id, score in hits]
