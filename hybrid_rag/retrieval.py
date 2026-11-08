"""The two first-stage retrievers and the fusion step that combines them."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}"


def load_corpus(path: str | Path) -> list[Doc]:
    return [Doc(**row) for row in json.loads(Path(path).read_text(encoding="utf-8"))]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Okapi BM25 keyword search. Great at exact terms: error codes, flags, names."""

    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = [d.id for d in docs]
        self.term_counts = [Counter(tokenize(d.full_text)) for d in docs]
        self.lengths = [sum(c.values()) for c in self.term_counts]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        doc_freq = Counter(term for counts in self.term_counts for term in counts)
        # The "+ 1" keeps IDF positive even for terms found in most documents.
        self.idf = {
            term: math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def _score(self, query_terms: list[str], index: int) -> float:
        counts = self.term_counts[index]
        length_norm = 1 - self.b + self.b * self.lengths[index] / self.avg_length
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf:
                score += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        return score

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        scored = ((doc_id, self._score(terms, i)) for i, doc_id in enumerate(self.ids))
        hits = [hit for hit in scored if hit[1] > 0]
        return sorted(hits, key=lambda hit: hit[1], reverse=True)[:k]


class VectorRetriever:
    """Semantic search with embeddings. Finds paraphrases that share no keywords."""

    def __init__(self, docs: list[Doc], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # slow import, keep it lazy

        self.ids = [d.id for d in docs]
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [d.full_text for d in docs], normalize_embeddings=True, convert_to_numpy=True
        )

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        query_vector = self.model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )
        # On normalized vectors, the dot product equals cosine similarity.
        scores = self.embeddings @ query_vector
        best = scores.argsort()[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in best]


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Merge rankings using positions only: score(doc) = sum of 1 / (k + rank).

    BM25 and cosine scores live on different scales, so they can't be added
    directly. Ranks can, which is why RRF needs no score normalization.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
