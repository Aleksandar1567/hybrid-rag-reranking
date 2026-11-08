# Hybrid RAG with Reranking

A small, readable retrieval pipeline that compares four ways of finding the right passage for a question. It runs locally on a CPU, and generating an answer with Claude is optional.

```
                ┌─ BM25 (keywords) ──┐
question ───────┤                    ├─ RRF fusion ─ cross-encoder rerank ─ top passages ─ Claude (optional)
                └─ embeddings ───────┘
```

- **BM25** matches exact terms, such as `E4012` or `poll_interval_ms`.
- **Embeddings** match meaning, so "the box got too hot" can find a page about overheating.
- **Reciprocal Rank Fusion (RRF)** merges both rankings using positions only, because their scores aren't on the same scale.
- **Reranking** has a cross-encoder read the question and each candidate together. It's more accurate but too slow to run on every document, so it only sees the fused candidates.

## Setup

Requires Python 3.10 or newer. The models download on first run.

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# Compare all modes on the labeled questions
.venv/bin/python evaluate.py

# Search with one question
.venv/bin/python ask.py "How do I fix E4012?"
.venv/bin/python ask.py "the box got too hot" --mode bm25

# Also generate an answer with Claude (billed to your API key)
export ANTHROPIC_API_KEY=...
.venv/bin/python ask.py "How do I fix E4012?" --generate

# Run the unit tests (no model download needed)
.venv/bin/python -m pytest
```

## Results

Measured with `evaluate.py` on 18 hand-labeled questions over 24 documents, on a CPU:

| Mode | Hit@1 | Hit@3 | MRR | ms/query |
|---|---|---|---|---|
| bm25 | 0.61 | 0.72 | 0.69 | 0 |
| vector | 0.78 | 1.00 | 0.89 | 13 |
| hybrid | 0.72 | 0.83 | 0.81 | 12 |
| hybrid+rerank | 0.83 | 0.94 | 0.89 | 253 |

Hit@1 by question type:

| Mode | Keyword (6) | Semantic (7) | Mixed (5) |
|---|---|---|---|
| bm25 | 1.00 | 0.14 | 0.80 |
| vector | 0.67 | 0.71 | 1.00 |
| hybrid | 1.00 | 0.43 | 0.80 |
| hybrid+rerank | 1.00 | 0.57 | 1.00 |

What this run shows:

- **Hybrid is not automatically better.** It fixed vector search's misses on exact codes, but BM25's weak results on semantic questions pulled the fused ranking below vector search overall.
- **Reranking recovered part of that loss** and had the best Hit@1, at about 20 times the latency. It still missed one question in the top 3 that vector search found.
- **BM25 here has no stop-word removal**, so words like "the" can match on semantic questions. That is a likely cause of its weak results, and the first thing to try next.

Treat these numbers as an illustration, not a benchmark. The corpus is tiny and fictional, the same author wrote the documents and the questions, and one question moves an overall score by about 0.06. The 20 candidates per retriever also cover most of the 24 documents.

## Project layout

```
hybrid_rag/retrieval.py   BM25, vector search, RRF
hybrid_rag/pipeline.py    Cross-encoder and the search() modes
hybrid_rag/metrics.py     Hit@k and reciprocal rank
hybrid_rag/generate.py    Optional answer generation with Claude
data/corpus.json          Fictional IoT gateway handbook
data/eval.json            Labeled questions
```

The default models work best in English: `all-MiniLM-L6-v2` for embeddings and `ms-marco-MiniLM-L6-v2` for reranking. For other languages, swap in multi-language models.
