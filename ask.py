"""Ask one question: python ask.py "How do I fix E4012?" [--mode bm25] [--generate]"""

import argparse
import sys
from pathlib import Path

from hybrid_rag.pipeline import MODES, Pipeline
from hybrid_rag.retrieval import load_corpus

CORPUS = Path(__file__).parent / "data" / "corpus.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="hybrid+rerank")
    parser.add_argument("--k", type=int, default=3, help="number of passages to return")
    parser.add_argument("--generate", action="store_true", help="answer with Claude (billed)")
    args = parser.parse_args()

    pipeline = Pipeline.build(load_corpus(CORPUS))
    results = pipeline.search(args.question, mode=args.mode, k=args.k)

    print(f"Top {len(results)} passages ({args.mode}):")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"  {rank}. [{doc.id}] {doc.title}  (score {score:.3f})")

    if args.generate:
        import anthropic

        from hybrid_rag.generate import generate_answer

        try:
            answer = generate_answer(args.question, [doc for doc, _ in results])
        except anthropic.AuthenticationError:
            print("No valid credentials: set ANTHROPIC_API_KEY.", file=sys.stderr)
            return 1
        except anthropic.RateLimitError:
            print("Rate limited, try again in a moment.", file=sys.stderr)
            return 1
        except anthropic.APIStatusError as error:
            print(f"Claude API error {error.status_code}: {error.message}", file=sys.stderr)
            return 1
        except anthropic.APIConnectionError:
            print("Could not reach the Claude API.", file=sys.stderr)
            return 1
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
        print(f"\nAnswer:\n{answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
