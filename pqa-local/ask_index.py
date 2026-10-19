r"""Ask questions against the persistent index built by build_index.py.

The first query after loading rebuilds the in-memory NumPy matrix from the
cached embeddings (local CPU work only).  It does not call the embedding API
for the documents, and it does not re-parse any PDF.

Usage
-----
    cd pqa-local
    .\.venv\Scripts\python.exe ask_index.py "your question"
    .\.venv\Scripts\python.exe ask_index.py --retrieval-only -k 8 "keywords"

``--retrieval-only`` performs vector retrieval without any LLM call, which is
useful for checking index quality and costs one query embedding only.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from index_cache import cache_metadata, configure_utf8_stdio, load_docs
from qa import build_settings

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = BASE_DIR / "index" / "docs.pkl"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Query the persistent PaperQA index.")
    parser.add_argument("question", nargs="+", help="Question to ask.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only retrieve chunks (no LLM answer, no answer-generation cost).",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=8, help="Chunks to show in retrieval-only."
    )
    parser.add_argument(
        "--show-contexts",
        action="store_true",
        help="Print every evidence context used for the answer.",
    )
    parser.add_argument(
        "--evidence-k",
        type=int,
        default=None,
        help="Chunks to retrieve/summarize per question (PaperQA default: 10).",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Max sources cited in the answer (PaperQA default: 5).",
    )
    args = parser.parse_args()

    configure_utf8_stdio()
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    logging.getLogger("paperqa").setLevel(logging.WARNING)

    question = " ".join(args.question).strip()
    settings = build_settings()
    settings.verbosity = 0
    if args.evidence_k is not None:
        settings.answer.evidence_k = args.evidence_k
    if args.max_sources is not None:
        settings.answer.answer_max_sources = args.max_sources

    t0 = time.perf_counter()
    docs, payload = load_docs(args.cache, expected_meta=cache_metadata(settings))
    print(
        f"Loaded {len(docs.docs):,} docs / {len(docs.texts):,} chunks "
        f"from {args.cache.name} in {time.perf_counter() - t0:.1f}s "
        f"(built {payload.get('updated_at', '?')})",
        flush=True,
    )
    print(f"Question: {question}", flush=True)

    if args.retrieval_only:
        t0 = time.perf_counter()
        texts = await docs.retrieve_texts(question, k=args.top_k, settings=settings)
        elapsed = time.perf_counter() - t0
        print(f"\n=== Top {len(texts)} chunks ({elapsed:.2f}s) ===")
        for i, text in enumerate(texts, start=1):
            print(f"\n[{i}] {text.doc.citation} :: {text.name}")
            print(text.text[:700].replace("\n", " "))
        print(f"\nIn-memory vector index: {len(docs.texts_index):,} chunks")
        return

    t0 = time.perf_counter()
    session = await docs.aquery(question, settings=settings)
    elapsed = time.perf_counter() - t0

    print("\n=== Answer ===")
    print(session.answer)
    print(f"\n=== Contexts used: {len(session.contexts)} ===")
    if args.show_contexts:
        for i, context in enumerate(session.contexts, start=1):
            print(f"\n[{i}] {context.citation} (score={context.score:.2f})")
            print(context.text[:700].replace("\n", " "))

    total_prompt = sum(v[0] for v in session.token_counts.values())
    total_completion = sum(v[1] for v in session.token_counts.values())
    print("\n=== Usage ===")
    print(f"Wall time:       {elapsed:.1f}s")
    print(f"Prompt tokens:   {total_prompt:,}")
    print(f"Output tokens:   {total_completion:,}")
    if session.cost:
        print(f"Cost (USD):      ${session.cost:.6f}")
    else:
        print("Cost (USD):      n/a (model not in LiteLLM's price map)")
    print(f"Cache chunks:    {len(docs.texts):,}")


if __name__ == "__main__":
    asyncio.run(main())
