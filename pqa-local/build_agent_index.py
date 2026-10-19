"""Build PaperQA's persistent agent (Tantivy) index from the Docs cache.

This is the "B" path: PaperQA's ToolSelector agent can iteratively call
``paper_search`` / ``gather_evidence`` / ``gen_answer``.  The index is built
once and then reused with ``agent.rebuild_index=False``.

Why build from ``docs.pkl`` instead of re-parsing PDFs?
-------------------------------------------------------
``build_index.py`` already parsed and embedded all chunks.  The agent index
needs a Tantivy full-text index plus a pickled ``Docs`` object per paper.  We
can create both directly from the cache, so this step:

* makes **zero embedding API calls**;
* makes **zero LLM calls** (PaperQA's normal directory indexing would ask the
  LLM to generate a citation for every paper);
* keeps the exact same chunks/embeddings that ``ask_index.py`` uses.

CJK note
--------
Tantivy's default tokenizer does not split Chinese text, so we index CJK runs
as overlapping bigrams (see ``cjk_search.py``).  The agent's queries are
segmented the same way at search time by ``agent_ask.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from cjk_search import cjk_bigrams
from index_cache import cache_metadata, configure_utf8_stdio, load_docs
from paperqa import Docs
from paperqa.agents.search import SearchDocumentStorage, SearchIndex
from qa import build_settings

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = BASE_DIR / "index" / "docs.pkl"
DEFAULT_INDEX_DIR = BASE_DIR / "index_agent"
DEFAULT_NAME = "lit"


def human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the persistent Tantivy index used by agent_ask.py."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild the agent index from scratch.",
    )
    args = parser.parse_args()

    configure_utf8_stdio()
    logging.getLogger("paperqa").setLevel(logging.WARNING)

    settings = build_settings()
    settings.verbosity = 0
    docs, payload = load_docs(args.cache, expected_meta=cache_metadata(settings))
    print(
        f"Loaded cache: {len(docs.docs)} docs / {len(docs.texts):,} chunks "
        f"({payload.get('updated_at', '?')})",
        flush=True,
    )

    if Path(args.name).name != args.name:
        sys.exit("--name must be a simple directory name, not a path.")

    index_root = (args.index_dir / args.name).resolve()
    if args.rebuild and index_root.exists():
        print(f"Removing existing index: {index_root}", flush=True)
        shutil.rmtree(index_root)

    search_index = SearchIndex(
        fields=[*SearchIndex.REQUIRED_FIELDS, "title", "year"],
        index_name=args.name,
        index_directory=args.index_dir,
        storage=SearchDocumentStorage.PICKLE_COMPRESSED,
    )

    texts_by_dockey: dict[str, list] = defaultdict(list)
    for text in docs.texts:
        texts_by_dockey[text.doc.dockey].append(text)

    t0 = time.perf_counter()
    processed = skipped = 0
    for i, doc in enumerate(docs.docs.values(), start=1):
        texts = texts_by_dockey.get(doc.dockey, [])
        if not texts:
            skipped += 1
            continue
        body = "".join(t.text for t in texts)
        sub_docs = Docs(
            docs={doc.dockey: doc},
            texts=texts,
            docnames={doc.docname},
        )
        index_doc = {
            "title": cjk_bigrams(doc.docname),
            "year": "Unknown year",
            # The agent only uses this as a key, so the content hash is ideal.
            "file_location": doc.dockey,
            "body": cjk_bigrams(body),
        }
        await search_index.add_document(index_doc, document=sub_docs)
        processed += 1
        if i % 10 == 0 or i == len(docs.docs):
            print(
                f"[{i}/{len(docs.docs)}] indexed {processed} docs, "
                f"skipped {skipped}, {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    await search_index.save_index()
    elapsed = time.perf_counter() - t0

    size = sum(p.stat().st_size for p in index_root.rglob("*") if p.is_file())
    print("\n=== DONE ===")
    print(f"Docs processed: {processed}")
    print(f"Docs skipped:   {skipped}")
    print(f"Chunks:         {len(docs.texts):,}")
    print(f"Index:          {index_root}")
    print(f"Index size:     {human_mb(size)}")
    print(f"Elapsed:        {elapsed:.1f}s")
    print("\nQuery with:")
    print(
        '  .\\.venv\\Scripts\\python.exe agent_ask.py "your question here"'
    )


if __name__ == "__main__":
    asyncio.run(main())
