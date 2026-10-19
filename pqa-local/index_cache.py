"""Persist and reload a paper-qa ``Docs`` collection (parse + embeddings cache).

Why this exists
---------------
``Docs.aadd`` parses a PDF, chunks it, and calls the embedding API for every
chunk.  A plain ``Docs()`` object lives only in memory, so the next process
starts from zero.  PaperQA's docs explicitly support caching embeddings by
pickling ``Docs``:

    https://github.com/Future-House/paper-qa#caching-embeddings

This module stores the whole ``Docs`` object in one pickle (atomic replace),
with a small metadata header so a cache built with a different embedding model
or chunking configuration can be rejected instead of silently producing bad
retrieval.

Important detail
----------------
``NumpyVectorStore.texts_hashes`` stores Python's per-process string hashes.
Those hashes change between interpreter runs (hash randomisation), so a
pickled *built* search index would look stale in a new process and duplicate
every chunk.  We therefore clear ``docs.texts_index`` before pickling and let
PaperQA rebuild it locally from the already-stored embeddings on the first
query (no API calls: ``_build_texts_index`` only embeds texts whose
``embedding is None``).
"""

from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paperqa import Docs, Settings
from paperqa.utils import get_stable_str
from paperqa.version import __version__ as pqa_version

CACHE_FORMAT = 1


def configure_utf8_stdio() -> None:
    """Make Chinese paths/answers print correctly on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - best effort only
                pass


def cache_metadata(settings: Settings, corpus: Path | None = None) -> dict[str, Any]:
    try:
        pdf_parser = get_stable_str(settings.parsing.parse_pdf, for_hash=True)
    except Exception:  # noqa: BLE001 - identity is best-effort metadata
        pdf_parser = repr(settings.parsing.parse_pdf)
    return {
        "cache_format": CACHE_FORMAT,
        "paperqa_version": pqa_version,
        "embedding": settings.embedding,
        "chunk_chars": settings.parsing.reader_config.get("chunk_chars"),
        "overlap": settings.parsing.reader_config.get("overlap"),
        "multimodal": str(settings.parsing.multimodal),
        "pdf_parser": pdf_parser,
        "corpus": str(corpus) if corpus else None,
    }


def save_docs(
    docs: Docs,
    path: str | os.PathLike,
    meta: dict[str, Any],
    created_at: str | None = None,
) -> None:
    """Atomically pickle ``docs`` (without the process-dependent vector index)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # See module docstring: hashes of strings are not stable across processes.
    docs.texts_index.clear()

    now = datetime.now(timezone.utc).isoformat()
    payload = {
        **meta,
        "created_at": created_at or now,
        "updated_at": now,
        "n_docs": len(docs.docs),
        "n_texts": len(docs.texts),
        "docs": docs,
    }

    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_docs(
    path: str | os.PathLike, expected_meta: dict[str, Any] | None = None
) -> tuple[Docs, dict[str, Any]]:
    """Load a cache and reject caches built with incompatible settings."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No cache at {path}. Build it once with: python build_index.py"
        )

    with path.open("rb") as f:
        payload = pickle.load(f)  # noqa: S301 - local cache, not untrusted input

    if not isinstance(payload, dict) or "docs" not in payload:
        raise ValueError(f"{path} is not a paper-qa Docs cache produced by this tool.")
    if payload.get("cache_format") != CACHE_FORMAT:
        raise ValueError(
            f"Cache format {payload.get('cache_format')} is not supported "
            f"(expected {CACHE_FORMAT}). Rebuild with --rebuild."
        )
    if payload.get("paperqa_version") != pqa_version:
        raise ValueError(
            "Cache was built with paper-qa "
            f"{payload.get('paperqa_version')}, but the environment has {pqa_version}. "
            "Rebuild with --rebuild (or run the original environment)."
        )
    if expected_meta:
        for key in ("embedding", "chunk_chars", "overlap", "multimodal", "pdf_parser"):
            if payload.get(key) != expected_meta.get(key):
                raise ValueError(
                    f"Cache {key}={payload.get(key)!r} does not match current "
                    f"settings {key}={expected_meta.get(key)!r}. "
                    "Rebuild with --rebuild."
                )

    docs: Docs = payload["docs"]
    # Defensive: an older cache may carry the process-dependent index.
    docs.texts_index.clear()
    return docs, payload


def cache_info(path: str | os.PathLike) -> dict[str, Any]:
    """Read just the metadata header without unpickling the whole Docs."""
    # Pickle does not support partial loading, so this is a full load.  Kept as
    # a separate helper so callers do not need to know the payload layout.
    _, payload = load_docs(path)
    return {k: v for k, v in payload.items() if k != "docs"}
