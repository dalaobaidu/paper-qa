r"""Spot-check the indexing quality of a single paper in the persistent cache.

Given a filename keyword, show whether the paper made it into both indexes
and let you eyeball the actual indexed text:

* presence in the Docs cache (``index/docs.pkl``) and the Tantivy agent index
  (``index_agent/lit``);
* chunk count, per-chunk char stats, embedding presence;
* previews of the first N chunks plus the last one, with a printable-char
  ratio to catch broken PDF text extraction (garbled ToUnicode maps).

Everything is read from local cache files -- zero API calls.

Usage
-----
    .\.venv\Scripts\python.exe check_doc.py Smith-2020
    .\.venv\Scripts\python.exe check_doc.py Lee-2021 --chunks 5 --chars 600
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from index_cache import cache_metadata, configure_utf8_stdio, load_docs
from qa import build_settings

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = BASE_DIR / "index" / "docs.pkl"
DEFAULT_INDEX_DIR = BASE_DIR / "index_agent"


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spot-check one paper's indexing quality from local caches."
    )
    parser.add_argument("keyword", help="Filename substring (case-insensitive).")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--name", default="lit")
    parser.add_argument(
        "--chunks", type=int, default=3, help="Preview first N chunks (default 3)."
    )
    parser.add_argument(
        "--chars", type=int, default=400, help="Chars shown per chunk preview."
    )
    args = parser.parse_args()

    configure_utf8_stdio()
    settings = build_settings()
    settings.verbosity = 0
    docs, payload = load_docs(args.cache, expected_meta=cache_metadata(settings))
    print(
        f"缓存: {len(docs.docs)} 篇 / {len(docs.texts):,} chunks "
        f"(built {payload.get('updated_at', '?')})"
    )

    kw = args.keyword.lower()
    matches = [
        d
        for d in docs.docs.values()
        if kw in d.docname.lower() or kw in getattr(d, "citation", "").lower()
    ]
    if not matches:
        sys.exit(f"未找到文件名含 {args.keyword!r} 的论文,换个关键词试试。")
    if len(matches) > 1:
        print(f"命中 {len(matches)} 篇,检查第一篇;要查其他篇请细化关键词:")
        for d in matches[:20]:
            print(f"  - {d.docname}")
    doc = matches[0]

    chunks = [t for t in docs.texts if t.doc.dockey == doc.dockey]
    print(f"\n=== {doc.docname} ===")
    print(f"dockey: {doc.dockey}")
    if not chunks:
        print("!! 缓存里没有该论文的任何 chunk(等于没索引上)。")
    else:
        lens = [len(t.text) for t in chunks]
        embedded = sum(1 for t in chunks if getattr(t, "embedding", None))
        dim = next(
            (len(t.embedding) for t in chunks if getattr(t, "embedding", None)), 0
        )
        print(
            f"chunks: {len(chunks)} | 总字符: {sum(lens):,} | "
            f"每块 {min(lens)}~{max(lens)} 字符"
        )
        mark = "OK" if embedded == len(chunks) else "!! 有缺向量"
        print(f"向量: {embedded}/{len(chunks)} 块已 embedding(维度 {dim}) {mark}")

    # Membership in the Tantivy agent index (agent_qa.py / agent_ask.py path).
    try:
        from paperqa.agents.search import SearchDocumentStorage, SearchIndex

        search_index = SearchIndex(
            fields=[*SearchIndex.REQUIRED_FIELDS, "title", "year"],
            index_name=args.name,
            index_directory=args.index_dir,
            storage=SearchDocumentStorage.PICKLE_COMPRESSED,
        )
        in_agent = doc.dockey in await search_index.index_files
        print(
            f"agent 索引({args.index_dir.name}/{args.name}): "
            f"{'已收录' if in_agent else '!! 未收录'}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"agent 索引检查跳过({exc})")

    if not chunks:
        return
    show = list(range(min(args.chunks, len(chunks))))
    if len(chunks) > args.chunks:
        show.append(len(chunks) - 1)  # always show the last chunk too
    for i in show:
        t = chunks[i]
        printable = (
            sum(1 for c in t.text if c.isspace() or c.isprintable())
            / max(len(t.text), 1)
            * 100
        )
        flag = "" if printable >= 95 else "  !! 疑似乱码"
        print(
            f"\n--- chunk {i + 1}/{len(chunks)}: {t.name} | "
            f"{len(t.text)} 字符 | 可打印 {printable:.1f}%{flag} ---"
        )
        preview = t.text[: args.chars].replace("\n", " ")
        print(preview + ("…" if len(t.text) > args.chars else ""))
    print("\n提示: 加 --chunks 10 看更多;再用 ask_index.py --retrieval-only 实测检索命中。")


if __name__ == "__main__":
    asyncio.run(main())
