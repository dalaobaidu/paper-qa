r"""Corpus-wide health report for the persistent index built by build_index.py.

Loads ``index/docs.pkl`` once (no API calls), cross-references
``index/build_report.json``, and writes a single Markdown report covering:

* overview: doc/chunk counts, embedding coverage, cache metadata;
* build failures and duplicates from build_report.json;
* anomaly list: docs with too few chunks, missing embeddings, or garbled
  text (low printable-char ratio);
* chunks-per-doc distribution;
* per-doc detail table, flagged docs first;
* random chunk samples (fixed seed) for eyeballing actual indexed text.

Usage
-----
    cd pqa-local
    .\.venv\Scripts\python.exe inspect_index.py
    .\.venv\Scripts\python.exe inspect_index.py --samples 20 --chars 500
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from index_cache import cache_metadata, configure_utf8_stdio, load_docs
from qa import build_settings

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = BASE_DIR / "index" / "docs.pkl"
DEFAULT_REPORT = BASE_DIR / "index" / "build_report.json"
DEFAULT_OUTPUT = BASE_DIR / "index" / "health_report.md"

HISTOGRAM_BUCKETS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 10**9)]


def printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isspace() or c.isprintable()) / len(text) * 100


def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def cell(s: object) -> str:
    """Escape a value for use inside a Markdown table cell."""
    return str(s).replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corpus-wide health report for the PaperQA Docs cache."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=10, help="Random chunks shown.")
    parser.add_argument("--chars", type=int, default=300, help="Chars per sample.")
    parser.add_argument(
        "--min-chunks",
        type=int,
        default=2,
        help="Flag docs with at most this many chunks (default 2).",
    )
    parser.add_argument(
        "--printable-threshold",
        type=float,
        default=95.0,
        help="Chunks below this printable-char %% are flagged as garbled.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    args = parser.parse_args()

    configure_utf8_stdio()
    settings = build_settings()
    settings.verbosity = 0
    t0 = time.perf_counter()
    docs, payload = load_docs(args.cache, expected_meta=cache_metadata(settings))
    load_s = time.perf_counter() - t0

    build_report = {}
    if args.report.exists():
        try:
            build_report = json.loads(args.report.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt report just limits the output
            pass

    lines: list[str] = []
    out = lines.append

    # ---- per-doc stats ------------------------------------------------------
    rows = []
    bad_chunk_total = 0
    for doc in docs.docs.values():
        chunks = [t for t in docs.texts if t.doc.dockey == doc.dockey]
        lens = [len(t.text) for t in chunks]
        ratios = [printable_ratio(t.text) for t in chunks]
        embedded = sum(1 for t in chunks if getattr(t, "embedding", None))
        dim = next(
            (len(t.embedding) for t in chunks if getattr(t, "embedding", None)), 0
        )
        n_bad = sum(1 for r in ratios if r < args.printable_threshold)
        bad_chunk_total += n_bad
        flags = []
        if not chunks:
            flags.append("无chunk")
        elif len(chunks) <= args.min_chunks:
            flags.append("chunk过少")
        if chunks and embedded < len(chunks):
            flags.append(f"缺向量{len(chunks) - embedded}")
        if n_bad:
            flags.append(f"疑乱码{n_bad}块")
        rows.append(
            {
                "docname": doc.docname,
                "chunks": len(chunks),
                "chars": sum(lens),
                "avg_len": (sum(lens) // len(lens)) if lens else 0,
                "min_print": min(ratios) if ratios else 100.0,
                "embedded": embedded,
                "dim": dim,
                "flags": flags,
            }
        )
    rows.sort(key=lambda r: (not r["flags"], r["chunks"], r["docname"]))

    n_docs = len(docs.docs)
    n_texts = len(docs.texts)
    n_embedded = sum(1 for t in docs.texts if getattr(t, "embedding", None))
    total_chars = sum(len(t.text) for t in docs.texts)
    dim = next(
        (len(t.embedding) for t in docs.texts if getattr(t, "embedding", None)), 0
    )
    flagged = [r for r in rows if r["flags"]]
    errors = build_report.get("errors", [])
    duplicates = build_report.get("duplicates", [])
    in_corpus = build_report.get("pdfs_in_corpus")

    # ---- report -------------------------------------------------------------
    out("# PaperQA 索引全库体检报告")
    out("")
    out(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out(
        f"- 缓存: `{args.cache}` "
        f"({args.cache.stat().st_size / (1024 * 1024):.1f} MB, "
        f"built {payload.get('updated_at', '?')})"
    )
    out(f"- 向量模型: `{payload.get('embedding', '?')}`")
    out(
        f"- 分块配置: chunk_chars={payload.get('chunk_chars')}, "
        f"overlap={payload.get('overlap')}"
    )

    out("\n## 总览\n")
    out("| 指标 | 值 |")
    out("|---|---|")
    coverage = f"{n_docs}/{in_corpus}" if in_corpus else str(n_docs)
    out(f"| 已索引文档 | {coverage} 篇 |")
    out(f"| chunk 总数 | {n_texts:,}(平均每篇 {n_texts / max(n_docs, 1):.1f} 块)|")
    out(f"| 索引总字符 | {total_chars:,} |")
    out(
        f"| 向量覆盖 | {n_embedded}/{n_texts}"
        f"({'✅ 100%' if n_embedded == n_texts else '⚠️ 有缺失'},维度 {dim}) |"
    )
    out(f"| 构建失败 | {len(errors)} 篇 |")
    out(f"| 重复跳过 | {len(duplicates)} 篇 |")
    out(f"| 异常文档 | {len(flagged)}/{n_docs} 篇 |")

    out("\n## 异常清单\n")
    if not flagged and not errors:
        out("无异常,全库健康。✅")
    if errors:
        out(f"### 构建失败({len(errors)} 篇)\n")
        out("| 文件 | 错误 |")
        out("|---|---|")
        for e in errors:
            out(f"| `{cell(e['path'])}` | {e['type']}: {cell(e['message'][:80])} |")
        out("")
        out(
            "> 构建失败的 PDF 未进入索引;若是解析器兼容问题,可尝试 "
            "`build_index.py --allow-low-entropy-text` 或先 OCR 再重建。"
        )
        out("")
    if flagged:
        out(f"### 异常文档({len(flagged)} 篇)\n")
        out("| docname | 标记 | 块数 | 总字符 | 最低可打印% |")
        out("|---|---|---:|---:|---:|")
        for r in flagged:
            out(
                f"| {cell(truncate(r['docname'], 50))} | {', '.join(r['flags'])} "
                f"| {r['chunks']} | {r['chars']:,} | {r['min_print']:.1f} |"
            )
        out("")

    out("\n## 每篇 chunk 数分布\n")
    out("| 块数区间 | 篇数 | 分布 |")
    out("|---|---:|---|")
    for lo, hi in HISTOGRAM_BUCKETS:
        n = sum(1 for r in rows if lo <= r["chunks"] <= hi)
        label = f"{lo}+" if hi == 10**9 else f"{lo}-{hi}"
        out(f"| {label} 块 | {n} | {'█' * n} |")

    out("\n## 逐篇明细(异常在前,同组按 chunk 数升序)\n")
    out("| docname | 块数 | 总字符 | 均块长 | 最低可打印% | 向量 | 标记 |")
    out("|---|---:|---:|---:|---:|---|---|")
    for r in rows:
        emb = f"{r['embedded']}/{r['chunks']}"
        flag_text = ", ".join(r["flags"])
        out(
            f"| {cell(truncate(r['docname'], 50))} | {r['chunks']} | {r['chars']:,} "
            f"| {r['avg_len']} | {r['min_print']:.1f} | {emb} | {flag_text} |"
        )

    out(f"\n## 随机抽样 {args.samples} 个 chunk\n")
    out(f"> seed={args.seed},重跑结果一致;换一批抽样请用 `--seed`。")
    if n_texts:
        rng = random.Random(args.seed)  # noqa: S311 - sampling only
        picks = sorted(rng.sample(range(n_texts), min(args.samples, n_texts)))
        for seq, idx in enumerate(picks, start=1):
            t = docs.texts[idx]
            ratio = printable_ratio(t.text)
            flag = " ⚠️ 疑乱码" if ratio < args.printable_threshold else ""
            out(
                f"\n### #{seq} {t.doc.docname} :: {t.name}\n"
            )
            out(
                f"全局第 {idx} 块 · {len(t.text)} 字符 · 可打印 {ratio:.1f}%{flag}\n"
            )
            preview = t.text[: args.chars].replace("\n", " ")
            out("```text")
            out(preview + ("…" if len(t.text) > args.chars else ""))
            out("```")

    out("\n## 下一步建议\n")
    out("1. 先处理异常清单里的构建失败和疑乱码文档。")
    out('2. 用 `ask_index.py --retrieval-only -k 8 "问题"` 实测检索命中。')
    out("3. 抽查单篇细节: `check_doc.py <文件名关键词> --chunks 5`。")

    text = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")

    # Console: summary only; the file is the deliverable.
    print(f"加载缓存 {n_docs} 篇 / {n_texts:,} chunks,耗时 {load_s:.1f}s")
    print(f"构建失败 {len(errors)} 篇 | 异常文档 {len(flagged)} 篇 | 疑乱码块 {bad_chunk_total} 个")
    print(f"完整报告: {args.output}")


if __name__ == "__main__":
    main()
