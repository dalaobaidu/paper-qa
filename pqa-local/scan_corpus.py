"""Read-only corpus scan: page counts, extractable text, duplicate detection.

This does not call any API. It estimates how many chunks paper-qa will create
at the default chunk_chars=5000 / overlap=250 and flags PDFs with little or no
extractable text (likely scanned images), plus exact duplicate files.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import md5
from pathlib import Path

from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
CORPUS = BASE_DIR.parent / "文件"
CHUNK_CHARS = 5000
OVERLAP = 250
WORKERS = 4


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


def file_md5(path: Path) -> str:
    h = md5()  # noqa: S324 - content identity, not security
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan(path_str: str) -> dict:
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    path = Path(path_str)
    record: dict = {
        "path": str(path.relative_to(CORPUS)),
        "bytes": path.stat().st_size,
        "md5": file_md5(path),
    }
    try:
        reader = PdfReader(str(path))
        record["pages"] = len(reader.pages)
        chars = 0
        empty_pages = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            n = len(text.strip())
            chars += n
            if n < 20:
                empty_pages += 1
        record["chars"] = chars
        record["empty_pages"] = empty_pages
        record["chars_per_page"] = (
            round(chars / len(reader.pages), 1) if reader.pages else 0
        )
        record["est_chunks"] = max(
            0, -(-max(chars - OVERLAP, 0) // (CHUNK_CHARS - OVERLAP))
        )
    except Exception as e:  # noqa: BLE001 - report per-file, keep scanning
        record["error"] = f"{type(e).__name__}: {e}"
    return record


def main() -> None:
    configure_stdio()
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    pdfs = sorted(CORPUS.rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs under {CORPUS}")
    records = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(scan, str(p)): p for p in pdfs}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            records.append(rec)
            print(
                f"[{i}/{len(pdfs)}] {rec['path']}: "
                f"{rec.get('pages', '?')}p, {rec.get('chars', 0):,} chars"
                + (f", ERROR {rec['error']}" if "error" in rec else ""),
                flush=True,
            )
    records.sort(key=lambda r: r["path"])

    ok = [r for r in records if "error" not in r]
    total_pages = sum(r["pages"] for r in ok)
    total_chars = sum(r["chars"] for r in ok)
    total_chunks = sum(r["est_chunks"] for r in ok)
    hashes: dict[str, list[str]] = {}
    for r in records:
        hashes.setdefault(r["md5"], []).append(r["path"])
    duplicates = {h: ps for h, ps in hashes.items() if len(ps) > 1}
    low_text = [r for r in ok if r["chars_per_page"] < 200]

    summary = {
        "corpus": str(CORPUS),
        "pdf_count": len(pdfs),
        "parsed_ok": len(ok),
        "errors": len(records) - len(ok),
        "total_pages": total_pages,
        "total_chars": total_chars,
        "est_chunks": total_chunks,
        "est_embedding_tokens": round(total_chars / 1.6),
        "est_embedding_vector_gb_f32_1024d": round(total_chunks * 1024 * 4 / 1e9, 3),
        "duplicate_groups": duplicates,
        "low_text_pdfs": [
            {"path": r["path"], "chars_per_page": r["chars_per_page"]} for r in low_text
        ],
        "records": records,
    }
    out = BASE_DIR / "corpus_scan.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(
        f"PDFs: {summary['pdf_count']} "
        f"(ok {summary['parsed_ok']}, errors {summary['errors']})"
    )
    print(f"Pages: {total_pages:,}")
    print(f"Extractable chars: {total_chars:,}")
    print(f"Estimated chunks @5000/250: {total_chunks:,}")
    print(f"Estimated embedding tokens: {summary['est_embedding_tokens']:,}")
    print(
        "Estimated vectors (1024-d f32): "
        f"{summary['est_embedding_vector_gb_f32_1024d']} GB"
    )
    print(f"Duplicate groups: {len(duplicates)}")
    print(f"Low-text PDFs (<200 chars/page): {len(low_text)}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
