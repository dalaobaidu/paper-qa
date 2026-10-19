"""CJK-friendly keyword indexing for PaperQA's Tantivy search index.

PaperQA's default Tantivy tokenizer splits on whitespace/punctuation.  A run of
Chinese characters therefore becomes one huge token, so a query like
``不确定性`` does not match a document containing ``不确定性来源``.

This module converts each CJK run into space-separated overlapping bigrams
before indexing and before querying.  Tantivy's default tokenizer then sees
normal tokens, and the query parser can match them.  English text is left
unchanged.

Example:
    能源系统生命周期评价 -> 能源 源系 系统 统生 生命 命周 周期 期评 评价
    不确定性             -> 不确 确定 定性
"""

from __future__ import annotations

CJK_RANGES = (
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x3040, 0x30FF),  # Hiragana + Katakana (harmless for Chinese corpora)
)


def is_cjk(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in CJK_RANGES)


def _segment_run(run: str) -> str:
    if len(run) == 1:
        return run
    return " ".join(run[i : i + 2] for i in range(len(run) - 1))


def cjk_bigrams(text: str) -> str:
    """Replace CJK runs with space-separated overlapping bigrams."""
    if not text:
        return text
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            out.append(_segment_run("".join(run)))
            run.clear()

    for char in text:
        if is_cjk(char):
            run.append(char)
        else:
            flush()
            out.append(char)
    flush()
    return "".join(out)


def install_cjk_query_patch() -> None:
    """Make PaperQA's ``SearchIndex.query`` bigram-segment CJK queries."""
    from paperqa.agents.search import SearchIndex

    if getattr(SearchIndex, "_cjk_bigram_patched", False):
        return

    original_query = SearchIndex.query

    async def query(self, query: str, *args, **kwargs):
        return await original_query(self, cjk_bigrams(query), *args, **kwargs)

    SearchIndex.query = query  # type: ignore[method-assign]
    SearchIndex._cjk_bigram_patched = True  # type: ignore[attr-defined]
