# pqa-local：PaperQA 持久化文献检索

这个目录把「建库」和「提问」拆成两步：

1. `build_index.py`：一次性解析、切块、embedding，把结果存成 `index/docs.pkl`。
2. `ask_index.py`：每次提问只加载缓存、只 embedding 问题本身，不再解析 PDF、不再 embedding 全部 chunk。

原来的 `qa.py` 每次运行都会新建 `Docs()` 并对每个 PDF 调用 `aadd`，因此每次都会重新解析、切块、embedding，还会让 LLM 重新生成 citation（docname 也会跟着变）。这就是 token 消耗大、结果不稳定的主要原因。

## 快速开始

在 `pqa-local` 目录下：

```powershell
# 1) 建库（第一次约 4 分钟；以后只处理新增/修改的 PDF）
.\.venv\Scripts\python.exe build_index.py --allow-low-entropy-text

# 2) 提问（每次只加载缓存 + embedding 一个问题）
.\.venv\Scripts\python.exe ask_index.py "生命周期评价中主要的不确定性来源有哪些？"

# 3) 只看检索结果，不调用 LLM 生成答案（最省钱，适合调关键词）
.\.venv\Scripts\python.exe ask_index.py --retrieval-only -k 10 "life cycle assessment uncertainty"
```

`--allow-low-entropy-text` 只影响 PaperQA 的「文本熵」合法性检查，用来收录 2 篇字符种类特别多的中文论文；如果从零重建且不需要这两篇，可以去掉。

当前 `index/docs.pkl` 的实测结果：

| 指标 | 数值 |
| --- | --- |
| 成功索引 | 95 篇 PDF |
| chunk 数 | 1,373 |
| 缓存大小 | 19.1 MB |
| 向量矩阵 | 约 5.4 MB（1,373 × 1024 × 4B） |
| 建库耗时 | 约 4 分钟（232.8s） |
| 失败 | 8 篇（字体/ToUnicode 编码损坏） |
| 重复文件 | 0 |

常用参数：

```powershell
# 同时索引 文件 和 papers 两个目录
.\.venv\Scripts\python.exe build_index.py --corpus ..\文件 .\papers

# 只索引前 2 篇做小规模测试
.\.venv\Scripts\python.exe build_index.py --rebuild --limit 2

# 减少每次问答检索/摘要的 chunk 数（默认 10）
.\.venv\Scripts\python.exe ask_index.py --evidence-k 5 "your question"
```

## B：agent 多轮检索（持久化）

如果你要的是“agent 反复搜索、补证据、最后回答”，用这一组：

```powershell
# 第一次：从 index/docs.pkl 建 Tantivy 索引（不调用 embedding / LLM）
.\.venv\Scripts\python.exe build_agent_index.py

# 之后：每次加载索引，rebuild_index=False，agent 多轮检索
.\.venv\Scripts\python.exe agent_ask.py "能源系统生命周期评价中主要的不确定性来源有哪些？"

# 控制 agent 的搜索轮数和每轮证据数
.\.venv\Scripts\python.exe agent_ask.py --search-count 6 --max-timesteps 10 --evidence-k 6 "your question"
```

`build_agent_index.py` 直接从 `docs.pkl` 生成 Tantivy 全文索引和每篇论文的 chunk 缓存，**不会重新 embedding，也不会重新让 LLM 生成 citation**。本机实测约 4.7 秒，索引 13.4 MB。

`agent_ask.py` 的实测行为（中文问题）：

```text
Tools called: paper_search, paper_search, gather_evidence, gather_evidence,
              paper_search, gather_evidence, paper_search, gather_evidence,
              gather_evidence, gen_answer, complete
Contexts used: 24
Wall time: 23.2s
Prompt tokens: 57,721
Output tokens: 4,524
Status: success
```

也就是说，agent 确实会多轮搜索，而不是只做一次 top-k。

### 中文关键词问题

Tantivy 默认按空格/标点分词，中文长串会变成一个 token，直接 `paper_search("不确定性")` 往往召回为 0。`cjk_search.py` 在**建索引和查询时**都把中文切成重叠 bigram：

```text
不确定性 -> 不确 确定 定性
能源系统 -> 能源 源系 系统
```

这样中文问题也能命中中文论文，同时英文查询保持不变。

### 新增 PDF 后

```powershell
.\.venv\Scripts\python.exe build_index.py --allow-low-entropy-text   # 更新 Docs 缓存（增量）
.\.venv\Scripts\python.exe build_agent_index.py                      # 更新 Tantivy 索引（增量）
```

两者都只处理新增/修改的论文。若删除了论文或想彻底清理旧索引，运行 `build_agent_index.py --rebuild`。

## 什么时候需要重建

`index/docs.pkl` 与以下配置绑定，任一项变化都需要 `--rebuild`：

- paper-qa 版本
- embedding 模型（当前 `openai/BAAI/bge-m3`）
- chunk 大小/overlap（当前 5000/250）
- PDF 解析器
- multimodal 设置

**新增/删除/修改 PDF 不需要 `--rebuild`**：直接重跑 `build_index.py`。它按文件 MD5 增量处理，并默认 `--prune` 删除语料里已经不存在的文档。

## 设计要点

- **向量库**：`NumpyVectorStore`。当前 1,373 个 chunk、1024 维，检索矩阵约 5.4 MB，精确余弦检索足够快；不需要 Qdrant。
- **缓存**：pickle 整个 `Docs`（官方文档明确支持用 pickle 缓存 embeddings）。写盘前清空 `texts_index`，因为它的 `texts_hashes` 依赖 Python 进程级字符串哈希；加载后 PaperQA 会用已保存的 embedding 在本地重建矩阵，不调用 API。
- **稳定的 docname/citation**：建库时不传 `citation=None`，而是用文件名 + 文件夹 ID，避免每次让 LLM 生成不同 citation。切块和 docname 固定，检索结果可复现。
- **增量/并发**：按排序后的路径顺序逐篇添加，保证 docname 去重结果稳定；每 10 篇保存一次，失败可续跑。

## 已知限制

- 当前只索引 PDF。`文件` 里的 2 个 `.xlsx`、1 个 `.docx`、1 个 `.doc` 不会被索引；`.docx/.xlsx` 还需要 `unstructured` 依赖，`.doc` PaperQA 不支持。
- 8 个 PDF（多数为中文，含 4 篇 60–118 页的学位论文）因字体/ToUnicode 编码损坏，`pypdf` 会报 `UnicodeDecodeError: 'utf-16-be' ... truncated data`。详见 `index/build_report.json` 的 `errors`。建议用 Word 打开后另存为新的 PDF/TXT，再放回语料重跑。
- PaperQA 有一个「文本熵」合法性检查，会把字符种类特别多的中文论文误判为非文本（`This does not look like a text document`）。对这类高质量中文 PDF，可加 `--allow-low-entropy-text` 重跑；本目录最终索引已用该参数补入 2 篇。
- 引用显示为「文件名 [文件夹ID]」，不是正式文献格式；如需作者/年份/DOI，可后续接 manifest 或改用 `DocDetails`。
- `.env` 里有真实 API key，不要提交到 Git 或发给别人。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `build_index.py` | 一次性/增量建库（Docs 缓存） |
| `ask_index.py` | 加载缓存并做一次性问答（A） |
| `build_agent_index.py` | 从 Docs 缓存建 Tantivy 索引（B，不重新 embedding） |
| `agent_ask.py` | agent 多轮检索问答（B） |
| `cjk_search.py` | 中文 bigram 索引/查询，修复 Tantivy 中文召回 |
| `index_cache.py` | pickle 缓存读写与版本校验 |
| `scan_corpus.py` | 只读语料体检（页数、字符数、失败 PDF、重复文件） |
| `index/docs.pkl` | 生成的索引缓存（不要手动编辑） |
| `index/build_report.json` | 建库报告：每篇 chunk 数、耗时、失败原因 |
| `qa.py` | 原始 demo，每次重新 embedding；仅作对照 |
| `agent_qa.py` | 原始 agent demo；未复用本目录的持久化索引 |
