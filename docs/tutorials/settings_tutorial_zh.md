---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: test
    language: python
    name: python3
---

# 设置（Setup）

> 本教程也可作为 Jupyter notebook 查看，点击[这里](https://github.com/Future-House/paper-qa/blob/main/docs/tutorials/settings_tutorial.ipynb)。

本教程旨在展示如何使用 `Settings` 类来配置 `PaperQA`。
首先，我们将使用 `OpenAI` 和 `Anthropic` 模型，因此需要设置 `OPENAI_API_KEY` 和 `ANTHROPIC_API_KEY` 环境变量。
我们会同时使用这两个模型，以便清楚地看到 `paperqa` agent 何时使用其中一个、何时使用另一个。
我们使用 `python-dotenv` 从 `.env` 文件中加载环境变量。
因此，第一步是创建一个 `.env` 文件并安装所需的依赖包。

```python
# fmt: off
# 创建包含 OpenAI API 和 Anthropic API 密钥的 .env 文件
# 将 <your-openai-api-key> 和 <your-anthropic-api-key> 替换为你实际的 API 密钥
!echo "OPENAI_API_KEY=<your-openai-api-key>" > .env # fmt: skip
!echo "ANTHROPIC_API_KEY=<your-anthropic-api-key>" >> .env # fmt: skip

!uv pip install -q nest-asyncio python-dotenv aiohttp fhlmi "paper-qa[local]"
# fmt: on
```

```python
import os

import aiohttp
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv(".env")
```

```python
print("你已设置以下环境变量：")
print(
    f"OPENAI_API_KEY:    {'已设置' if os.environ['OPENAI_API_KEY'] else '未设置'}"
)
print(
    f"ANTHROPIC_API_KEY: {'已设置' if os.environ['ANTHROPIC_API_KEY'] else '未设置'}"
)
```

我们将使用 `lmi` 包来获取模型名称，并使用 `.papers` 目录来保存将要使用的文档。

```python
from lmi import CommonLLMNames

llm_openai = CommonLLMNames.OPENAI_TEST.value
llm_anthropic = CommonLLMNames.ANTHROPIC_TEST.value

# 如果 papers 目录不存在则创建
os.makedirs("papers", exist_ok=True)

# 从 arXiv 下载论文并保存到 papers 目录
url = "https://arxiv.org/pdf/2407.01603"
async with aiohttp.ClientSession() as session, session.get(url, timeout=60) as response:
    content = await response.read()
    with open("papers/2407.01603.pdf", "wb") as f:
        f.write(content)
```

`Settings` 类用于配置 PaperQA 的各项设置。
官方文档可在[这里](https://github.com/Future-House/paper-qa?tab=readme-ov-file#settings-cheatsheet)查看，开源代码可在[这里](https://github.com/Future-House/paper-qa/blob/main/src/paperqa/settings.py)查看。

下面是一个如何使用 `Settings` 类的基本示例。为了清晰起见，我们会写得比较冗长。请注意，大多数设置都是可选的，默认值在大多数情况下已经足够好用。更多信息请参阅[每个设置的描述](https://github.com/Future-House/paper-qa/blob/main/src/paperqa/settings.py)。

在这个 `Settings` 对象中，我想专门讨论一下 llm 是如何配置的，以及 `paperqa` 是如何查找论文的。

一个常见的困惑来源是 `paperqa` 中使用了多个 `llm`。我们有 `llm`、`summary_llm`、`agent_llm` 和 `embedding`。因此，如果将 `llm` 设置为 `Anthropic` 模型，`summary_llm` 和 `agent_llm` 仍然需要 `OPENAI_API_KEY`，因为默认使用的是 `OpenAI` 模型。

在 `paperqa` 中使用 `llm` 的对象包括 `llm`、`summary_llm`、`agent_llm` 和 `embedding`：

- `llm`：agent 用来推理问题、从文档中提取元数据等的主 LLM。
- `summary_llm`：用于对论文进行摘要的 LLM。
- `agent_llm`：用于回答问题和选择工具的 LLM。
- `embedding`：用于对论文进行向量化的嵌入模型。

让我们看一些围绕这个概念的示例。首先，我们将 `llm` 设置为 `OpenAI` 模型来定义设置。请注意这并不是一个完整的设置列表。请花点时间通读这个 `Settings` 类，了解所有可以进行的自定义配置。

```python
import pathlib

from paperqa.prompts import (
    CONTEXT_INNER_PROMPT,
    CONTEXT_OUTER_PROMPT,
    citation_prompt,
    default_system_prompt,
    env_reset_prompt,
    env_system_prompt,
    qa_prompt,
    select_paper_prompt,
    structured_citation_prompt,
    summary_json_prompt,
    summary_json_system_prompt,
    summary_prompt,
)
from paperqa.settings import (
    AgentSettings,
    AnswerSettings,
    IndexSettings,
    ParsingSettings,
    PromptSettings,
    Settings,
)

settings = Settings(
    llm=llm_openai,
    llm_config={
        "model_list": [
            {
                "model_name": llm_openai,
                "litellm_params": {
                    "model": llm_openai,
                    "temperature": 0.1,
                    "max_tokens": 4096,
                },
            }
        ],
        "rate_limit": {
            llm_openai: "30000 per 1 minute",
        },
    },
    summary_llm=llm_openai,
    summary_llm_config={
        "rate_limit": {
            llm_openai: "30000 per 1 minute",
        },
    },
    embedding="text-embedding-3-small",
    embedding_config={},
    temperature=0.1,
    batch_size=1,
    verbosity=1,
    answer=AnswerSettings(
        evidence_k=10,
        evidence_retrieval=True,
        evidence_summary_length="about 100 words",
        evidence_skip_summary=False,
        answer_max_sources=5,
        max_answer_attempts=None,
        answer_length="about 200 words, but can be longer",
        max_concurrent_requests=10,
    ),
    parsing=ParsingSettings(
        reader_config={"chunk_chars": 5000, "overlap": 250},
        citation_prompt=citation_prompt,
        structured_citation_prompt=structured_citation_prompt,
    ),
    prompts=PromptSettings(
        summary=summary_prompt,
        qa=qa_prompt,
        select=select_paper_prompt,
        pre=None,
        post=None,
        system=default_system_prompt,
        use_json=True,
        summary_json=summary_json_prompt,
        summary_json_system=summary_json_system_prompt,
        context_outer=CONTEXT_OUTER_PROMPT,
        context_inner=CONTEXT_INNER_PROMPT,
    ),
    agent=AgentSettings(
        agent_llm=llm_openai,
        agent_llm_config={
            "model_list": [
                {
                    "model_name": llm_openai,
                    "litellm_params": {
                        "model": llm_openai,
                    },
                }
            ],
            "rate_limit": {
                llm_openai: "30000 per 1 minute",
            },
        },
        agent_prompt=env_reset_prompt,
        agent_system_prompt=env_system_prompt,
        search_count=8,
        index=IndexSettings(
            paper_directory=pathlib.Path.cwd().joinpath("papers"),
            manifest_file=None,
            index_directory=pathlib.Path.cwd().joinpath("papers/index"),
        ),
    ),
)
```

显而易见，`Paperqa` 是完全可定制的。在此我们再次强调，尽管可以进行如此精细的自定义，默认值在大多数情况下已经足够好用。当然，欢迎用户探索这些设置，根据自己的需求自定义 `paperqa`。

我们还将 `settings.verbosity` 设置为 1，这会打印出 agent 的配置信息。首次运行后，你可以将其设置为 0 来关闭日志输出。

```python
from paperqa import ask

response = ask(
    "What are the most relevant language models used for chemistry?", settings=settings
)
```

这应该能正常工作。现在让我们尝试移除 `OPENAI_API_KEY`，然后用相同的设置再次运行相同的问题。

```python
os.environ["OPENAI_API_KEY"] = ""
print("你已设置以下环境变量：")
print(
    f"OPENAI_API_KEY:    {'已设置' if os.environ['OPENAI_API_KEY'] else '未设置'}"
)
print(
    f"ANTHROPIC_API_KEY: {'已设置' if os.environ['ANTHROPIC_API_KEY'] else '未设置'}"
)
```

```python
response = ask(
    "What are the most relevant language models used for chemistry?", settings=settings
)
```

这显然会失败。我们没有有效的 `OPENAI_API_KEY`，所以 agent 将无法使用 `OpenAI` 模型。让我们把它改成 `Anthropic` 模型，看看是否能正常工作。

```python
settings.llm = llm_anthropic
settings.llm_config = {
    "model_list": [
        {
            "model_name": llm_anthropic,
            "litellm_params": {
                "model": llm_anthropic,
                "temperature": 0.1,
                "max_tokens": 512,
            },
        }
    ],
    "rate_limit": {
        llm_anthropic: "30000 per 1 minute",
    },
}
settings.summary_llm = llm_anthropic
settings.summary_llm_config = {
    "rate_limit": {
        llm_anthropic: "30000 per 1 minute",
    },
}
settings.agent = AgentSettings(
    agent_llm=llm_anthropic,
    agent_llm_config={
        "rate_limit": {
            llm_anthropic: "30000 per 1 minute",
        },
    },
    index=IndexSettings(
        paper_directory=pathlib.Path.cwd().joinpath("papers"),
        manifest_file=None,
        index_directory=pathlib.Path.cwd().joinpath("papers/index"),
    ),
)
settings.embedding = "st-multi-qa-MiniLM-L6-cos-v1"
response = ask(
    "What are the most relevant language models used for chemistry?", settings=settings
)
```

<!-- #region -->
现在 agent 只能使用 `Anthropic` 模型，尽管我们没有有效的 `OPENAI_API_KEY`，问题仍然能得到回答，因为 agent 不会使用 `OpenAI` 模型。
注意我们还更改了 `embedding`，因为它默认使用的是 `text-embedding-3-small`，这是一个 `OpenAI` 模型。`Paperqa` 实现了多种嵌入模型。
更多信息请参阅[文档](https://github.com/Future-House/paper-qa?tab=readme-ov-file#embedding-model)。

此外，请注意这是一个为了清晰起见而写得非常冗长的示例。我们其实可以只设置 llm 的名称，其余部分使用默认设置：

```python
llm_anthropic_config = {
    "model_list": [{
            "model_name": llm_anthropic,
    }]
}

settings.llm = llm_anthropic
settings.llm_config = llm_anthropic_config
settings.summary_llm = llm_anthropic
settings.summary_llm_config = llm_anthropic_config
settings.agent = AgentSettings(
    agent_llm=llm_anthropic,
    agent_llm_config=llm_anthropic_config,
    index=IndexSettings(
        paper_directory=pathlib.Path.cwd().joinpath("papers"),
        manifest_file=None,
        index_directory=pathlib.Path.cwd().joinpath("papers/index"),
    ),
)
settings.embedding = "st-multi-qa-MiniLM-L6-cos-v1"
```

<!-- #endregion -->

# 输出结果

`Paperqa` 返回一个 `PQASession` 对象，它不仅包含答案，还包含为回答问题而收集的所有信息。我们建议打印 `PQASession` 对象（`print(response.session)`）以了解它所包含的信息。让我们来查看 `PQASession` 对象：

```python
print(response.session)
```

```python
print("让我们检查 paperqa 返回的 PQASession 对象：\n")

print(f"状态：{response.status.value}")

print("1. 提出的问题：")
print(f"{response.session.question}\n")

print("2. 提供的答案：")
print(f"{response.session.answer}\n")
```

除了答案之外，`PQASession` 对象还包含用于生成答案的所有参考文献和上下文。

由于 `paperqa` 会将文档拆分成多个块（chunks），每个块都是一个有效的参考文献。你可以看到它还引用了上下文所在的页码。

```python
print("3. 引用的参考文献：")
print(f"{response.session.references}\n")
```

最后，`PQASession.session.contexts` 包含用于生成答案的上下文。每个上下文都有一个分数（score），表示问题与该上下文之间的相似度。
`Paperqa` 使用这个分数来选择哪些上下文与回答问题更相关。

```python
print("4. 用于生成答案的上下文：")
print(
    "这些是被检索到并用于组织答案的相关文本段落："
)
for i, ctx in enumerate(response.session.contexts, 1):
    print(f"\n上下文 {i}：")
    print(f"来源：{ctx.text.name}")
    print(f"内容：{ctx.context}")
    print(f"分数：{ctx.score}")
```
