# PaperQA2 用于临床试验

PaperQA2 现在原生支持查询临床试验，除此之外还能查询用户提供的任何文档。它使用了一个新工具，恰当地命名为 `clinical_trials_search` 工具。用户不需要向工具本身提供任何临床试验数据，它会使用 `clinicaltrials.gov` 的 API 即时检索。截至 2025 年 1 月，该工具默认未启用，但配置起来很容易。下面是一个示例，我们只查询临床试验，不使用任何文档：

```python
from paperqa import Settings, agent_query

answer_response = await agent_query(
    query="What drugs have been found to effectively treat Ulcerative Colitis?",
    settings=Settings.from_name("search_only_clinical_trials"),
)

print(answer_response.session.answer)
```

### 输出

    已发现多种药物可有效治疗溃疡性结肠炎（Ulcerative Colitis, UC），
    它们针对该疾病的不同作用机制。

    Golimumab（戈利木单抗）是一种肿瘤坏死因子（TNF）抑制剂，商品名为 Simponi®，
    已被证明在治疗中重度 UC 方面具有疗效。通过皮下注射给药，研究表明它能在
    患者中维持临床反应至第 54 周（通过部分 Mayo 评分评估）（NCT02092285）。

    Mesalazine（美沙拉嗪）是一种抗炎药，常用于 UC 治疗。在一项比较美沙拉嗪灌肠
    与粪便菌群移植（FMT）治疗左侧 UC 的研究中，美沙拉嗪灌肠（每日 4 克）
    可有效诱导临床缓解（Mayo 评分 ≤ 2）（NCT03104036）。

    抗生素在 UC 管理中也显示出潜力。在先前的研究中，多西环素、阿莫西林和
    甲硝唑的组合使 60-70% 的中重度 UC 患者获得缓解。这些抗生素被认为可以
    改变肠道菌群，减少条件致病菌并促进有益菌生长（NCT02217722, NCT03986996）。

    Roflumilast（罗氟司特）是一种磷酸二酯酶-4（PDE4）抑制剂，目前正在研究
    用于治疗轻中度 UC。初步研究结果表明，将其加入常规治疗后可能改善
    疾病严重程度和生化指标（NCT05684484）。

    这些治疗方法突显了多种治疗策略，包括免疫抑制、菌群调节和抗炎机制。

你可以在每个查询的回复中看到所使用的每个临床试验的内联引用。如果你想查看用于回答该查询的具体上下文的更多数据：

```python
print(answer_response.session.contexts)
```

    [Context(context='该摘要提到，在 ClinicalTrials.gov 上搜索与治疗溃疡性结肠炎的
    药物相关的临床试验得到了 689 项试验。然而，它没有提供关于哪些药物
    被发现可有效治疗溃疡性结肠炎的具体信息。', text=Text(text='', name=...

使用 `Settings.from_name('search_only_clinical_trials')` 是一个快捷方式，但请注意，你可以通过将 `clinical_trials_search` 明确指定为工具，轻松将其添加到任何自定义的 `Settings` 中：

```python
from pathlib import Path
from paperqa import Settings, agent_query, AgentSetting
from paperqa.agents.tools import DEFAULT_TOOL_NAMES

# 你可以从 PaperQA 的默认工具列表开始
print(DEFAULT_TOOL_NAMES)
# >>> ['paper_search', 'gather_evidence', 'gen_answer', 'reset', 'complete'],

# 我们可以从一个包含可能有用论文的目录开始
print(list(Path("my_papers").iterdir()))

# 现在让我们使用标准工具 + 临床试验来查询
answer_response = await agent_query(
    query="What drugs have been found to effectively treat Ulcerative Colitis?",
    settings=Settings(
        paper_directory="my_papers",
        agent={"tool_names": DEFAULT_TOOL_NAMES + ["clinical_trials_search"]},
    ),
)

# 让我们查看格式化后的答案（包含参考文献）
print(answer_response.session.formatted_answer)
```

    问题：已发现哪些药物可有效治疗溃疡性结肠炎？

    已发现多种药物可有效治疗溃疡性结肠炎（UC），治疗策略因疾病严重程度
    和范围而异。对于轻中度 UC，5-氨基水杨酸（5-ASA）是一线治疗方案。
    局部 5-ASA，如美沙拉嗪栓剂（1 克/天），对直肠炎或远端结肠炎有效，
    可诱导 31-80% 的患者缓解。口服较高剂量的美沙拉嗪（如 4.8 克/天）
    可加速更广泛病变的临床改善（meier2011currenttreatmentof 第 1-2 页；
    meier2011currenttreatmentof 第 3-4 页）。

    对于中重度病例，常用皮质类固醇。口服类固醇如泼尼松龙
    （40-60 毫克/天）或静脉注射类固醇如甲基泼尼松龙（60 毫克/天）
    和氢化可的松（400 毫克/天）是诱导缓解的标准方案
    （meier2011currenttreatmentof 第 3-4 页）。肿瘤坏死因子（TNF）-α 阻滞剂，
    如英夫利昔单抗，对类固醇难治性病例有效（meier2011currenttreatmentof 第 2-3 页；
    meier2011currenttreatmentof 第 3-4 页）。

    免疫抑制剂，包括硫唑嘌呤和 6-巯基嘌呤，用于类固醇依赖性或难治性病例的
    维持治疗（meier2011currenttreatmentof 第 2-3 页；meier2011currenttreatmentof 第 3-4 页）。
    抗生素，如青霉素、四环素和甲硝唑的组合，在改变菌群和诱导部分患者
    缓解方面显示出前景，尽管其疗效不一（NCT02217722）。

    参考文献

    1. (meier2011currenttreatmentof 第 2-3 页)：Johannes Meier 和 Andreas Sturm。
    溃疡性结肠炎的当前治疗。World journal of gastroenterology, 17 27:3204-12, 2011。
    URL: https://doi.org/10.3748/wjg.v17.i27.3204, doi:10.3748/wjg.v17.i27.3204。

    2. (meier2011currenttreatmentof 第 3-4 页)：Johannes Meier 和 Andreas Sturm。
    溃疡性结肠炎的当前治疗。World journal of gastroenterology, 17 27:3204-12, 2011。
    URL: https://doi.org/10.3748/wjg.v17.i27.3204, doi:10.3748/wjg.v17.i27.3204。

    3. (NCT02217722)：Arie Levine 教授。使用溃疡性结肠炎饮食诱导缓解。
    Arie Levine 教授。2014。ClinicalTrials.gov 标识符：NCT02217722

    4. (meier2011currenttreatmentof 第 1-2 页)：Johannes Meier 和 Andreas Sturm。
    溃疡性结肠炎的当前治疗。World journal of gastroenterology, 17 27:3204-12, 2011。
    URL: https://doi.org/10.3748/wjg.v17.i27.3204, doi:10.3748/wjg.v17.i27.3204。

我们现在可以看到回复中同时引用了论文和临床试验。为了方便起见，我们还有一个可以直接使用的 `Settings.from_name`：

```python
from paperqa import Settings, agent_query

answer_response = await agent_query(
    query="What drugs have been found to effectively treat Ulcerative Colitis?",
    settings=Settings.from_name("clinical_trials"),
)
```

此外，这也适用于 `pqa` 命令行：

```bash
>>> pqa --settings 'search_only_clinical_trials' ask 'what is Ibuprofen effective at treating?'
```

    ...
    [13:29:50] 正在以 'certain' 完成 'what is Ibuprofen effective at treating?'。
            回答：布洛芬（Ibuprofen）是一种非甾体抗炎药（NSAID），可有效
            治疗多种疾病，包括疼痛、炎症和发热。
            它被广泛用于紧张型头痛，研究表明布洛芬钠与标准布洛芬和安慰剂相比，
            在 3 小时内可显著缓解疼痛并降低疼痛强度（NCT01362491）。
            静脉注射布洛芬可有效控制术后疼痛，特别是在骨科手术中，
            并有助于控制炎症过程。当与阿片类药物联合使用时，可减少
            阿片类药物的用量及相关副作用，使其成为
            多模式镇痛的关键组成部分（NCT05401916, NCT01773005）。

            布洛芬在儿科人群中也作为一线
            抗炎和解热药有效，因为与其他 NSAID 相比，其
            不良反应相对较低（NCT01478022）。
            此外，还研究了其通过 2% 布洛芬漱口水龈下冲洗
            治疗慢性牙周炎的潜力，可减少牙周袋深度和
            探诊出血，改善牙周健康（NCT02538237）。

            这些发现突显了布洛芬在治疗疼痛、炎症、
            发热以及紧张型头痛、术后疼痛和牙周疾病等特定病症方面的多功能性。
