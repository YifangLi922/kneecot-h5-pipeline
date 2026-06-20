# 代码问题整理

> 核心原则：**先生成原始输出 -> 用一把尺子统一评分 -> 统一比较 -> 写进论文。**

## 一、`LLM` 和 `VLM` 线中的 `evaluate/evaluation.py` 没有对齐

### 问题描述

`LLM` 和 `VLM` 两条线中的 `evaluate/evaluation.py` 没有对齐：虽然都能跑通，但各自估计各自的结果，最后合到 `comparison` 中时并不是在同一条线上比较，无法回应 research questions。

### 解决方式

`LLM` 和 `VLM` 线中的 `evaluation` 只负责**提取数据**，不负责评估本身；评估共同使用同一套代码来做，可以放在 `analysis` 文件夹中。

### 按严重程度排序

#### 1. 字段命名不一致

这一点会直接导致对比代码跑不通或算错。同一份 `eval_set.json` 两边读取出来的字段名不一样：

| 含义 | LLM 线（`evaluation_llm`） | VLM 线（`evaluation_vlm`） |
|---|---|---|
| 题型值 | `"yes_no"` | `"yesno"` |
| prompt 条件 | `prompt_mode` | `prompt_key` |
| 标准答案 | `gt_label` | `ground_truth` |

最危险的是题型：VLM 代码筛的是 `c["qtype"] == "yesno"`，但如果共享的 `eval_set` 里写的是 `yes_no`，它会评出 **0 个 case** 还不报错。这三处必须统一成同一套，以 `build_eval_set.py` 实际生成的为准。

#### 2. 两条线使用了不同的 yes/no 解析器

- `LLM` 线在 `evaluation_llm` 里自带一个三级瀑布解析器：`答案 -> 英文 yes/no -> 中文是/否`。
- `VLM` 线通过 `from prompts import parse_yes_no` 从另一个文件导入。

只要这两个解析器有一点点不一样，`LLM` 和 `VLM` 的对比就不公平：你分不清差异是模型造成的，还是解析器造成的。

**要求：**两条线必须共用同一个 `parse_yes_no`。

#### 3. `inference` 评分不是最终方案

`LLM` 线目前是“中文 bigram 重叠 > 30% 算对”这种粗糙自动规则，而且这个函数在文件里还重复定义了三遍，后面的覆盖前面的。`VLM` 线则完全没评分，`run_inference_eval` 只是把 `raw` 原样存下来。

这两者不对齐。已经决定：`inference` 走 **judge 初筛 + 人工抽查**。

因此，两条线的 `inference` 评分都要废掉，统一交给后面的判官流程。不要只改 DS 的代码，否则会变成“LLM 用 bigram 评、VLM 用 judge 评”，仍然不 matched。

#### 4. `McNemar` 和指标不对齐

- `LLM` 线会算 `accuracy + McNemar`，即 `direct vs CoT` 成对检验。
- `VLM` 线算的是 `accuracy / f1 / precision / recall`，没有 `McNemar`。

分析方案需要的成对 `McNemar`、`CoT gain`、`visual gain` 都是跨条件的，本来就应该放在一个统一的对比脚本里算，不该塞在各自的 run 里。

### 还需要手动确认的两点

1. `VLM` 的 `CONDITIONS` 是否把 4 个主条件 + image-only ablation 的 prompt 都接上了。
2. 切片是否按 Methodology 那样做了 `CLAHE + [D//3, D//2]`。

---

## 二、Inference questions 回答准确率标准未定

### 问题描述

衡量 `inference questions` 回答准确率的标准还没有定下来：到底要不要走 `LLM Judge + 人工筛查`？

现有 `evaluation codes` 中评判 `inference questions` 回答质量的标准只是字符匹配，`VLM` 线甚至没有做 `inference evaluation` 的代码。

### 解决方式

同上一节：`evaluation.py` 只保存分析后的结果，评估用其他代码来做。

`inference questions` 的准确度应该采用：

```text
LLM Judge + Human Validation
```

目前 `rubric` 和 `judge` 已经写好，应该可以直接使用。

---

## 三、根本修改：把“生成”和“评分”分家

上面 1-4 看着很多，其实一个改动就能全解决：这也是 driver 当前最该拍板的架构决定。

### 生成层：两条线各自做

`LLM` 线和 `VLM` 线都只负责跑出原始逐题记录，用完全相同的字段名保存：

```text
case_id
question_id
question
qtype
prompt_mode
modality
gt_label
gt_answer
raw_output
```

`VLM` 线可以多存一个 `nii_path` / 图片信息。

**注意：生成层不算分。**

### 评分层：全组只有一份，共享使用

统一脚本读取两条线的原始记录，并完成：

1. 用同一个解析器计算 `yes/no accuracy`。
2. 用同一个判官 + 人工流程评 `inference`。
3. 统一计算 `McNemar / CoT gain / visual gain`。
4. 输出结果表。

这样 `LLM` 和 `VLM` 必然会被同一套代码、同一套标准评判，`matched comparison` 才真正成立。

DS 那份代码的“生成”部分基本能用，需要改的主要是字段名对齐和评分逻辑抽出去。对应 README 里提到的 `code/analysis/compare.py`，就应该是这个共享评分层。

---

## 四、`LLM-judge` 初筛 + 人工抽查：具体五步

### 第一步：冻结“评分规则单”（`rubric`）

这一步绕不开。模型判官也得有标准，不能让它凭自己的医学观点判。

规则单应包括：

- 封闭类别清单；
- 每个类别的 positive / negative 例子；
- `correct / incorrect / UNCLEAR` 的判定规则。

好消息是：有了判官之后，就不再需要每个同义词都手工列一遍。只要给它“类别 + 几个示例 + 规则”，它照着判断即可。规则单从 DS 那份导出文件里读 25-30 条 inference 答案就能写出来，工作量比纯手工标小很多。

### 第二步：选择判官模型

判官模型必须满足两个要求：

1. **必须是本地模型。** 你们整个设计的立身之本就是“不把医疗数据传外部 API”。README 和 Methodology 都写了这一点。判官要读进 MR 结论文本，所以判官也得本地跑，否则自相矛盾。
2. **判官模型要比被评的模型更强。** 判官不能和被评的 7B 一样弱，否则“裁判和选手水平一样”就没有意义。推荐 `Qwen2.5-32B-Instruct`。如果 HPC 上 4-bit 能跑，就明显强于 7B；算力够的话 72B 更好。注意：不要用正在被评测的那个模型实例去评自己的输出，避免自我偏好。用更大的同族模型当判官是可以的。

判官使用 `greedy / temperature=0`，保证可复现。

### 第三步：让判官跑第一遍

给判官一个结构化 prompt，喂进去三样东西：

1. 这道题的 ground-truth 结论，即同它的类别；
2. 模型抠出来的最终结论，即答案后那段；
3. 第一步的规则单。

要求判官输出结构化 JSON，例如：

```json
{
  "label": "correct/incorrect/unclear",
  "matched_category": "...",
  "reason": "一句话理由"
}
```

对所有 `inference` 题 × 4 个条件跑一遍，逐题存下来。`reason` 很关键，人工抽查时读这一句话会非常快。

### 第四步：人工抽查纠错

这一步是站得住的核心，不是全标，而是按风险抽：

- **必看：**所有 `UNCLEAR` + 所有判官判 `incorrect` 的样本。这些是会扣某个条件分数的关键判定。一旦误判成 `incorrect`，就会不公平地拉低某个条件的 accuracy。
- **抽看：**判 `correct` 的样本里随机抽 15-20%，抓“假阳性”，即判官把不对的判成对了。
- **双人独立：**你和 LST2 各自独立判同一小批，比如 15-20 题，计算你们俩的一致率，以及你们和判官的一致率。
- **改掉判错的样本：**人工标的是最终标签。记录“抽查了百分之多少、人机一致率多少”。

### 第五步：在论文里诚实写明

可以写成：

> 采用本地 LLM-as-judge 做初筛，再人工校对全部边界 case 及抽查 X%，人机一致率 Y%，双标注者一致率 Z%。

这段不但不是弱点，反而正好呼应 SOTA 里 Turpin / Lanham 那条“模型输出不一定可信”的线：你们用人工兜住了判官，说明你们没把另一个模型当成无条件的真理。这是加分项。

### 按风险抽查的本质

> 把上百题逐条手判压缩成“只精修高风险的几十条 + 抽查一部分”，省力但可信度不掉。

你朋友的直觉对，关键就是：

- 判官放“初筛”；
- 人工放“终审”。

---

## 五、判官怎么跑：是不是在 HPC

是的，在 HPC 跑。

原因有两个：

1. 判官是一个 32B 的大模型，需要 GPU。
2. 数据要留在本地，不能传外部 API。

但不要把“在 HPC 跑”吓到。判官本质上又是一个推理任务，和 DS 现在跑 VLM 是同一类活，只是它不看图、只读文字。而且能直接复用 DS 已经在用的 Ollama。

### 推荐流程

1. **生成层先把原始输出跑完存好。** 也就是你们现在正在做的。判官是事后处理这些文件，所以不用等标注，也不是 VLM 同学。
2. **在 HPC 上把判官模型拉下来。** DS 已经会用 Ollama，直接执行：

   ```bash
   ollama pull qwen2.5:32b
   ```

   然后和拉 VLM 模型是一个套路。
3. **写一个小脚本 `judge.py`。** 读所有 inference 的原始记录，对每条拼一个判官 prompt：

   ```text
   规则单 + 这题的 gt 结论 + 模型【答案】后那段 -> 调 ollama.chat（纯文字，不传图） -> 让它输出 {label, matched_category, reason} 的 JSON -> 存成 judged_inference.json
   ```

   判官也用 `greedy / temperature=0`。
4. **把它当作一个 SLURM 任务提交。** 和 DS 跑别的任务同样的提交方式。跑完结果落盘。
5. **人工抽查 `judged_inference.json`。** 必看所有 `UNCLEAR` 和判 `incorrect` 的样本，抽看 15-20% 的 `correct` 样本。你和 LST2 各标一份，算一致率，改掉判错的样本。

### 谁来实际跑 `judge.py`

它是纯文本任务，又复用 Ollama，所以让 LST2（他把 LLM 线跑通）或 DS 接都行。

你负责：

- 把规则单和流程定清楚；
- 协调谁来跑；
- 不需要自己去敲 HPC 命令。

但你要理解到这个程度：你能判断进度，知道卡在哪。

---

## 六、完整的跑实验流程

整条流水线拆成四个阶段：

### 阶段 0（已完成）：锁定 `eval_set`

用 `seed=42` 冻结一份题目清单，明确哪些 case、哪些题。

两条线都读同一个文件，这是“同一批考生”的保证。

### 阶段 1：生成原始输出

> 这一步要上 HPC，是这周的主要工作量。

#### LLM 线（LST2）

纯文本 `Qwen2.5-7B`，跑两个条件：

- `direct`
- `cot`

#### VLM 线（DS）

`Qwen2.5-VL` 走 Ollama，跑以下条件：

- 图 + 文 `direct`
- 图 + 文 `cot`
- image-only 消融：`direct / cot`

#### 统一输出要求

两条线对 `yes/no` 和 `inference` 题都跑。每条记录用相同字段名保存：

```text
case_id
question_id
question
qtype
condition
modality
gt_label
gt_answer
raw_output
```

这一步只有原文，不算分。这一点要立刻给 VLM 同学解释清楚：不是等你们定完评分，只要把原始输出存好就能开跑。

### 阶段 2：统一评分

> 在阶段 1 之后做，不阻塞生成。

#### yes/no 题

用同一个共享解析器抠出 `Yes/No`，和 `gt_label` 比较，输出：

```text
对 / 错 / UNCLEAR
```

#### inference 题

走判官 + 人工流程：

1. 判官（本地 `Qwen2.5-32B`，在 HPC 上跑，纯文字不看图）读取每条记录。
2. 结合规则单 + 标准答案，输出：

   ```json
   {
     "label": "correct/incorrect/unclear",
     "reason": "..."
   }
   ```

3. 然后人工抽查所有 `UNCLEAR` 和判错的样本。
4. 抽看 15-20% 判对的样本。
5. 你和 LST2 各标一份，算一致率，改掉判错的样本。

### 阶段 3：统一比较

用一个共享的 `compare` 脚本读取阶段 2 的评分结果。

#### 输出结果表

包含：

- 各条件 × `yes/no` 准确率；
- `inference` 准确率；
- `UNCLEAR` 比例。

#### 统计检验和增益

计算成对 `McNemar`：

- 每条线内：`direct vs cot`；
- 每种 prompt 下：`LLM vs VLM`。

同时计算：

- `CoT gain`
- 视觉增益
- `CoT` 是否放大视觉增益，即 RQ2

#### 消融分析

比较图像-only vs 图 + 文，用来回答 RQ3。

按题型切片，挑几个定性例子。

### 阶段 4：写论文

`results` 部分基本就是把阶段 3 的表、统计和例子填进去。

不要把事情理解成“代码细节很多，所以很乱”。抽象到这四个阶段后，整件事就是一条直线：

1. 阶段 1 是这周的算力大头，需要 HPC；
2. 阶段 2 / 3 是事后处理，属于纯逻辑活；
3. 阶段 4 是你和写作线的主场。
