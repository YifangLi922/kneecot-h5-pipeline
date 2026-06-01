"""Prompt templates for the H5 text-only LLM line.

Two conditions are compared on the SAME questions:
  - DIRECT_TEMPLATE: the model answers directly (baseline).
  - COT_TEMPLATE: the model is forced through a four-step structured reasoning
    process before answering.

The four steps mirror the clinical reasoning paradigm in the KneeCoT expert
annotations (CoT_2): systematic observation -> interpretation & verification ->
anatomical structure analysis -> diagnostic reasoning & verification.

NOTE: this is the TEXT-ONLY line, so Step 1 reads the free-text MR findings
("MR表现") rather than observing an image. The VLM line keeps Step 1 as image
observation; the other three steps are shared.

Both templates expose two placeholders: {findings} and {question}.
Both force the answer to end with the marker 【答案】 so evaluation.py can
parse it reliably, and ask yes/no questions to be answered as Yes / No.
"""

DIRECT_TEMPLATE = """你是一位资深骨骼肌肉放射科医生。下面给出一份膝关节 MR 表现和一个相关问题。
请根据 MR 表现直接回答问题，不要写出推理过程。

- 若为是非类问题：请在【答案】后只回答 Yes 或 No。
- 若为推理类问题：请在【答案】后给出明确结论及简要依据。

【MR 表现】
{findings}

【问题】
{question}

【答案】"""


COT_TEMPLATE = """你是一位资深骨骼肌肉放射科医生。请根据下面的膝关节 MR 表现，按以下四个步骤进行系统推理，然后回答问题。

步骤一 系统性梳理（Systematic Observation）
按解剖部位（半月板、韧带、骨与软骨、关节腔与滑膜、其他结构如脂肪垫与软组织）逐项梳理 MR 表现中提到的关键征象，记录每个部位的信号、形态与连续性。

步骤二 解读与核对（Interpretation and Verification）
对每条征象判断属于正常还是异常，并说明该改变通常提示什么（如 T2WI 高信号提示水肿或损伤），核对前后是否一致。

步骤三 解剖结构分析（Anatomical Structure Analysis）
按系统逐一分析：
3.1 半月板：形态是否完整、高信号是否达关节面、损伤程度。
3.2 韧带：前/后交叉韧带、内/外侧副韧带的连续性与信号。
3.3 骨与软骨：骨髓信号（水肿/挫伤）、关节面软骨是否光整。
3.4 关节腔与滑膜：有无积液、滑膜情况。
3.5 其他结构：髌下脂肪垫、关节周围软组织。

步骤四 诊断推理与核对（Diagnostic Reasoning and Verification）
综合以上分析推导出针对问题的结论；如适用，简要排除主要鉴别诊断；自检结论是否由前述证据支持。

最后在【答案】后给出对问题的明确回答：
- 是非类问题：回答 Yes 或 No。
- 推理类问题：给出明确结论及推理依据。

【MR 表现】
{findings}

【问题】
{question}

请依次完成步骤一至步骤四，最后给出【答案】。"""
