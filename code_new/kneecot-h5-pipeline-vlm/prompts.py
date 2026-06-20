"""
prompts.py  –  Chinese VLM prompts (DA, CoT, DA_findings, CoT_findings).

Yes/No answer parsing lives only in code_new/analysis/compare.py (the
shared scoring code) — see that module's docstring. Do not add a parser
copy here; a stale duplicate caused real confusion before.
"""

ANSWER_MARKER = "【答案】"

YESNO_INSTRUCTION = (
    "- 若为是非类问题：请在【答案】后单独另起一行，只写英文单词 Yes 或 No，"
    "不要写中文“是”/“否”，不要附加任何其他文字、标点或解释。"
)

VLM_PROMPTS = {
    "DA": """你是一位资深骨骼肌肉放射科医生。下面给出一组膝关节 MR 图像和一个相关问题。
请根据图像直接回答问题，不要写出推理过程。
""" + YESNO_INSTRUCTION + """
- 若为推理类问题：请在【答案】后给出明确结论及简要依据。
【问题】{question}
【答案】""",

    "CoT": """你是一位资深骨骼肌肉放射科医生。请根据下面给出的膝关节 MR 图像，按以下四个步骤进行系统推理，然后回答问题。

步骤一 系统性观察（Systematic Observation）
... 系统性地观察所给的 MR 图像，按解剖部位逐项描述关键征象。

步骤二 解读与核对（Interpretation and Verification）
对每条征象判断正常还是异常，说明该改变通常提示什么。

步骤三 解剖结构分析（Anatomical Structure Analysis）
3.1 半月板  3.2 韧带  3.3 骨与软骨  3.4 关节腔与滑膜  3.5 其他结构

步骤四 诊断推理与核对（Diagnostic Reasoning and Verification）
综合分析推导结论；自检结论是否由证据支持。

完成以上推理步骤后，最后必须单独另起一行给出【答案】：
""" + YESNO_INSTRUCTION + """
- 若为推理类问题：在【答案】后给出明确结论及依据。
【问题】{question}
请依次完成步骤一至步骤四，最后给出【答案】""",

    "DA_findings": """你是一位资深骨骼肌肉放射科医生。下面给出一组膝关节 MR 图像、对应的 MR 表现文字和一个相关问题。
请结合图像与 MR 表现直接回答问题，不要写出推理过程。
""" + YESNO_INSTRUCTION + """
- 若为推理类问题：请在【答案】后给出明确结论及简要依据。
【MR 表现】{findings}
【问题】{question}
【答案】""",

    "CoT_findings": """你是一位资深骨骼肌肉放射科医生。请根据下面给出的膝关节 MR 图像及其对应的 MR 表现文字，按以下四个步骤进行系统推理，然后回答问题。

步骤一 系统性观察（Systematic Observation）
系统性地观察所给的 MR 图像，并对照所提供的 MR 表现文字，按解剖部位逐项梳理关键征象。

步骤二 解读与核对（Interpretation and Verification）
对每条征象判断正常还是异常，说明该改变通常提示什么。

步骤三 解剖结构分析（Anatomical Structure Analysis）
3.1 半月板  3.2 韧带  3.3 骨与软骨  3.4 关节腔与滑膜  3.5 其他结构

步骤四 诊断推理与核对（Diagnostic Reasoning and Verification）
综合分析推导结论；如适用简要排除主要鉴别诊断；自检结论是否由证据支持。

完成以上推理步骤后，最后必须单独另起一行给出【答案】：
""" + YESNO_INSTRUCTION + """
- 若为推理类问题：在【答案】后给出明确结论及依据。
【MR 表现】{findings}
【问题】{question}
请依次完成步骤一至步骤四，最后给出【答案】。""",
}
