## Project Setup
At the beginning of the project, our group created a GitHub team and repository for collaboration.

## Initial Topic Discussion
As a group, we discussed several possible project topics. At this stage, the final topic has not yet been assigned or confirmed. However, we have started to compare different options based on their feasibility, relevance to the course, available resources, and the amount of technical work required. We plan to choose Knee MRI Diagnosis or Maritime Trajectories as our topic.

This week, we discussed how to narrow the scope of our knee MRI diagnosis project so that it remains feasible within one semester while still addressing the main research questions. We agreed to center the project on comparing direct-answer prompting and Chain-of-Thought prompting, with a particular focus on whether performance gains come from structured reasoning alone or from access to visual information.

In our discussion, we decided that the main comparison should be between text-only language models and multimodal vision-language models under similar prompting settings. We also agreed that VQA-style evaluation should be the primary target of the project because it provides a clearer and more manageable setup than full diagnostic report generation for testing the hypotheses in H5 and H6.

We further discussed project scope and concluded that fine-tuning should remain an optional extension rather than a core requirement. As a result, our main plan is to first build a solid prompting-based evaluation pipeline and then consider a smaller fine-tuning experiment only if time and compute resources allow.

We also organized the structure of the related work section. We agreed to divide it into four thematic parts: Chain-of-Thought reasoning and its limitations, medical language models and diagnostic reasoning, medical vision-language models, and Chinese medical NLP, so that the literature review directly supports our research gap and motivates our final experimental design.

## Week 2
Last week, our group focused on completing the writing for the Introduction + Motivation section and the State-of-the-Art section of the project report. The State-of-the-Art section was organized into three main parts: Chain-of-Thought prompting and clinical reasoning, Medical VQA and medical LLMs/VLMs, and Visual grounding and multimodal medical reasoning. (We actually chage the frame a little bit according to the papers we read to form a complete and appropirate related works part.)

More specifically, based on the conclusions from our previous discussion, we finalized the writing direction and overall structure for the Introduction and Motivation sections and completed the corresponding drafts. In parallel, team members extensively reviewed related literature, discussed papers that were highly relevant to our project, and incorporated these references into the State-of-the-Art section.

On Friday, the group met together to integrate each member’s contributions, revise the report as a whole, and complete the submission of Milestone 1. In addition, we discussed whether suitable figures should be added to improve the presentation of the report (we decided to add figures later because we need to involve more parts in the figures) and also planned the tasks and objectives for the following week.

## Week 3
This week, our team further refined the project design and divided the work into two parallel research tracks.

The first group focused on the text-only language model (LLM) pipeline. The objective was to investigate whether Chain-of-Thought (CoT) prompting can improve diagnostic reasoning when only textual MRI findings are provided. The group implemented the preprocessing, prompting, inference, and evaluation pipeline, and successfully tested the code on dozens of cases from the KneeCoT dataset. Preliminary experiments confirmed that the pipeline can correctly load the data, generate predictions, and compute evaluation metrics.

The second group focused on the vision-language model (VLM) pipeline. This branch aims to explore the contribution of visual information in medical reasoning tasks by incorporating MRI images into the diagnostic process. The group worked on model setup, image processing, and multimodal inference, and also conducted initial tests on dozens of samples to verify the functionality of the framework.

In addition to the implementation work, the team discussed the overall experimental design and the structure of the methodology section. We agreed to compare Direct Prompting and Chain-of-Thought Prompting across both text-only and multimodal settings, allowing us to investigate the individual and combined effects of reasoning strategies and visual information.

## Week 4

This week, our group focused on developing and refining the Round 2 methodology. 

For the text-only LLM pipeline, we use Qwen2.5-7B-Instruct. The model receives only the free-text MR findings and the question. We deliberately exclude the diagnostic impression and structured diagnostic labels because these fields may directly contain the correct answer. This leakage prevention step became an important part of our methodology, since including those fields would make the evaluation invalid.

For the multimodal VLM pipeline, we use Qwen2.5-VL as the primary vision-language model. The VLM receives the same textual input as the LLM, together with selected MRI slice images from the corresponding case. We also clarified the image preprocessing procedure: MRI volumes are loaded from `.nii` files, five slices are selected using a fixed rule, and CLAHE is applied to enhance image contrast. This makes the visual input more reproducible and avoids manually selecting slices that may favor the model.

We also refined our matched 2 × 2 experimental design. The two factors are model modality and prompting strategy. The modality factor compares a text-only LLM with a multimodal VLM, while the prompting factor compares direct prompting with structured Chain-of-Thought prompting. This produces four main conditions: LLM direct, LLM CoT, VLM direct, and VLM CoT. All four conditions are evaluated on the same sampled examples, using seed 42, so that the comparisons are matched and fair.

Another important part of this week’s work was improving the prompt templates and answer parsing strategy. We decided to use Chinese prompts because the KneeCoT dataset is in Chinese. The direct prompt asks the model to answer directly, while the CoT prompt asks the model to follow a four-step clinical reasoning structure: systematic observation, interpretation and verification, anatomical structure analysis, and diagnostic reasoning with final verification. We also use a fixed final answer marker so that answers can be extracted consistently from model outputs.

For evaluation, we clarified that yes/no questions form the main quantitative accuracy backbone. Inference questions are included as a pilot diagnostic-inference analysis because their answers are more open-ended and cannot be evaluated by exact string matching. For these questions, we extract the final diagnostic conclusion and compare it with the ground-truth conclusion using rule-based matching. We also plan to report yes/no accuracy and inference accuracy separately, since the two task types differ in difficulty.

Finally, we updated the project documentation and created a cleaner pipeline figure for Round 2. The figure summarizes the process from dataset input, leakage prevention, task filtering, sampling, model inference, prompt comparison, answer parsing, and evaluation. This helped us make the methodology easier to understand and more visually organized.
