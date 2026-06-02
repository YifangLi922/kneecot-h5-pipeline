"""Inference pipeline for the H5 text-only LLM line.

Runs the text-only LLM (default Qwen2.5-7B-Instruct) over the evaluation set
under two prompting conditions -- 'direct' and 'cot' -- using GREEDY
(deterministic) decoding, and saves the raw model outputs for later
parsing/scoring by evaluation.py.

Greedy decoding (do_sample=False) is used on purpose: it removes sampling
randomness so the only thing that differs between conditions is the prompt,
and so results are exactly reproducible.
"""
import json
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from prompts import COT_TEMPLATE, DIRECT_TEMPLATE

PROMPT_TEMPLATES = {"direct": DIRECT_TEMPLATE, "cot": COT_TEMPLATE}

# CoT outputs are long (four reasoning steps); direct answers are short.
MAX_NEW_TOKENS = {"direct": 128, "cot": 1024}


def load_model(model_name="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True):
    """Load tokenizer + model. 4-bit fits a 7B model on a free Colab T4 GPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"device_map": "auto"}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=512):
    """Run one greedy generation and return only the newly generated text."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy -> deterministic & reproducible
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = out[0][inputs.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_inference(items, model, tokenizer, prompt_modes=("direct", "cot")):
    """Run every (item x prompt_mode) pair; return a list of result records."""
    results = []
    total = len(items) * len(prompt_modes)
    done = 0
    for item in items:
        for mode in prompt_modes:
            prompt = PROMPT_TEMPLATES[mode].format(
                findings=item["findings"], question=item["question"]
            )
            raw = generate(
                model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS[mode]
            )
            rec = dict(item)
            rec["prompt_mode"] = mode
            rec["raw_output"] = raw
            results.append(rec)
            done += 1
            print(f"[{done}/{total}] {item['case_id']} | {item['qtype']} | {mode}")
    return results


def save_results(results, path):
    """Save raw results as UTF-8 JSON (keeps Chinese readable)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} records -> {path}")
