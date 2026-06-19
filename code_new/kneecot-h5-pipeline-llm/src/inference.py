"""Inference pipeline for the H5 text-only LLM line.

Runs the text-only LLM (default Qwen2.5-7B-Instruct) over the evaluation set
under two prompting conditions -- 'DA' (direct answer) and 'CoT' -- using
GREEDY (deterministic) decoding, and saves the raw model outputs for later
parsing/scoring by evaluation.py.

Prompt keys match the VLM pipeline keys (DA / CoT) so that both pipelines
produce records with the same `prompt_key` values and can be compared directly.

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

# Keys match VLM pipeline: DA = direct answer, CoT = chain-of-thought
PROMPT_TEMPLATES = {"DA": DIRECT_TEMPLATE, "CoT": COT_TEMPLATE}

# CoT outputs are long (four reasoning steps); direct answers are short.
MAX_NEW_TOKENS = {"DA": 128, "CoT": 1024}


def checkpoint_path(final_path: str) -> str:
    """Return the sidecar checkpoint filename while keeping final filename unchanged."""
    base, _ = os.path.splitext(final_path)
    return base + "_checkpoint.json"


def load_checkpoint(path: str):
    """Load existing checkpoint; return (results_list, done_keys_set)."""
    if not path or not os.path.exists(path):
        return [], set()
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    done = {(r["case_id"], r["question"], r["prompt_key"]) for r in results}
    print(f"  [RESUME] Loaded {len(results)} completed records from checkpoint.")
    return results, done


def save_checkpoint(results: list, path: str):
    """Overwrite the checkpoint with current results."""
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


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
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy -> deterministic & reproducible
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_inference(items, model, tokenizer,
                  prompt_modes=("DA", "CoT"), checkpoint_file=None):
    """Run every (item x prompt_key) pair; return a list of result records."""
    results, done_keys = load_checkpoint(checkpoint_file)
    total = len(items) * len(prompt_modes)
    done = len(results)

    for item in items:
        for mode in prompt_modes:
            key = (item["case_id"], item["question"], mode)
            if key in done_keys:
                done += 1
                continue

            prompt = PROMPT_TEMPLATES[mode].format(
                findings=item["findings"], question=item["question"]
            )
            raw = generate(
                model, tokenizer, prompt,
                max_new_tokens=MAX_NEW_TOKENS[mode]
            )
            rec = dict(item)
            rec["prompt_key"] = mode   # aligned with VLM pipeline
            rec["raw_output"] = raw
            results.append(rec)
            done_keys.add(key)
            done += 1
            print(f"[{done}/{total}] {item['case_id']} | {item['qtype']} | {mode}")

            # save after EVERY completed pair
            save_checkpoint(results, checkpoint_file)

    return results


def save_results(results, path):
    """Save raw results as UTF-8 JSON (keeps Chinese readable)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} records -> {path}")

    cp = checkpoint_path(path)
    if os.path.exists(cp):
        os.remove(cp)
        print(f"Checkpoint removed: {cp}")
