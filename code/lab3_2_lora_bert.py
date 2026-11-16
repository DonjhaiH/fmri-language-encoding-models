#!/usr/bin/env python3
"""
lab3.2_lora_bert.py
===================
Parameter-efficient fine-tuning of bert-base-uncased with LoRA
(Low-Rank Adaptation) for Masked Language Modeling on fMRI podcast
train stories.

This is Stage C of the Lab 3.2 Part 1 pipeline.  The saved adapter
is then passed to lab3.2_extract_bert_features.py (--variant lora --lora_path)
for Stage D feature extraction.

Why LoRA?
  Full fine-tuning updates all ~110M BERT parameters, requiring large GPU
  memory and risking catastrophic forgetting of pre-trained knowledge.
  LoRA freezes the base model and inserts trainable low-rank matrices
  A ∈ R^(d×r) and B ∈ R^(r×d) into selected linear layers.  The adapted
  weight is W' = W + (alpha/r)·BA.  With r=16 only ~300K parameters are
  trained instead of 110M, using far less memory and converging faster.

  We apply LoRA to the query and value projection matrices in every BERT
  self-attention layer (common choice from the original LoRA paper).

Reference:
  Hu et al. (2022) "LoRA: Low-Rank Adaptation of Large Language Models"
  https://arxiv.org/abs/2106.09685

Requires:
  pip install peft

Typical usage on Bridges2 GPU node (run from code/):
    python lab3.2_lora_bert.py \\
        --raw_text_pkl /path/to/project/data/raw_text.pkl \\
        --output_dir ${HOME}/bert_lora_out \\
        --epochs 5 --batch_size 32 --lora_r 16 --lora_alpha 32 --fp16
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import torch
from torch.utils.data import Dataset

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CODE_DIR)
# raw_text.pkl stores ridge_utils.DataSequence; unpickling requires repo root on sys.path.
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _CODE_DIR)

DEFAULT_RAW_TEXT = (
    "/path/to/project/data/raw_text.pkl"
)
HF_SPLIT_REPO = "YOUR_HF_USER/train_test_pkl"
BERT_BASE = "bert-base-uncased"


# ---------------------------------------------------------------------------
# Story split helper (same logic as lab3.2_train_bert_mlm.py)
# ---------------------------------------------------------------------------

def _load_train_stories(
    stories_json: str | None,
    hf_repo: str,
    raw_stories: dict,
) -> list[str]:
    default_json = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "stories_split.json"
    )
    json_path = stories_json or (
        default_json if os.path.isfile(default_json) else None
    )

    if json_path and os.path.isfile(json_path):
        with open(json_path) as f:
            d = json.load(f)
        train_stories = d["train"]
        print(f"[split] loaded from {json_path}")
    else:
        try:
            from huggingface_hub import hf_hub_download

            remote_json = hf_hub_download(
                repo_id=hf_repo, repo_type="dataset", filename="stories_split.json"
            )
            with open(remote_json) as f:
                d = json.load(f)
            train_stories = d["train"]
            with open(default_json, "w") as f:
                json.dump(
                    {"train": d["train"], "test": d.get("test", [])}, f, indent=2
                )
            print(f"[split] loaded stories_split.json from Hub → cached {default_json}")
        except Exception as exc:
            print(
                "ERROR: need code/stories_split.json (or pass --stories_json). "
                f"Hub stories_split.json fetch failed ({type(exc).__name__}: {exc}).\n"
                "This repo does not auto-download X_tr_bow.pkl / X_tst_bow.pkl (~12 GB).\n"
                "Generate JSON locally: python lab3.2_make_stories_split.py",
                file=sys.stderr,
            )
            sys.exit(1)

    available = set(raw_stories.keys())
    train_stories = [s for s in train_stories if s in available]
    print(f"[split] {len(train_stories)} train stories")
    return train_stories


# ---------------------------------------------------------------------------
# Dataset (identical to lab3.2_train_bert_mlm.py)
# ---------------------------------------------------------------------------

class StoryMLMDataset(Dataset):
    def __init__(self, chunks: list[list[int]]):
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        ids = self.chunks[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
        }


def build_dataset(
    train_stories: list[str],
    raw_stories: dict,
    tokenizer,
    max_length: int,
) -> StoryMLMDataset:
    all_ids: list[int] = []
    for story in train_stories:
        enc = tokenizer(
            raw_stories[story].data,
            is_split_into_words=True,
            add_special_tokens=False,
        )
        all_ids.extend(enc["input_ids"])

    chunk_size = max_length - 2
    chunks: list[list[int]] = []
    for start in range(0, len(all_ids), chunk_size):
        chunk = all_ids[start : start + chunk_size]
        if len(chunk) < 10:
            continue
        chunks.append([tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id])

    print(f"[dataset] {len(all_ids):,} tokens → {len(chunks):,} chunks")
    return StoryMLMDataset(chunks)


def _latest_checkpoint_dir(output_dir: str) -> str | None:
    """If output_dir contains Trainer checkpoint-* folders, return the latest (by step)."""
    if not os.path.isdir(output_dir):
        return None
    sub = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith("checkpoint-")
        and os.path.isdir(os.path.join(output_dir, name))
    ]
    if not sub:
        return None

    def _step(path: str) -> int:
        try:
            return int(os.path.basename(path).split("-")[-1])
        except ValueError:
            return -1

    return max(sub, key=_step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--raw_text_pkl", default=DEFAULT_RAW_TEXT)
    p.add_argument("--stories_json", default=None)
    p.add_argument(
        "--output_dir", required=True,
        help="Directory to save the LoRA adapter weights",
    )
    p.add_argument("--epochs", type=int, default=5,
                   help="More epochs than full FT is fine — few params to overfit")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (can be larger than full FT; default 3e-4)",
    )
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--mlm_prob", type=float, default=0.15)
    p.add_argument(
        "--lora_r", type=int, default=16,
        help="LoRA rank r (higher = more capacity, more trainable params)",
    )
    p.add_argument(
        "--lora_alpha", type=int, default=32,
        help="LoRA scaling alpha (effective scale = alpha/r)",
    )
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument(
        "--target_modules", default="query,value",
        help="Comma-separated BERT attention module names to apply LoRA to "
             "(default: 'query,value')",
    )
    p.add_argument(
        "--save_steps", type=int, default=500,
        help="Checkpoint every N steps (guards against Slurm wall-clock kill)",
    )
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--push_to_hub", default=None,
        help="HF model repo to push adapter to (e.g. 'username/bert-fmri-lora')",
    )
    args = p.parse_args()

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("ERROR: peft is not installed.  Run:  pip install peft")
        sys.exit(1)

    from transformers import (
        BertForMaskedLM,
        BertTokenizerFast,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # Load data
    print(f"[data] loading {args.raw_text_pkl} …")
    with open(args.raw_text_pkl, "rb") as f:
        raw_stories = pickle.load(f)

    train_stories = _load_train_stories(args.stories_json, HF_SPLIT_REPO, raw_stories)

    tokenizer = BertTokenizerFast.from_pretrained(BERT_BASE)
    dataset = build_dataset(train_stories, raw_stories, tokenizer, args.max_length)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
    )

    # ----- Build LoRA model -----
    base_model = BertForMaskedLM.from_pretrained(BERT_BASE)
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"[model] base BertForMaskedLM params: {total_params:,}")

    target_modules = [m.strip() for m in args.target_modules.split(",")]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        # No task_type needed when target_modules are specified explicitly
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ----- Training -----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=50,
        fp16=args.fp16,
        report_to="none",
        push_to_hub=args.push_to_hub is not None,
        hub_model_id=args.push_to_hub,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
    )

    print(
        f"[training] LoRA  r={args.lora_r}  alpha={args.lora_alpha}  "
        f"targets={target_modules}  epochs={args.epochs}  "
        f"batch={args.batch_size}  lr={args.lr}  fp16={args.fp16} …"
    )
    resume_ckpt = _latest_checkpoint_dir(args.output_dir)
    if resume_ckpt:
        print(f"[training] resuming from {resume_ckpt} (transformers v5 has no overwrite_output_dir)")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save adapter only (not the full base model — much smaller)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[saved] LoRA adapter → {args.output_dir}")
    print(
        f"        use with: lab3.2_extract_bert_features.py "
        f"--variant lora --lora_path {args.output_dir}"
    )

    if args.push_to_hub:
        # transformers v5: push_to_hub args are keyword-only (repo_id=…)
        model.push_to_hub(repo_id=args.push_to_hub)
        tokenizer.push_to_hub(repo_id=args.push_to_hub)
        print(f"[hub] pushed adapter to {args.push_to_hub}")


if __name__ == "__main__":
    main()
