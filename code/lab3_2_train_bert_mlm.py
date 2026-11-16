
#!/usr/bin/env python3
"""
lab3.2_train_bert_mlm.py
========================
Fine-tune bert-base-uncased with Masked Language Modeling (MLM)
on the fMRI podcast *train* stories only (no test-set leakage).

This is Stage B of the Lab 3.2 Part 1 pipeline.  The saved checkpoint
is then passed to lab3.2_extract_bert_features.py (--variant ft --model_path)
for Stage D feature extraction.

MLM training objective:
  Randomly mask 15 % of input tokens; the model learns to predict the
  original tokens from context.  This domain-adapts BERT's language
  model to the vocabulary and style of neuroscience podcast transcripts,
  which differs from its Wikipedia/BookCorpus pre-training corpus.

Typical usage on Bridges2 GPU node (GPU-shared partition; run from code/):
    python lab3.2_train_bert_mlm.py \\
        --raw_text_pkl /path/to/project/data/raw_text.pkl \\
        --output_dir ${HOME}/bert_mlm_out \\
        --epochs 3 --batch_size 16 --fp16

Slurm sbatch tip:
    Add --save_steps 200 so checkpoints survive Slurm wall-clock kills.
    If output_dir already contains checkpoint-*, training auto-resumes from
    the latest (transformers v5 removed TrainingArguments.overwrite_output_dir).
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
# Story split helper (shared logic with lab3.2_extract_bert_features.py)
# ---------------------------------------------------------------------------

def _load_train_stories(
    stories_json: str | None,
    hf_repo: str,
    raw_stories: dict,
) -> list[str]:
    """Return train story names filtered to those present in raw_stories."""
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
        # Small file only — never auto-download multi-GB BoW pkls.
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
    print(f"[split] {len(train_stories)} train stories (in raw_text.pkl)")
    return train_stories


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StoryMLMDataset(Dataset):
    """Non-overlapping token chunks drawn from the train-story transcripts.

    Each item is a dict with 'input_ids' and 'attention_mask'.
    DataCollatorForLanguageModeling handles the MLM masking at collation time.
    """

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
    """Tokenize all train stories and split into fixed-length chunks.

    Stories are concatenated (no special separator) then divided into chunks
    of (max_length - 2) tokens; [CLS] and [SEP] are prepended/appended.
    Very short tail chunks (<10 tokens) are discarded.
    """
    all_ids: list[int] = []
    for story in train_stories:
        words = raw_stories[story].data
        enc = tokenizer(
            words,
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

    print(
        f"[dataset] {len(all_ids):,} tokens → {len(chunks):,} chunks "
        f"(max_len={max_length})"
    )
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
    p.add_argument(
        "--stories_json", default=None,
        help="Pre-computed split JSON (skips HF download)",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Directory for checkpoints and final model",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument(
        "--max_length", type=int, default=128,
        help="Token chunk length (default 128)",
    )
    p.add_argument(
        "--mlm_prob", type=float, default=0.15,
        help="Fraction of tokens to mask (default 0.15)",
    )
    p.add_argument(
        "--save_steps", type=int, default=500,
        help="Save a checkpoint every N steps (important for long Slurm jobs)",
    )
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument(
        "--fp16", action="store_true",
        help="Mixed-precision training (requires GPU with fp16 support)",
    )
    p.add_argument(
        "--push_to_hub", default=None,
        help="HF model repo to push final model to (e.g. 'username/bert-fmri-ft')",
    )
    args = p.parse_args()

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

    model = BertForMaskedLM.from_pretrained(BERT_BASE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] BertForMaskedLM  total params: {n_params:,}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
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
        f"[training] MLM fine-tuning  "
        f"epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}  "
        f"mlm_prob={args.mlm_prob}  fp16={args.fp16} …"
    )
    resume_ckpt = _latest_checkpoint_dir(args.output_dir)
    if resume_ckpt:
        print(f"[training] resuming from {resume_ckpt} (transformers v5 has no overwrite_output_dir)")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save final model (pass --model_path here to lab3.2_extract_bert_features.py)
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[saved] final model → {final_dir}")
    print(f"        use with: lab3.2_extract_bert_features.py --variant ft --model_path {final_dir}")

    if args.push_to_hub:
        trainer.push_to_hub()
        print(f"[hub] pushed to {args.push_to_hub}")


if __name__ == "__main__":
    main()
