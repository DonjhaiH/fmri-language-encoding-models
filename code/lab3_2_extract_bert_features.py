#!/usr/bin/env python3
"""
lab3.2_extract_bert_features.py
================================
Extract word-level BERT embeddings for fMRI voxel prediction.

Supports three BERT variants:
  --variant pt    : pre-trained bert-base-uncased (no fine-tuning)
  --variant ft    : fine-tuned BertForMaskedLM checkpoint
  --variant lora  : LoRA adapter merged into bert-base-uncased

Pipeline (mirrors BoW pipeline in preprocessing.py):
  1. BERT forward pass → word-level hidden states (768-dim per word)
  2. downsample_word_vectors  (Lanczos interp to TR grid)
  3. Trim first 5 / last 10 TRs
  4. make_delayed(delays=[1,2,3,4])
  5. Save X_tr_bert_{variant}.pkl / X_tst_bert_{variant}.pkl + meta JSON

Typical usage on Bridges2 GPU node (run from code/; paths match lab3.2_sbatch_*.sh):
    # Stage A — pre-trained
    python lab3.2_extract_bert_features.py \\
        --variant pt \\
        --output_dir ${HOME}/bert_features

    # Stage D — from fine-tuned checkpoint
    python lab3.2_extract_bert_features.py \\
        --variant ft \\
        --model_path ${HOME}/bert_mlm_out/final \\
        --output_dir ${HOME}/bert_features

    # Stage D — from LoRA adapter
    python lab3.2_extract_bert_features.py \\
        --variant lora \\
        --lora_path ${HOME}/bert_lora_out \\
        --output_dir ${HOME}/bert_features
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import torch

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CODE_DIR)
# raw_text.pkl stores ridge_utils.DataSequence; unpickling requires repo root on sys.path.
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _CODE_DIR)

try:
    from tqdm.auto import tqdm as _tqdm
    tqdm = _tqdm  # noqa: A001 — keep name tqdm for loop wrapper

    def _tqdm_write(msg: str) -> None:
        _tqdm.write(msg)
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

    def _tqdm_write(msg: str) -> None:
        print(msg, flush=True)

from preprocessing import downsample_word_vectors, make_delayed

DEFAULT_RAW_TEXT = (
    "/path/to/project/data/raw_text.pkl"
)
HF_SPLIT_REPO = "YOUR_HF_USER/train_test_pkl"
TRIM_START = 5
TRIM_END = 10
DELAYS = [1, 2, 3, 4]
BERT_DIM = 768
CHUNK_WORDS = 250  # ~250 words ≈ ≤512 BERT subword tokens for typical podcast text


# ---------------------------------------------------------------------------
# Story split
# ---------------------------------------------------------------------------

def _load_story_split(
    stories_json: str | None,
    hf_repo: str,
    available: set,
) -> tuple[list[str], list[str]]:
    """Return (train_stories, test_stories) filtered to available stories.

    Priority:
      1. stories_json argument (pre-computed JSON)
      2. code/stories_split.json if it exists next to this script
      3. stories_split.json from HF dataset repo (small file only)

    Does **not** download multi-GB BoW pickles; add JSON if Hub has no split file.
    """
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
        test_stories = d["test"]
        print(
            f"[split] loaded from {json_path}: "
            f"{len(train_stories)} train / {len(test_stories)} test"
        )
    else:
        from huggingface_hub import hf_hub_download

        # Prefer a small split file on the Hub if the dataset provides one.
        try:
            remote_json = hf_hub_download(
                repo_id=hf_repo, repo_type="dataset", filename="stories_split.json"
            )
            with open(remote_json) as f:
                d = json.load(f)
            train_stories = d["train"]
            test_stories = d["test"]
            with open(default_json, "w") as f:
                json.dump({"train": train_stories, "test": test_stories}, f, indent=2)
            print(
                f"[split] loaded stories_split.json from Hub → cached {default_json}: "
                f"{len(train_stories)} train / {len(test_stories)} test"
            )
        except Exception as exc:
            raise RuntimeError(
                "Missing story split: add code/stories_split.json or pass --stories_json. "
                f"Could not fetch stories_split.json from Hub ({type(exc).__name__}: {exc}). "
                "This project does not download X_tr_bow.pkl / X_tst_bow.pkl (~12 GB)."
            ) from exc

    train_stories = [s for s in train_stories if s in available]
    test_stories = [s for s in test_stories if s in available]
    print(
        f"[split] after filtering to raw_text.pkl: "
        f"{len(train_stories)} train / {len(test_stories)} test"
    )
    return train_stories, test_stories


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build_model(
    variant: str,
    model_path: str | None,
    lora_path: str | None,
    device: str,
):
    """Load the appropriate BertModel (encoder only) for feature extraction.

    Returns (model: BertModel, tokenizer: BertTokenizerFast).
    For all variants the returned model is a BertModel instance so that
    hidden_states are accessible uniformly via output_hidden_states=True.
    """
    from transformers import BertForMaskedLM, BertModel, BertTokenizerFast

    base_name = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(base_name)

    if variant == "pt":
        model = BertModel.from_pretrained(base_name)

    elif variant == "ft":
        if model_path is None:
            raise ValueError("--model_path is required for --variant ft")
        # Load fine-tuned BertForMaskedLM, then extract the encoder sub-module
        mlm_model = BertForMaskedLM.from_pretrained(model_path)
        model = mlm_model.bert  # BertModel
        del mlm_model

    elif variant == "lora":
        if lora_path is None:
            raise ValueError("--lora_path is required for --variant lora")
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft not installed — run: pip install peft")
        # Load base BertForMaskedLM, apply LoRA adapter, merge, extract encoder
        base = BertForMaskedLM.from_pretrained(base_name)
        peft_model = PeftModel.from_pretrained(base, lora_path)
        merged = peft_model.merge_and_unload()  # LoRA merged into base weights
        model = merged.bert  # BertModel
        del base, peft_model, merged

    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    model.eval()
    model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Word-level embedding extraction
# ---------------------------------------------------------------------------

def _story_word_embeddings(
    words: list[str],
    model,
    tokenizer,
    device: str,
    chunk_words: int = CHUNK_WORDS,
    layer: int = -1,
) -> np.ndarray:
    """Return float32 array of shape (n_words, 768).

    Each word embedding is the mean of its subword-token hidden states
    from the specified hidden layer (default: last layer).

    The story is split into non-overlapping chunks of ``chunk_words`` words
    to stay within BERT's 512-token limit.  The context at chunk boundaries
    is limited; increase chunk_words (or implement an overlap strategy) if
    boundary quality matters.
    """
    n_words = len(words)
    all_vecs = np.zeros((n_words, BERT_DIM), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_words, chunk_words):
            chunk = words[start : start + chunk_words]

            # Tokenize as a pre-split word list; the encoding tracks
            # which token belongs to which word via word_ids().
            enc = tokenizer(
                chunk,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # hidden_states: tuple[(n_layers+1) x Tensor(1, seq_len, 768)]
            hidden = outputs.hidden_states[layer][0].cpu().float().numpy()

            # Map token positions → word indices
            word_ids = enc.word_ids(batch_index=0)  # list[int | None]

            for wi in range(len(chunk)):
                tok_idxs = [ti for ti, wid in enumerate(word_ids) if wid == wi]
                if tok_idxs:
                    all_vecs[start + wi] = hidden[tok_idxs].mean(axis=0)
                # else: word gets a zero vector (very rare)

    return all_vecs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_and_save(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # Load raw stories
    print(f"[data] loading raw_text.pkl from {args.raw_text_pkl} …")
    with open(args.raw_text_pkl, "rb") as f:
        raw_stories = pickle.load(f)
    print(f"[data] {len(raw_stories)} stories in raw_text.pkl")

    train_stories, test_stories = _load_story_split(
        args.stories_json, HF_SPLIT_REPO, set(raw_stories.keys())
    )
    all_stories = train_stories + test_stories
    train_set = set(train_stories)

    # Build model
    model, tokenizer = _build_model(
        args.variant, args.model_path, args.lora_path, device
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] BertModel params: {n_params:,}  (variant={args.variant})")

    # ----- Step 1: BERT word embeddings -----
    print("[step 1] extracting BERT word embeddings …")
    word_vectors: dict[str, np.ndarray] = {}
    for story in tqdm(all_stories, desc="step1/stories", unit="story", dynamic_ncols=True):
        words = raw_stories[story].data
        vecs = _story_word_embeddings(
            words, model, tokenizer, device,
            chunk_words=args.chunk_words,
            layer=args.layer,
        )
        word_vectors[story] = vecs
        _tqdm_write(f"  {story}: {len(words)} words → {vecs.shape}")

    # Free GPU memory before numpy operations
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----- Step 2: Lanczos downsample to TR grid -----
    print("[step 2] Lanczos downsampling to TR grid …")
    tr_word_vecs = downsample_word_vectors(all_stories, word_vectors, raw_stories)

    # ----- Step 3: Trim -----
    print(f"[step 3] trimming [{TRIM_START}:{-TRIM_END}] TRs …")
    trimmed: dict[str, np.ndarray] = {
        story: mat[TRIM_START:-TRIM_END, :]
        for story, mat in tr_word_vecs.items()
    }

    # ----- Step 4: make_delayed -----
    print(f"[step 4] make_delayed(delays={DELAYS}) …")
    X_tr: dict[str, np.ndarray] = {}
    X_tst: dict[str, np.ndarray] = {}
    for story, mat in trimmed.items():
        delayed = make_delayed(mat, DELAYS).astype(np.float32)
        if story in train_set:
            X_tr[story] = delayed
        else:
            X_tst[story] = delayed

    # Sanity-check a few stories
    for story, mat in list(X_tr.items())[:2]:
        print(f"  X_tr[{story}]: {mat.shape}")
    for story, mat in list(X_tst.items())[:2]:
        print(f"  X_tst[{story}]: {mat.shape}")
    expected_cols = BERT_DIM * len(DELAYS)
    assert next(iter(X_tr.values())).shape[1] == expected_cols, (
        f"Expected {expected_cols} columns, got {next(iter(X_tr.values())).shape[1]}"
    )

    # ----- Save -----
    os.makedirs(args.output_dir, exist_ok=True)
    v = args.variant
    tr_path = os.path.join(args.output_dir, f"X_tr_bert_{v}.pkl")
    tst_path = os.path.join(args.output_dir, f"X_tst_bert_{v}.pkl")

    with open(tr_path, "wb") as f:
        pickle.dump(X_tr, f, protocol=4)
    print(f"[saved] {tr_path}")

    with open(tst_path, "wb") as f:
        pickle.dump(X_tst, f, protocol=4)
    print(f"[saved] {tst_path}")

    # ----- Meta JSON -----
    try:
        import transformers as _tr
        transformers_version = _tr.__version__
    except Exception:
        transformers_version = "unknown"

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    meta = {
        "variant": args.variant,
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "layer": args.layer,
        "chunk_words": args.chunk_words,
        "trim_start": TRIM_START,
        "trim_end": TRIM_END,
        "delays": DELAYS,
        "bert_dim": BERT_DIM,
        "output_cols": BERT_DIM * len(DELAYS),
        "train_stories": train_stories,
        "test_stories": test_stories,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "git_commit": git_commit,
    }
    meta_path = os.path.join(args.output_dir, f"bert_meta_{v}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] {meta_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--variant", choices=["pt", "ft", "lora"], default="pt",
        help="pt=pretrained, ft=fine-tuned MLM checkpoint, lora=LoRA adapter",
    )
    p.add_argument(
        "--model_path", default=None,
        help="Path to fine-tuned BertForMaskedLM checkpoint (for --variant ft)",
    )
    p.add_argument(
        "--lora_path", default=None,
        help="Path to saved LoRA adapter directory (for --variant lora)",
    )
    p.add_argument(
        "--raw_text_pkl", default=DEFAULT_RAW_TEXT,
        help="Path to raw_text.pkl on Bridges2",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Directory to write output pkl files and meta JSON",
    )
    p.add_argument(
        "--stories_json", default=None,
        help="JSON {\"train\": [...], \"test\": [...]}; if unset, uses stories_split.json next to this script",
    )
    p.add_argument(
        "--chunk_words", type=int, default=CHUNK_WORDS,
        help=f"Words per BERT forward-pass chunk (default {CHUNK_WORDS}; keep ≤512 tokens)",
    )
    p.add_argument(
        "--layer", type=int, default=-1,
        help="Hidden layer index to extract from (-1=last, -2=second-to-last, etc.)",
    )
    args = p.parse_args()
    extract_and_save(args)


if __name__ == "__main__":
    main()
