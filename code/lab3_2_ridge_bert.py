#!/usr/bin/env python3
"""
lab3.2_ridge_bert.py
====================
Lab 3.2 Part 1 — Ridge decoding on BERT features (same pattern as 3.1 / BoW notebook).

Loads ``X_tr_bert_{variant}.pkl`` / ``X_tst_bert_{variant}.pkl`` produced by
``lab3.2_extract_bert_features.py``, stacks train stories, fits ``RidgeCV`` on
BOLD responses, evaluates per-voxel Pearson correlation on held-out test stories.

Typical usage (from ``code/`` on Bridges or locally; set ``--data_root`` to your
subject2/3 folder parent):

    python lab3.2_ridge_bert.py \\
        --bert_features_dir ~/bert_features \\
        --variant pt \\
        --data_root /path/to/project/data \\
        --subject subject2

Outputs under ``--output_dir`` (default ``results/`` next to this script):
  - ``ridge_bert_{variant}_{subject}.pkl``  fitted ``RidgeCV``
  - ``ridge_bert_{variant}_{subject}_summary.json``  metrics + per-story CC stats

Stories without matching ``{data_root}/{subject}/{story}.npy`` are skipped.
The same ``INVALID_STORIES`` set as in ``ridge_exploratoin_bow.ipynb`` is applied
(dialogue* / missing BOLD in standard release).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Any

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CODE_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _CODE_DIR)

INVALID_STORIES = frozenset(
    {
        "dialogue1",
        "dialogue2",
        "dialogue3",
        "dialogue4",
        "dialogue5",
        "dialogue6",
        "myfirstdaywiththeyankees",
        "onlyonewaytofindout",
    }
)

DEFAULT_DATA_ROOT = (
    "/path/to/project/data"
)


def _stories_with_bold(data_root: str, subject: str) -> set[str]:
    d = os.path.join(data_root, subject)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Missing BOLD directory: {d}")
    return {f[:-4] for f in os.listdir(d) if f.endswith(".npy")}


def load_bold(story: str, subject: str, data_root: str) -> np.ndarray:
    path = os.path.join(data_root, subject, f"{story}.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    y = np.load(path)
    return np.asarray(y, dtype=np.float32)


def per_voxel_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Pearson r per column; y_* shape (n_time, n_voxels)."""
    yt = y_true.astype(np.float64, copy=False)
    yp = y_pred.astype(np.float64, copy=False)
    yt = yt - yt.mean(axis=0, keepdims=True)
    yp = yp - yp.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt * yt).sum(axis=0) * (yp * yp).sum(axis=0))
    return np.where(den > 1e-12, num / den, 0.0)


def _filter_story_list(
    names: list[str],
    x_dict: dict,
    has_bold: set[str],
) -> list[str]:
    out: list[str] = []
    for s in names:
        if s in INVALID_STORIES:
            continue
        if s not in x_dict:
            continue
        if s not in has_bold:
            continue
        out.append(s)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bert_features_dir",
        default=None,
        help="Directory containing X_tr_bert_{variant}.pkl, X_tst_bert_{variant}.pkl, bert_meta_{variant}.json",
    )
    p.add_argument(
        "--x_tr_pkl", default=None, help="Override train X pickle path",
    )
    p.add_argument(
        "--x_tst_pkl", default=None, help="Override test X pickle path",
    )
    p.add_argument(
        "--meta_json", default=None, help="Override bert_meta JSON path",
    )
    p.add_argument(
        "--variant", choices=["pt", "ft", "lora"], default="pt",
        help="Which BERT feature set (file suffix)",
    )
    p.add_argument(
        "--data_root", default=DEFAULT_DATA_ROOT,
        help="Parent of subject2/ subject3/ with {story}.npy",
    )
    p.add_argument("--subject", default="subject2", help="subject2 or subject3")
    p.add_argument(
        "--output_dir", default=None,
        help="Where to save ridge pkl + summary (default: code/results/)",
    )
    p.add_argument(
        "--alphas", default="1,10,100,1000,10000,100000",
        help="Comma-separated RidgeCV alphas",
    )
    p.add_argument(
        "--max_voxels", type=int, default=None,
        help="If set, only use first N voxels of Y (debug / low RAM)",
    )
    args = p.parse_args()

    from sklearn.linear_model import RidgeCV

    v = args.variant
    if args.bert_features_dir:
        base = os.path.abspath(args.bert_features_dir)
        x_tr_path = os.path.join(base, f"X_tr_bert_{v}.pkl")
        x_tst_path = os.path.join(base, f"X_tst_bert_{v}.pkl")
        meta_path = os.path.join(base, f"bert_meta_{v}.json")
    else:
        if not (args.x_tr_pkl and args.x_tst_pkl):
            print(
                "ERROR: provide --bert_features_dir or both --x_tr_pkl and --x_tst_pkl",
                file=sys.stderr,
            )
            sys.exit(1)
        x_tr_path = os.path.abspath(args.x_tr_pkl)
        x_tst_path = os.path.abspath(args.x_tst_pkl)
        meta_path = (
            os.path.abspath(args.meta_json)
            if args.meta_json
            else os.path.join(os.path.dirname(x_tr_path), f"bert_meta_{v}.json")
        )

    out_dir = args.output_dir or os.path.join(_CODE_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    print(f"[load] X train: {x_tr_path}")
    with open(x_tr_path, "rb") as f:
        X_tr: dict[str, np.ndarray] = pickle.load(f)
    print(f"[load] X test: {x_tst_path}")
    with open(x_tst_path, "rb") as f:
        X_tst: dict[str, np.ndarray] = pickle.load(f)

    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta: dict[str, Any] = json.load(f)
        train_order = list(meta.get("train_stories", []))
        test_order = list(meta.get("test_stories", []))
    else:
        print(f"[warn] no meta at {meta_path}; using dict key order")
        train_order = list(X_tr.keys())
        test_order = list(X_tst.keys())

    has_bold = _stories_with_bold(args.data_root, args.subject)
    train_stories = _filter_story_list(train_order, X_tr, has_bold)
    test_stories = _filter_story_list(test_order, X_tst, has_bold)

    if not train_stories:
        print("ERROR: no train stories with both X and BOLD", file=sys.stderr)
        sys.exit(1)
    if not test_stories:
        print("ERROR: no test stories with both X and BOLD", file=sys.stderr)
        sys.exit(1)

    print(f"[split] {len(train_stories)} train / {len(test_stories)} test stories")

    print("[stack] train X / Y …")
    X_list = [np.asarray(X_tr[s], dtype=np.float32) for s in train_stories]
    Y_list = [load_bold(s, args.subject, args.data_root) for s in train_stories]
    for s, x, y in zip(train_stories, X_list, Y_list, strict=True):
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Row mismatch {s}: X {x.shape} vs Y {y.shape} "
                "(check trim/delay alignment with preprocessing)"
            )
    X_train = np.vstack(X_list)
    Y_train = np.vstack(Y_list)
    if args.max_voxels is not None:
        Y_train = Y_train[:, : args.max_voxels]

    print(f"[stack] X_train {X_train.shape}  Y_train {Y_train.shape}")

    print(f"[ridge] RidgeCV alphas={alphas} …")
    ridge = RidgeCV(alphas=alphas)
    ridge.fit(X_train, Y_train)
    print(f"[ridge] best alpha = {ridge.alpha_}")

    per_story: dict[str, dict[str, float]] = {}
    all_cc: list[np.ndarray] = []

    print("[eval] test stories …")
    for s in test_stories:
        Xt = np.asarray(X_tst[s], dtype=np.float32)
        Yt = load_bold(s, args.subject, args.data_root)
        if Xt.shape[0] != Yt.shape[0]:
            raise ValueError(f"Row mismatch test {s}: X {Xt.shape} vs Y {Yt.shape}")
        if args.max_voxels is not None:
            Yt = Yt[:, : args.max_voxels]
        pred = ridge.predict(Xt)
        cc = per_voxel_pearson(Yt, pred)
        all_cc.append(cc)
        per_story[s] = {
            "mean_r": float(np.nanmean(cc)),
            "median_r": float(np.nanmedian(cc)),
            "frac_positive": float(np.mean(cc > 0)),
        }
        print(f"  {s}: mean_r={per_story[s]['mean_r']:.4f}")

    stacked = np.concatenate(all_cc)
    summary = {
        "variant": v,
        "subject": args.subject,
        "data_root": os.path.abspath(args.data_root),
        "x_tr_pkl": os.path.abspath(x_tr_path),
        "x_tst_pkl": os.path.abspath(x_tst_path),
        "meta_json": os.path.abspath(meta_path) if os.path.isfile(meta_path) else None,
        "train_stories": train_stories,
        "test_stories": test_stories,
        "ridgecv_alpha": float(ridge.alpha_),
        "alphas_tried": alphas,
        "max_voxels": args.max_voxels,
        "global_mean_r": float(np.mean(stacked)),
        "global_median_r": float(np.median(stacked)),
        "per_story": per_story,
    }

    ridge_path = os.path.join(out_dir, f"ridge_bert_{v}_{args.subject}.pkl")
    with open(ridge_path, "wb") as f:
        pickle.dump(ridge, f, protocol=4)
    print(f"[saved] {ridge_path}")

    summ_path = os.path.join(out_dir, f"ridge_bert_{v}_{args.subject}_summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {summ_path}")


if __name__ == "__main__":
    main()
