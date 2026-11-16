#!/usr/bin/env python3
"""
lab3.2_make_stories_split.py
============================
Build ``stories_split.json`` (train / test story names) for Lab 3.2 scripts.

Priority:
  1. If the HF dataset has ``stories_split.json``, download only that small file.
  2. Else if ``--tr_pkl`` and ``--tst_pkl`` point to local BoW pickles, read keys only.
  3. Else: error (this script does not download the large BoW pickles from HF).

Typical local run (from ``code/`` or anywhere):

    python lab3.2_make_stories_split.py

    # Large cache on disk D:, then:
    set HF_HOME=D:\\hf_cache
    python lab3.2_make_stories_split.py --output stories_split.json

    # Already downloaded BoW pkls:
    python lab3.2_make_stories_split.py \\
        --tr_pkl X_tr_bow.pkl --tst_pkl X_tst_bow.pkl --output code/stories_split.json

Optional: ``--raw_text_pkl`` filters story names to those present in ``raw_text.pkl``
(Bridges shared path or a copy on your machine).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CODE_DIR)
# Optional --raw_text_pkl filter unpickles ridge_utils.DataSequence.
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _CODE_DIR)

DEFAULT_REPO = "YOUR_HF_USER/train_test_pkl"


def _try_hub_small_split(repo_id: str, dest_path: str) -> bool:
    """Return True if stories_split.json was fetched from Hub and written."""
    try:
        from huggingface_hub import hf_hub_download

        src = hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename="stories_split.json"
        )
        with open(src) as f:
            data = json.load(f)
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[ok] Wrote {dest_path} from Hub stories_split.json")
        return True
    except Exception as exc:
        print(f"[info] Hub has no usable stories_split.json ({type(exc).__name__}: {exc})")
        return False


def _keys_from_pickle(path: str) -> list[str]:
    print(f"[data] loading keys from {path} …")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not hasattr(obj, "keys"):
        raise TypeError(f"Expected dict-like pickle at {path}, got {type(obj)}")
    return list(obj.keys())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DEFAULT_REPO, help="HF dataset repo id")
    p.add_argument(
        "--tr_pkl", default=None, help="Local path to X_tr_bow.pkl (skip HF download if both set)",
    )
    p.add_argument(
        "--tst_pkl", default=None, help="Local path to X_tst_bow.pkl",
    )
    p.add_argument(
        "--output", "-o",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "stories_split.json"),
        help="Output JSON path (default: code/stories_split.json next to this script)",
    )
    p.add_argument(
        "--raw_text_pkl", default=None,
        help="If set, filter train/test lists to stories present in this raw_text.pkl",
    )
    args = p.parse_args()

    if _try_hub_small_split(args.repo_id, args.output):
        if args.raw_text_pkl:
            _filter_by_raw_text(args.output, args.raw_text_pkl)
        return

    if args.tr_pkl and args.tst_pkl:
        train_stories = _keys_from_pickle(args.tr_pkl)
        test_stories = _keys_from_pickle(args.tst_pkl)
    elif args.tr_pkl or args.tst_pkl:
        print("ERROR: provide both --tr_pkl and --tst_pkl.", file=sys.stderr)
        sys.exit(1)
    else:
        print(
            "ERROR: no stories_split.json on Hub and no --tr_pkl/--tst_pkl.\n"
            "This script does not download X_tr_bow.pkl / X_tst_bow.pkl (~12 GB).\n"
            "Add stories_split.json to the dataset repo, or run with local:\n"
            "  python lab3.2_make_stories_split.py --tr_pkl ... --tst_pkl ... -o code/stories_split.json",
            file=sys.stderr,
        )
        sys.exit(1)

    data = {"train": train_stories, "test": test_stories}
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(
        f"[ok] Wrote {args.output}  ({len(train_stories)} train / {len(test_stories)} test stories)"
    )

    if args.raw_text_pkl:
        _filter_by_raw_text(args.output, args.raw_text_pkl)


def _filter_by_raw_text(json_path: str, raw_text_pkl: str) -> None:
    print(f"[filter] intersecting with stories in {raw_text_pkl} …")
    with open(raw_text_pkl, "rb") as f:
        raw = pickle.load(f)
    available = set(raw.keys())
    with open(json_path) as f:
        data = json.load(f)
    t_tr = [s for s in data["train"] if s in available]
    t_te = [s for s in data["test"] if s in available]
    data = {"train": t_tr, "test": t_te}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(
        f"[ok] Updated {json_path} after raw_text filter: "
        f"{len(t_tr)} train / {len(t_te)} test"
    )


if __name__ == "__main__":
    main()
