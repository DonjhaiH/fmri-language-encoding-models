"""
lab3.1_fit_ridge.py

Fits a ridge regression model mapping embeddings (X) to
BOLD fMRI responses (Y) for each subject using bootstrap ridge regression.
Saves per-voxel CC metrics and selected alphas to the results directory.

Usage:
    python fit_ridge_bow.py \
        --bold_dir /path/to/bold/data \
        --out_dir  /path/to/results \
        --subject  subject2

Notes:
    - BOLD .npy files should match X row counts per story
    - Delays should already applied in preprocessing pipeline (lags 1-4)
    - Uses bootstrap ridge from ridge_utils for per-voxel alpha selection
"""

import argparse
import pickle
import sys
import os
import numpy as np

from huggingface_hub import hf_hub_download
from ridge_utils.ridge import bootstrap_ridge

# fs = HfFileSystem()
# with fs.open("datasets/USER/repo/data.jsonl", "r") as f:
#     print(f.readline())

# stream data from hugging face
train_path = hf_hub_download(
    repo_id="Donjhaih/lab3_stat214_files",
    filename="X_train_w2v.pkl",
    repo_type="dataset"
)

test_path = hf_hub_download(
    repo_id="Donjhaih/lab3_stat214_files",
    filename="X_test_w2v.pkl",
    repo_type="dataset"
)

INVALID_STORIES = {
    'dialogue1', 'dialogue2', 'dialogue3',
    'dialogue4', 'dialogue5', 'dialogue6',
    'myfirstdaywiththeyankees', 'onlyonewaytofindout'
}

ALPHAS   = np.logspace(1, 5, 10)  # log-spaced to cover useful range
NBOOTS   = 8
CHUNKLEN = 10
NCHUNKS  = 10


def parse_args():
    parser = argparse.ArgumentParser(description='Fit ridge regression on embeddings')
    # parser.add_argument('--x_tr',    required=True, help='path to X_tr pkl')
    # parser.add_argument('--x_tst',   required=True, help='path to X_tst pkl')
    parser.add_argument('--bold_dir', required=True, help='directory with subject BOLD data')
    parser.add_argument('--out_dir',  required=True, help='directory to save results')
    parser.add_argument('--subject',  required=True, help='e.g. subject2 or subject3')
    parser.add_argument('--embedding', required=True, help='embedding type e.g. glove or w2v')
    return parser.parse_args()


def load_x(path):
    print(f"loading X from {path}")
    with open(path, 'rb') as f:
        X = pickle.load(f)
    X_clean = {k: v for k, v in X.items() if k not in INVALID_STORIES}
    print(f"  {len(X)} stories, {len(X_clean)} after removing invalid")
    return X_clean


def load_bold(story, subject, bold_dir):
    path = os.path.join(bold_dir, subject, f'{story}.npy')
    if not os.path.exists(path):
        return None
    y = np.load(path)
    if np.isnan(y).any():
        print(f"  warning: NaNs found in {subject}/{story} -- replacing with 0")
        y = np.nan_to_num(y, nan=0.0)
    return y


def stack_data(stories, X_dict, subject, bold_dir):
    X_list, Y_list, skipped = [], [], []
    for story in stories:
        if story not in X_dict:
            skipped.append(story)
            continue
        y = load_bold(story, subject, bold_dir)
        if y is None:
            skipped.append(story)
            continue
        x = X_dict[story]
        if x.shape[0] != y.shape[0]:
            print(f"  warning: row mismatch for {story} "
                  f"(X={x.shape[0]}, Y={y.shape[0]}) -- skipping")
            skipped.append(story)
            continue
        if np.isnan(x).any():
            x = np.nan_to_num(x, nan=0.0)
        X_list.append(x)
        Y_list.append(y)
    if skipped:
        print(f"  skipped {len(skipped)} stories: {skipped}")
    return np.vstack(X_list), np.vstack(Y_list)


def fast_zscore(x):
    std = x.std(0)
    std[std < 1e-8] = 1.0  # avoid divide by zero
    return (x - x.mean(0)) / std


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    subject = args.subject

    # load embeddings
    X_tr  = load_x(train_path)
    X_tst = load_x(test_path)

    train_stories = list(X_tr.keys())
    test_stories  = list(X_tst.keys())
    print(f"train stories: {len(train_stories)}")
    print(f"test stories:  {len(test_stories)}")

    print(f"\n{'='*50}\nfitting model for {subject}\n{'='*50}")

    # stack train data
    print("stacking train data...")
    X_train, Y_train = stack_data(train_stories, X_tr, subject, args.bold_dir)
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")

    # stack test data
    print("stacking test data...")
    X_test, Y_test = stack_data(test_stories, X_tst, subject, args.bold_dir)
    print(f"  X_test: {X_test.shape}, Y_test: {Y_test.shape}")

    # cast to float32
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    Y_test  = Y_test.astype(np.float32)

    # reduce X dimensionality via SVD (applied after delays already in pkl)
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=300, random_state=42)
    X_train = svd.fit_transform(X_train).astype(np.float32)
    X_test  = svd.transform(X_test).astype(np.float32)

    # z-score
    X_train = fast_zscore(X_train)
    X_test  = fast_zscore(X_test)
    Y_train = fast_zscore(Y_train)
    Y_test  = fast_zscore(Y_test)

    # bootstrap ridge
    print(f"running bootstrap_ridge: nboots={NBOOTS}, "
          f"chunklen={CHUNKLEN}, nchunks={NCHUNKS}, "
          f"n_alphas={len(ALPHAS)}")
    print(f"  alphas: {ALPHAS}")

    wt, corrs, valphas, allRcorrs, valinds = bootstrap_ridge(
        Rstim=X_train,
        Rresp=Y_train,
        Pstim=X_test,
        Presp=Y_test,
        alphas=ALPHAS,
        nboots=NBOOTS,
        chunklen=CHUNKLEN,
        nchunks=NCHUNKS,
        use_corr=True,
        return_wt=False,
    )

    # save outputs
    np.save(os.path.join(args.out_dir, f'cc_{args.embedding}_{subject}.npy'), corrs)
    np.save(os.path.join(args.out_dir, f'valphas_{args.embedding}_{subject}.npy'), valphas)
    print(f"  saved CC to {args.out_dir}/cc_{subject}.npy")
    print(f"  saved valphas to {args.out_dir}/valphas_{subject}.npy")

    # summary
    print(f"\n--- CC summary ({subject}) ---")
    print(f"  mean CC:   {corrs.mean():.4f}")
    print(f"  median CC: {np.median(corrs):.4f}")
    print(f"  top 1% CC: {np.percentile(corrs, 99):.4f}")
    print(f"  top 5% CC: {np.percentile(corrs, 95):.4f}")
    print(f"  unique alphas selected: {np.unique(valphas)}")
    print("done")


if __name__ == '__main__':
    main()