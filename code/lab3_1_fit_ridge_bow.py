"""
fit_ridge_bow.py

Fits a ridge regression model mapping BoW embeddings (X) to
BOLD fMRI responses (Y) for each subject. Saves the fitted model
and per-voxel CC metrics to the results directory.

Usage:
    python fit_ridge_bow.py \
        --x_tr    /path/to/X_tr_bow.pkl \
        --x_tst   /path/to/X_tst_bow.pkl \
        --bold_dir /path/to/bold/data \
        --out_dir  /path/to/results

Example (Bridges2):
    python fit_ridge_bow.py \
        --x_tr    /path/to/project/results/X_tr_bow.pkl \
        --x_tst   /path/to/project/results/X_tst_bow.pkl \
        --bold_dir /path/to/project/data \
        --out_dir  /path/to/project/results
"""
import argparse
import os
import pickle
import sys
import numpy as np
import sys
sys.path.append('/path/to/fmri-language-encoding-models')

from ridge_utils.ridge import bootstrap_ridge
from ridge_utils.utils import zscore
from sklearn.decomposition import TruncatedSVD

INVALID_STORIES = {
    'dialogue1', 'dialogue2', 'dialogue3',
    'dialogue4', 'dialogue5', 'dialogue6',
    'myfirstdaywiththeyankees', 'onlyonewaytofindout'
}

ALPHAS = np.logspace(1, 6, 8)  # log-spaced for better coverage

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_tr',    required=True)
    parser.add_argument('--x_tst',   required=True)
    parser.add_argument('--bold_dir', required=True)
    parser.add_argument('--out_dir',  required=True)
    parser.add_argument('--subject',  required=True, help='e.g. subject2')
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
        y = np.nan_to_num(y, nan=0.0)
    return y

def stack_data(stories, X_dict, subject, bold_dir):
    X_list, Y_list, skipped = [], [], []
    for story in stories:
        if story not in X_dict:
            skipped.append(story); continue
        y = load_bold(story, subject, bold_dir)
        if y is None:
            skipped.append(story); continue
        x = X_dict[story]
        if x.shape[0] != y.shape[0]:
            print(f"  row mismatch {story}, skipping")
            skipped.append(story); continue
        if np.isnan(x).any():
            x = np.nan_to_num(x, nan=0.0)
        X_list.append(x)
        Y_list.append(y)
    if skipped:
        print(f"  skipped {len(skipped)} stories")
    return np.vstack(X_list), np.vstack(Y_list)

def fast_zscore(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    subject = args.subject

    X_tr  = load_x(args.x_tr)
    X_tst = load_x(args.x_tst)

    train_stories = list(X_tr.keys())
    test_stories  = list(X_tst.keys())

    print(f"\n{'='*50}\nfitting model for {subject}\n{'='*50}")

    # stack data
    print("stacking train data...")
    X_train, Y_train = stack_data(train_stories, X_tr, subject, args.bold_dir)
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")

    print("stacking test data...")
    X_test_list, Y_test_list = [], []
    for story in test_stories:
        y = load_bold(story, subject, args.bold_dir)
        if y is None or story not in X_tst: continue
        x = X_tst[story]
        if x.shape[0] != y.shape[0]: continue
        X_test_list.append(x)
        Y_test_list.append(y)
    X_test = np.vstack(X_test_list)
    Y_test = np.vstack(Y_test_list)

    # cast to float32 to save memory
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    Y_test  = Y_test.astype(np.float32)

    # reduce X size
    svd = TruncatedSVD(n_components=2000)
    X_train = svd.fit_transform(X_train)
    X_test  = svd.transform(X_test)

    # float 32 again
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    # --- filter uninformative voxels ---
    # keep top-k most variable voxels
    # voxel_var = Y_train.var(0)
    # k = 10000   # start with this, can increase
    # inds = np.argsort(voxel_var)[-k:]
    # Y_train_masked = Y_train[:, inds]
    # Y_test_masked  = Y_test[:, inds]

    # remove voxel filtering
    Y_train_masked = Y_train
    Y_test_masked  = Y_test

    # z-score (bootstrap_ridge expects z-scored inputs)
    X_train = fast_zscore(X_train)
    X_test  = fast_zscore(X_test)
    Y_train_masked = fast_zscore(Y_train_masked)
    Y_test_masked  = fast_zscore(Y_test_masked)

    # --- run bootstrap_ridge ---
    # nboots=15, chunklen=10, nchunks ~ 20% of data
    nchunks = 5
    #max(1, int(X_train.shape[0] * 0.2 / 10))
    print(f"running bootstrap_ridge: nboots=2, chunklen=10, nchunks={nchunks}")
    
    wt, corrs, valphas, allRcorrs, valinds = bootstrap_ridge(
        Rstim=X_train,
        Rresp=Y_train_masked,
        Pstim=X_test,
        Presp=Y_test_masked,
        alphas=ALPHAS,
        nboots=2,
        chunklen=10,
        nchunks=nchunks,
        use_corr=True,
        return_wt=False,   # set False if you run into memory issues
    )

    # expand corrs back to full voxel space
    corrs_full = corrs
    valphas_full = valphas

    # save outputs
    np.save(os.path.join(args.out_dir, f'cc_bow_{subject}.npy'), corrs_full)
    np.save(os.path.join(args.out_dir, f'valphas_{subject}.npy'), valphas_full)
    if wt is not None and len(wt) > 0:
        with open(os.path.join(args.out_dir, f'weights_bow_{subject}.pkl'), 'wb') as f:
            pickle.dump(wt, f)

    print(f"\n--- CC summary ({subject}) ---")
    print(f"  mean CC:   {corrs.mean():.4f}")
    print(f"  median CC: {np.median(corrs):.4f}")
    print(f"  top 1% CC: {np.percentile(corrs, 99):.4f}")
    print(f"  top 5% CC: {np.percentile(corrs, 95):.4f}")
    print("done")

if __name__ == '__main__':
    main()