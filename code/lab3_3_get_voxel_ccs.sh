import numpy as np
import pickle
import os
from huggingface_hub import hf_hub_download
from scipy.stats import pearsonr

REPO = "YOUR_HF_USER/stat214-lab3.2"
BOLD_DIR = "/path/to/data"
OUT_DIR = "/path/to/fmri-language-encoding-models/results"
os.makedirs(OUT_DIR, exist_ok=True)

INVALID_STORIES = {
    'dialogue1', 'dialogue2', 'dialogue3',
    'dialogue4', 'dialogue5', 'dialogue6',
    'myfirstdaywiththeyankees', 'onlyonewaytofindout'
}

def per_voxel_pearson(y_true, y_pred):
    yt = y_true.astype(np.float64) - y_true.mean(axis=0)
    yp = y_pred.astype(np.float64) - y_pred.mean(axis=0)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0))
    return np.where(den > 1e-12, num / den, 0.0)

def load_bold(story, subject):
    path = os.path.join(BOLD_DIR, subject, f'{story}.npy')
    if not os.path.exists(path):
        return None
    return np.nan_to_num(np.load(path), nan=0.0)

for model_name in ["lora", "ft", "pt"]:
    print(f"\n{'='*40}\nProcessing BERT {model_name}\n{'='*40}")

    # load test features
    tst_path = hf_hub_download(repo_id=REPO, filename=f"X_tst_bert_{model_name}.pkl", repo_type="dataset")
    with open(tst_path, "rb") as f:
        X_tst = pickle.load(f)
    X_tst = {k: v for k, v in X_tst.items() if k not in INVALID_STORIES}
    test_stories = list(X_tst.keys())
    print(f"test stories: {len(test_stories)}")

    for subject in ["subject2", "subject3"]:
        print(f"\n--- {subject} ---")

        # load coef instead of ridge pickle
        coef_path = hf_hub_download(
            repo_id=REPO, 
            filename=f"ridge_bert_{model_name}_{subject}_coef.npy", 
            repo_type="dataset"
        )
        coef = np.load(coef_path)  # (94219, 3072)
        print(f"coef shape: {coef.shape}")

        # stack test data + bold
        X_list, Y_list = [], []
        for story in test_stories:
            y = load_bold(story, subject)
            if y is None:
                print(f"  skipping {story} - no BOLD")
                continue
            x = X_tst[story]
            if x.shape[0] != y.shape[0]:
                print(f"  skipping {story} - shape mismatch")
                continue
            X_list.append(x)
            Y_list.append(y)

        X_test = np.vstack(X_list).astype(np.float32)
        Y_test = np.vstack(Y_list).astype(np.float32)
        print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")

        # predict + CC in chunks to avoid OOM
        print("computing per-voxel CCs in chunks...")
        CHUNK = 5000
        n_voxels = coef.shape[0]
        voxel_ccs = np.zeros(n_voxels)
        for start in range(0, n_voxels, CHUNK):
            end = min(start + CHUNK, n_voxels)
            coef_chunk = coef[start:end]          # (5000, 3072)
            Y_pred_chunk = X_test @ coef_chunk.T  # (T, 5000)
            Y_test_chunk = Y_test[:, start:end]   # (T, 5000)
            voxel_ccs[start:end] = per_voxel_pearson(Y_test_chunk, Y_pred_chunk)
            if start % 20000 == 0:
                print(f"  {start}/{n_voxels} voxels done...")

        # save
        out_path = os.path.join(OUT_DIR, f"cc_bert_{model_name}_{subject}.npy")
        np.save(out_path, voxel_ccs)
        print(f"saved to {out_path}")
        print(f"mean CC:   {np.nanmean(voxel_ccs):.4f}")
        print(f"median CC: {np.nanmedian(voxel_ccs):.4f}")
        print(f"top 1%:    {np.nanpercentile(voxel_ccs, 99):.4f}")
        print(f"top 5%:    {np.nanpercentile(voxel_ccs, 95):.4f}")