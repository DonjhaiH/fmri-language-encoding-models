import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
import shap
import pickle
import os
import sys
import json
from transformers import BertModel, BertTokenizerFast
from huggingface_hub import hf_hub_download
from lab3_2_extract_bert_features import _story_word_embeddings as get_bert_embeddings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from preprocessing import downsample_word_vectors, make_delayed
from ridge_utils.DataSequence import DataSequence

REPO = "YOUR_HF_USER/stat214-lab3.2"
BERT_DIM = 768
DELAYS = [1, 2, 3, 4]
TRIM_START = 5
TRIM_END = 10
CHUNK_WORDS = 250
BASE_DIR = "/path/to/fmri-language-encoding-models"


class fMRI_Predictor:
    def __init__(self, coef, tokenizer, bert_model, device, wordseq, voxel_index):
        self.coef = coef
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.device = device
        self.wordseq = wordseq
        self.voxel_index = voxel_index
        self.n_orig = len(wordseq.data)

    def predict(self, texts):
        predictions = []
        for text in texts:
            words = text.split()

            if len(words) == 0:
                predictions.append(0.0)
                continue

            # full BERT forward pass for every perturbation
            embeddings = get_bert_embeddings(
                words, self.bert_model, self.tokenizer, self.device
            )  # (n_words, 768)

            # truncate data_times to match word count
            n_words = len(words)
            data_times = self.wordseq.data_times[:n_words]
            tr_times = self.wordseq.tr_times

            # create temporary DataSequence with truncated timing
            tmp_seq = DataSequence(
                np.array(words),
                self.wordseq.split_inds,
                data_times,
                tr_times
            )

            # downsample to TR grid
            downsampled = downsample_word_vectors(
                ["tmp"], {"tmp": embeddings}, {"tmp": tmp_seq}
            )["tmp"]

            # trim + delay
            n_trs = downsampled.shape[0]
            downsampled = downsampled[TRIM_START:n_trs - TRIM_END]
            delayed = make_delayed(downsampled, DELAYS)

            # predict
            pred = delayed @ self.coef.T
            predictions.append(pred[:, self.voxel_index].sum())

        return np.array(predictions)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Load coef
    coef_path = hf_hub_download(
        repo_id=REPO,
        filename="ridge_bert_lora_subject3_coef.npy",
        repo_type="dataset"
    )
    coef = np.load(coef_path)

    # Load voxel CCs
    voxel_ccs = np.load(f"{BASE_DIR}/results/ridge_results/cc_bert_lora_subject3.npy")
    top_voxels = np.argsort(voxel_ccs)[::-1][:20]

    # Load BERT
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval().to(device)

    # Load stories
    BOLD_DIR = "/path/to/data"
    with open(f"{BOLD_DIR}/raw_text.pkl", "rb") as f:
        raw_stories = pickle.load(f)

    for story_name in ["sweetaspie", "thetriangleshirtwaistconnection"]:
        print(f"\n{'='*40}\nStory: {story_name}\n{'='*40}")
        wordseq = raw_stories[story_name]
        text_input = " ".join(wordseq.data)

        os.makedirs(f"{BASE_DIR}/results/shap_lime/subject3/{story_name}_s3", exist_ok=True)

        for voxel_idx in top_voxels[:5]:
            print(f"running voxel {voxel_idx} (CC={voxel_ccs[voxel_idx]:.4f})")

            shap_path = f"{BASE_DIR}/data/shap_lime/{story_name}_s3/shap_voxel{voxel_idx}.npy"
            lime_path = f"{BASE_DIR}/data/shap_lime/{story_name}_s3/lime_voxel{voxel_idx}.json"

            if os.path.exists(shap_path) and os.path.exists(lime_path):
                print(f"skipping voxel {voxel_idx} - already done")
                continue

            predictor = fMRI_Predictor(
                coef=coef,
                tokenizer=tokenizer,
                bert_model=bert_model,
                device=device,
                wordseq=wordseq,
                voxel_index=voxel_idx
            )

            # SHAP
            if not os.path.exists(shap_path):
                masker = shap.maskers.Text(r"\s+", mask_token="[MASK]")
                explainer = shap.Explainer(predictor.predict, masker)
                shap_values = explainer([text_input])
                np.save(shap_path, shap_values.values)
                print(f"  SHAP saved")


            # LIME
            if not os.path.exists(lime_path):
                def lime_predict(texts):
                    preds = predictor.predict(texts)
                    return np.column_stack([-preds, preds])

                lime_explainer = LimeTextExplainer(
                    class_names=["low activation", "high activation"]
                )
                exp = lime_explainer.explain_instance(
                    text_input,
                    lime_predict,
                    num_features=20,
                    num_samples=500,
                    labels=[1]
                )
                with open(lime_path, "w") as f:
                    json.dump(dict(exp.as_list(label=1)), f)
                print(f"  LIME saved")

    print("done!")


if __name__ == "__main__":
    main()
