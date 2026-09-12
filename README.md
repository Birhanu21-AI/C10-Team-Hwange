# C10-Team-Hwange
# Ubuntu‑Aligned Latent Probing for Toxicity Detection with Gemma 2B

This repository contains code and resources for training **latent probes** on the multilingual Ubuntu‑oriented toxicity dataset. The project aligns with Ubuntu ethics by focusing on fairness, inclusivity, and harm‑aware detection across English and Amharic.

---

## Dataset

- **Source**: 4,000‑row synthetic dataset containing paired English and Amharic sentences, annotated for `hate_speech` vs. `safe`.  
- **Creation**: Each row includes parallel translations. Labels are binary (`1 = hate_speech`, `0 = safe`). Metadata fields (e.g., `ubuntu_harm_type_en`, `severity_en`) describe relational harms such as exclusion, silencing, stereotyping, or threat.  
- **Selection**: The dataset was designed to reflect Ubuntu principles by capturing harms relevant to community belonging and dignity.  
- **Preprocessing**:  
  - Each sentence expanded into two examples (English + Amharic).  
  - Group IDs ensure translations of the same sentence remain together during splits.  
  - Grouped train/validation/test split (70/15/15) prevents leakage across languages.

---

## Training Pipeline

1. **Model**: [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b) loaded via Hugging Face Transformers.  
2. **Embedding Extraction**:  
   - Mean‑pooled hidden states from each transformer layer.  
   - Attention mask applied to exclude padding.  
   - Embeddings saved layer‑wise as `.npy` files.  
3. **Probe Training**:  
   - StandardScaler + LogisticRegression pipeline.  
   - Hyperparameters:  
     - `max_iter = 2000`  
     - `class_weight = balanced`  
     - `random_state = 42`  
   - Probes trained per layer, evaluated on validation split.  
   - Best layer selected by F1 score.  
4. **Final Probe**: Retrained on train+validation sets, evaluated on held‑out test set.  
5. **Ubuntu Alignment**: Metadata on harm types used for post‑hoc analysis (not fed into probe). This ensures ethical reflection on which harms are easier/harder to detect.

---

## Evaluation

- **Metrics**: Accuracy, Precision, Recall, F1 (binary).  
- **Baselines**:  
  - Majority class baseline.  
  - Direct fine‑tuning of Gemma without probes.  
- **Verification**:  
  - Confusion matrices across English and Amharic subsets.  
  - Harm‑type analysis (e.g., belonging denial vs. stereotyping).  
  - Layer‑wise probe performance saved in `layer_probe_results.csv`.  
- **Results**:  
  - Best layer identified with highest validation F1.  
  - Final probe tested separately on English and Amharic to confirm cross‑lingual consistency.

---

## Reproduction

To reproduce results:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess dataset
python preprocess.py

# 3. Extract embeddings
python extract_embeddings.py

# 4. Train probes
python train_probe.py

# 5. Evaluate
python evaluate.py
