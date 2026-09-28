# Business Entity Resolution Pipeline

This repository contains the complete, self-contained reproduction code for the Amazon ML Challenge 2026 Business Entity Resolution solution.

## 1. Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Recommended Python version: `>= 3.10`

### Dependencies:
- `numpy>=2.0.0`
- `pandas>=2.2.0`
- `scikit-learn>=1.5.0`
- `lightgbm>=4.5.0`
- `joblib>=1.4.0`
- `pyarrow>=17.0.0`

---

## 2. Directory Structure

```
business_entity_resolution/
├── requirements.txt
├── README.md
└── src/
    ├── normalization_engine.py      # Multi-representation profiling, OCR/diacritic cleaning, script tagging
    ├── feature_extractor_v2.py      # 46-dimensional feature extraction engine
    ├── model_experiments_v2.py      # LightGBM training & cross-validation ablation pipeline
    ├── predict_submission_v2.py     # Partitioned streaming inference with Dynamic Agreement logic
    ├── fast_submission_generator.py # Heuristic baseline generator
    ├── evaluate_local_validation.py # Local validation scoring script (Macro F0.5)
    ├── optimize_threshold_v2.py     # Threshold optimization & decision rule tuner
    └── benchmark_candidate_limits.py# Candidate budget & recall ceiling evaluation
```

---

## 3. End-to-End Execution & Reproduction

### Step 1: Feature Extraction & Model Training (Optional, pre-trained model included)
To retrain LightGBM Model D from training data:
```bash
python src/model_experiments_v2.py \
    --train-source1 ../../student_resource/dataset/train/train_source1.tsv \
    --train-source2 ../../student_resource/dataset/train/train_source2.tsv \
    --train-source3 ../../student_resource/dataset/train/train_source3.tsv \
    --ground-truth ../../student_resource/dataset/train/train_ground_truth.tsv \
    --output-model src/entity_matching_model_v2.joblib
```

### Step 2: Full Inference on Test Set
To generate both `candidate_pairs.tsv` and `matching_results.tsv`:
```bash
python src/predict_submission_v2.py \
    --test-dir ../../student_resource/dataset/test \
    --output-dir ../../output
```

This command executes:
1. Multi-tier inverted candidate blocking partitioned by country (`INDIA`, `US`, `FRANCE`).
2. Candidate budget enforcement ($\le 15$ multi-block and $\le 5$ single-block candidates per entity).
3. 46-dimensional pairwise feature vector generation.
4. Model D scoring and Dynamic Agreement filtering ($T_{\text{high}}=0.75, T_{\text{med}}=0.58$ with exact name consensus or building match).
5. Output streaming directly into `candidate_pairs.tsv` and `matching_results.tsv`.

---

## 4. Output Validation

Run the official competition validation script:
```bash
python ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```
Expected output:
```
PASS — no blocking issues found. Safe to submit.
```
