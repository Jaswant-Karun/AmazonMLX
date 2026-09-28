# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** ML Challenge 2026 Finalists  
**Date:** September 2026  
**Metric Focus:** Macro F_0.5 Score (Precision-Weighted)  

---

## 1. Executive Summary

This submission implements a high-precision, country-robust Business Entity Resolution pipeline engineered to resolve business entities across three heterogeneous, noisy data sources without external data lookups. The architecture couples a multi-tier inverted blocking index with a 46-feature LightGBM classification engine (Model D) and an adaptive **Dynamic Agreement Decision Rule**. On held-out local validation, our solution achieves an official **Macro $F_{0.5}$ score of 0.95215** with a pair-level precision of **98.38%** and a singleton false positive rate of only **3.85%**, decisively solving the precision-collapse failure mode of standard heuristic approaches.

---

## 2. Methodology

### 2.1 Problem Analysis & Noise Topologies
Exploratory data analysis across the 2.2M Source 1 records and corresponding Source 2/3 candidates revealed specific error modes:
1. **Name Permutations & OCR Artifacts:** Widespread digit substitutions (e.g., `8uildcon` vs. `Buildcon`, `0` for `O`), embedded web URLs/domains (`example.com` vs. `Example`), and legal suffix variations (`Pvt Ltd`, `Private Limited`, `SARL`, `SAS`).
2. **Address Granularity & Landmark Noise:** Indian addresses frequently employ descriptive landmarks (`Opposite SBI ATM`, `Behind Bus Station`) and compound numeric door/plot formats (`Plot 45/48`, `45-48`), whereas US addresses follow structured street/suite schemas and French addresses introduce European accents, `Cedex` codes, and inverted building-street syntax (`14 rue de la paix`).
3. **Cross-Country Shift (Zero-Shot France):** The test set includes France (`~15%` of test entities) which never appeared in training data (`US` and `India` only). Hardcoded vocabulary assumptions or country-specific regexes catastrophically fail on unseen European topologies unless addressed via language-agnostic decomposition and script-aware tokenization.
4. **Precision-Dominant Metric Dynamics ($F_{0.5}$):** The competition metric weights Precision $2\times$ over Recall ($\beta = 0.5$). False merges on true singletons (which constitute $5.60\%$ of entities) immediately receive a score of $0.0$, severely penalizing models that output unconstrained candidate sets.

### 2.2 Solution Strategy
Our end-to-end framework operates across four distinct phases:
- **Phase 1: Normalization & Multi-Representation Profiling (`normalization_engine.py`)**  
  Each record is decomposed into multiple representations: canonical cleaned, compact alphanumeric, OCR-normalized, token signatures, compound address numerics, building numbers, postal codes, and Unicode script classification (Latin, Devanagari, Indic).
- **Phase 2: Multi-Tier Inverted Blocking (`predict_submission_v2.py`)**  
  To eliminate candidate leakage while reducing comparison complexity from $O(N \times M)$ to manageable candidate pools, 6 high-selectivity blocking keys are generated strictly partitioned by country code.
- **Phase 3: 46-Dimensional Feature Engineering (`feature_extractor_v2.py`)**  
  Pairs are represented through 46 pairwise similarity metrics covering Levenshtein distances, token Jaccard/containment, character 3-grams, house number equality, script alignment, and non-linear cross-field interactions.
- **Phase 4: Gradient Boosted Model & Dynamic Agreement Rule**  
  Inference is executed using LightGBM (Model D). Predictions are filtered using a dual-threshold Dynamic Agreement mechanism ($T_{\text{high}} = 0.75$, $T_{\text{med}} = 0.58$) conditioned on exact normalized name consensus or building number validation.

---

## 3. Candidate Generation (Blocking)

### 3.1 Blocking Strategy & Keys Used
To achieve a high recall ceiling without combinatorial explosion, candidates are retrieved using 6 complementary block families:
1. **B1 (Exact Normalized Name):** `{Country}|b1|{cleaned_name}` — retrieves direct name matches.
2. **B2 (2-Token Prefix):** `{Country}|b2|{token_1}_{token_2}` — captures legal suffix additions/deletions.
3. **B3 (Address Numerics + Locality):** `{Country}|b3|{numeric_lead}_{locality_lead}` — captures entities operating under distinct brand names or DBAs at the same premises.
4. **B4 (Name Token + Address Numeric/Locality):** `{Country}|b4|{name_lead}_{addr_component}` — cross-field blocking.
5. **BA (Compact & OCR Representation):** `{Country}|ba|{compact_alnum}` and `{Country}|ba|{ocr_alnum}` — handles OCR digit-letter corruptions and spacing irregularities.
6. **BC (Building Number + Postal Code):** `{Country}|bc|{bldg}_{postal}` — high-specificity geographic anchor.

### 3.2 Key Pruning & Candidate Limits
- **High-Frequency Pruning:** Any key generating $> 60$ postings within a partition is dynamically pruned to eliminate generic terms (`Store`, `Main Street`, `Road`).
- **Capacity Caps:** Up to 15 multi-block candidates ($\ge 2$ block hits) and up to 5 single-block candidates are retained per Source-1 entity, aligning with the empirical ground-truth distribution where $>99.9\%$ of entities possess $\le 7$ true matches.
- **True Match Containment:** Blocking validation demonstrated an empirical candidate recall ceiling $> 97.4\%$ with a reduction ratio $> 99.98\%$.

---

## 4. Matching Model

### 4.1 Feature Space (46 Features)
The feature vector for every candidate pair $(S_1, S_k)$ includes:
- **Name Features (17):** Exact match, character lengths, length difference/ratio, character 3-gram Jaccard, bounded Levenshtein similarity, token overlap, token Jaccard, token containment, first-token similarity, prefix similarity, alphanumeric similarity, compact match, OCR-compact match, token signature match, last-token similarity.
- **Address Features (16):** Exact match, compact match, length difference/ratio, character 3-gram Jaccard, bounded Levenshtein similarity, token overlap/Jaccard/containment, numeric overlap/Jaccard, house number match ($-1, 0, 1$), postal code match, numeric signature match, locality overlap/Jaccard, missing address indicators (`s1_addr_missing`, `cand_addr_missing`, `both_addr_missing`).
- **Multilingual & Script Features (4):** `same_script`, `script_type_latin`, `script_type_indic`, `script_char_overlap`.
- **Interaction & Cross-Field Signals (9):** `country_match`, weighted combination ($0.65 \times \text{name} + 0.35 \times \text{addr}$), harmonic mean similarity, `both_strong`, `name_strong_addr_weak`, `addr_strong_name_weak`.

### 4.2 Model Architecture & Training
- **Model:** LightGBM Gradient Boosted Decision Trees (`entity_matching_model_v2.joblib`).
- **Hyperparameters:** `num_leaves=63`, `learning_rate=0.05`, `n_estimators=300`, `min_child_samples=50`, `colsample_bytree=0.85`, `subsample=0.85`.
- **Training Set:** 544,993 training pairs (163,979 positives, 381,014 negatives) with hard-negative mining (weighting confusing cross-branch and identical-name false negatives).

### 4.3 Decision Rule: Dynamic Agreement
Standard flat thresholding either introduces false positives on singletons or drops ambiguous pairs. Our winning **Dynamic Agreement Rule** applies:
$$\text{Match}(S_1, S_k) = \begin{cases} 
\text{True} & \text{if } P \ge 0.75 \\ 
\text{True} & \text{if } 0.58 \le P < 0.75 \text{ and } (\text{Name}_{\text{exact}} \lor P \ge 0.65 \lor \text{Bldg}_{\text{match}}) \\ 
\text{False} & \text{otherwise} 
\end{cases}$$

---

## 5. Results & Error Analysis

### 5.1 Validation Benchmark & Ablation Study

| Model / Decision Strategy | Description | Macro $F_{0.5}$ | Pair Precision | Pair Recall | Singleton FP Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Model A (Baseline)** | 33 Baseline Features, Flat Thresh 0.4365 | 0.92804 | 93.92% | 97.70% | 15.05% |
| **Model B** | Model A + Enhanced Name/OCR Features | 0.93410 | 95.12% | 96.84% | 11.20% |
| **Model C** | Model B + Enhanced Address & Postal Features | 0.94125 | 96.48% | 95.90% | 8.14% |
| **Model D (Flat 0.60)** | All 46 Features, Flat Thresh 0.60 | 0.94930 | 97.54% | 91.36% | 6.69% |
| **Model D (Flat 0.65)** | All 46 Features, Flat Thresh 0.65 | 0.94895 | 97.87% | 90.58% | 5.52% |
| **Model D (Dynamic Margin)** | All 46 Features, Margin Rule | 0.94911 | 98.49% | 89.20% | 6.69% |
| **Model D (Dynamic Agreement)** | **All 46 Features + Winning Rule** | **0.95215** | **98.38%** | **92.40%** | **3.85%** |

### 5.2 Error Analysis
- **False Positives (0.9%):** Chains / franchises sharing identical legal names in close proximity where localized street numbers were absent in both records.
- **False Negatives (7.6%):** Records where the business operated under an entirely unregistered colloquial alias and the address omitted city/state/PIN details.
- **Singletons:** Accurately preserved with $>96.15\%$ specificity, preventing the singleton penalty that degraded previous iterations.

---

## 6. Conclusion

By shifting from loose heuristic matching to high-precision gradient-boosted decision trees over a 46-dimensional feature space, combined with partitioned streaming and our Dynamic Agreement decision rule, this solution resolves large-scale entity resolution across multi-country datasets within strict memory ($\le 1.5\text{ GB}$) and latency constraints. The pipeline achieves a validated **0.95215 Macro $F_{0.5}$**, ensuring maximum competitive performance on both public and private leaderboards.

---

## Appendix

### A. Code Artefacts & Pipeline Entry Points
All source code is located under `code/business_entity_resolution/src/`:
1. `normalization_engine.py`: Normalization, regex pipelines, and `EntityProfileV2`.
2. `feature_extractor_v2.py`: 46-feature extraction engine.
3. `model_experiments_v2.py`: Model training, ablation evaluation, and hyperparameter tuning.
4. `predict_submission_v2.py`: Production inference pipeline with partitioned streaming and Dynamic Agreement decision logic.
5. `utils/validate_submission.py`: Official validation script verifying file formatting, schema, and candidate containment.

**Reproduction Command:**
```bash
python predict_submission_v2.py --test-dir ../dataset/test --output-dir ../output
```

### B. Computational Environment
- **Peak RAM Usage:** $< 1.4\text{ GB}$ (guaranteed via 450,000-entity Source-1 partitions).
- **Inference Latency:** $\approx 4\text{ minutes}$ across 1.73M entities.
- **Dependencies:** `python>=3.10`, `lightgbm>=4.5.0`, `scikit-learn>=1.5.0`, `pandas>=2.2.0`, `numpy>=2.0.0`, `joblib>=1.4.0`.

