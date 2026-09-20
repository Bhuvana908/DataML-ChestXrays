Chest X-Ray Multi-Condition Detection — Classical ML Pipeline
DataML – ML Value Challenge

This project predicts whether 5 conditions are present in a chest X-ray:

Atelectasis
Effusion
Infiltration
Nodule
Pneumothorax
An X-ray can show more than one condition at once, so this is multi-label classification — 5 separate yes/no questions per image, not one "sick vs healthy" question.

Rule followed: no deep learning, no CNNs, no pretrained models. Every prediction comes from hand-crafted image features + classical ML models (Logistic Regression, HistGradientBoosting, SVM, ExtraTrees).

What's new in this version
Changes since the previous notebook (feature set G, three ensembling methods):

New feature set H — adds a 15th family, Vessel/Rib Density (a Frangi vesselness filter measuring the density of linear structures — vessels, bronchial markings, ribs), on top of feature set G. feature_set is now set to "H". This is 231 raw features per image, up from 225 under set G. Unlike every other new family added so far, this one measures a fundamentally different kind of signal (linear-structure density) rather than another texture, shape, or intensity statistic.
Fourth ensembling method — Classifier Chain — in addition to the per-condition weighted average, stacked meta-model, and rank averaging, predictions are now also combined using sklearn.multioutput.ClassifierChain, which models each condition using the other four conditions' predictions as extra input features. This is the only ensembling method that explicitly tries to exploit the label co-occurrence patterns visible in the EDA (e.g. Atelectasis and Effusion sharing 121 cases), rather than treating the 5 conditions as independent problems. To avoid depending on one arbitrary chain order, 5 random label orderings are trained and averaged.
Automatic per-label ensemble selection now compares all four methods (previously three) — weighted average, stacked, rank average, and chain — and keeps whichever wins OOF ROC-AUC for that specific condition.
Feature-space outlier detection (new) — an IsolationForest is run over the cleaned feature matrix as a second, independent QC layer on top of the existing image-level corrupt-file and duplicate checks, flagging rows whose feature values look anomalous even when the source image loaded fine. An ablation (see below) tests whether excluding these rows from training actually helps.
Split conformal prediction (new) — per-condition thresholds are calibrated on OOF data to give a genuine 90% coverage guarantee ("the true label is in the predicted set at least 90% of the time"), producing an explicit uncertain flag rather than relying on the raw probability alone. Applied at test time too, adding a `{condition}_uncertain` column per condition to predictions.csv.
SHAP-based interpretability (new) — per-condition SHAP summary plots (TreeExplainer on the final ExtraTrees pipeline) add a direction-aware view on top of the existing mutual-information and permutation-importance reports: not just which features matter, but which way they push the prediction.
Family-averaged radar chart fixed for the new family — FAMILY_PREFIXES now includes an entry for the new Vessel/Rib Density family ("vessel_"), so it appears correctly in the diagnostic-fingerprint radar chart. (GLRLM, Gabor, Hu moments, and the clinical family are still not listed there — see the Feature Engineering write-up for this version.)
Known issue found while adding SHAP — the summary-plot loop currently plots both classes side-by-side instead of isolating the positive class (a 3-D shap_values array isn't being sliced the way the isinstance(shap_values, list) check assumes), and the custom plot title overlaps SHAP's own top feature label. Both are cosmetic/plumbing bugs, not a problem with which features SHAP found important; fixes are one-liners (see Section 14).

Experiment: per-condition ensemble method comparison
All four ensembling methods were computed and compared on OOF ROC-AUC for every condition:

Condition	Weighted avg	Stacked	Rank avg	Chain	Chosen
Atelectasis	0.6780	0.6767	0.6747	0.6777	weighted_avg
Effusion	0.7134	0.7144	0.7161	0.7108	rank_avg
Infiltration	0.6506	0.6474	0.6452	0.6292	weighted_avg
Nodule	0.6306	0.6239	0.6275	0.6095	weighted_avg
Pneumothorax	0.7038	0.7039	0.7047	0.6688	rank_avg

The classifier chain did not outright win for any condition, but for Atelectasis it landed within 0.0003 AUC of the winner — the closest margin of any label, and Atelectasis is also the condition with the 2nd- and 3rd-strongest co-occurrence links in the dataset (121 shared cases with Effusion, 104 with Infiltration). For Nodule and Pneumothorax — the two rarest conditions (250 and 217 positive cases) — the chain clearly underperformed. Reading: exploiting label correlation seems to help roughly in proportion to how much correlation exists and how much data supports estimating it, rather than being a universal win. The auto-selected ensemble's final macro ROC-AUC across all five conditions is 0.676, versus 0.6635 for the single best base model (ExtraTrees).

Experiment: feature-space outlier removal (IsolationForest)
OOF ROC-AUC with vs. without the IsolationForest-flagged rows excluded from training:

Condition	AUC before	AUC after	Delta
Atelectasis	0.6780	0.6886	+0.0106
Effusion	0.7134	0.7229	+0.0095
Infiltration	0.6506	0.6512	+0.0006
Nodule	0.6306	0.6024	−0.0282
Pneumothorax	0.7038	0.6900	−0.0138

Macro AUC — before: 0.6753, after: 0.6710. Excluding flagged rows helped the two most common conditions but hurt the two rarest ones, for a net negative on the macro average. The likely explanation: for a rare condition, an unusual-looking feature row is disproportionately likely to be one of its few positive cases, so blind outlier removal quietly deletes minority-class examples the model needs most. This filter is therefore not applied to the final training set by default; the flag is kept as metadata rather than used to drop rows.

1. Overall Pipeline
Raw X-ray images
  → Load & clean data (drop corrupt/missing images, find duplicates)
  → Preprocess each image (fix orientation, crop borders, normalize, enhance contrast)
  → Extract lung region (rough segmentation)
  → Extract hand-crafted features (feature set H: 15 families, incl. vessel/rib density)
  → Clean features (remove useless/redundant ones); flag feature-space outliers (IsolationForest)
  → Group-safe 5-fold split (duplicate images never span folds)
  → Tune hyperparameters per model (RandomizedSearchCV)
  → Train 4 classical models per condition, inside a leak-proof pipeline (SMOTE on SVM branch only)
  → Calibrate probabilities (configurable method/cv)
  → Combine 4 models per condition four ways: weighted average, stacked meta-model, rank average, classifier chain
  → Auto-select the best-performing ensemble method per condition
  → Calibrate a 90%-coverage conformal threshold per condition
  → Retrain everything on full training data
  → Predict on test images → predictions.csv (with an uncertain flag per condition)
Built and run in Google Colab, using data stored on Google Drive.

2. Data Handling
Auto-locates the labels CSV and image folders inside the extracted zip.
Drops corrupt/unreadable images, logs them separately rather than silently skipping.
Finds duplicate/near-duplicate images via perceptual hashing, so they don't leak across folds.
A second, independent data-quality pass (new this version) flags feature-space outliers with an IsolationForest, catching images that loaded fine but produced anomalous feature values (e.g. from a failed segmentation) — see the ablation experiment above for what was found when acting on this flag.
3. Image Preprocessing
Every image goes through the same steps: polarity correction (fixes inverted scans), black-border cropping, aspect-ratio-preserving resize to 512×512, intensity normalization, and CLAHE/gamma contrast enhancement for the texture-based features. Each image ends up with a plain ("base") and an enhanced ("texture") version.

4. Lung Region Detection
Classical segmentation (Otsu thresholding + morphology) approximates the lung field, used for the body midline (left/right comparisons), ROI-only features, the clinical costophrenic-angle and lung-shape features, and — new this version — the vessel-density family's left/right split and in-mask restriction. A segmentation_ok flag flags unreliable segmentations rather than hiding them.

5. Feature Extraction — Feature Set H (15 families)
Family	What it measures
Intensity	Brightness, contrast, spread of pixel values
Texture (LBP + GLCM)	Local patterns/roughness
Gradient/Edges	Sharpness and location of edges
Morphology	Shapes, blobs, bright/dark regions
Spatial regions	Grid + top/middle/bottom zone stats
Symmetry	Left lung vs right lung comparison
Wavelet	Fine detail at multiple zoom levels
Frequency (FFT)	Fine detail vs smooth areas
Blob/lesion	Small round spot-like structures
ROI-aware	Same stats, restricted to the lung mask
GLRLM	Run-length of similar intensity — short high-intensity runs as a proxy for nodular texture
Gabor	Multi-frequency, multi-orientation texture response
Hu moments	Rotation/scale/translation-invariant shape descriptors of the lung mask
Clinical/Anatomical	Costophrenic-angle sharpness (per lung) + left-right lung shape/volume ratios
Vessel/Rib Density (new)	Frangi-filter vesselness response — density of linear structures (vessels, bronchial markings, ribs), overall and split left/right
The full column-by-column breakdown, exact feature counts per family, and reasoning behind each one is in the companion Feature Engineering write-up for this version.

6. Feature Cleaning
Drops near-constant features (VarianceThreshold) and highly correlated duplicates (>0.95 correlation). Unchanged from previous versions.

7. Validation Strategy (Leakage-Safe)
GroupKFold with duplicate-image groups: every duplicate/near-duplicate image is assigned a shared group id, so GroupKFold guarantees no group is split across folds.

8. Models
Four classical models trained per condition (One-vs-Rest):

Logistic Regression (L1, tuned C)
HistGradientBoosting (tuned depth/iterations/learning rate)
SVM (RBF, tuned C/gamma)
ExtraTreesClassifier (tuned n_estimators/min_samples_leaf)
Hyperparameters are tuned once via RandomizedSearchCV (per label, per model) and collapsed into a single global setting per model.

Each model's pipeline: mutual-information feature selection (~150 features, fit per training fold) → scaling → BorderlineSMOTE oversampling (SVM branch only by default, based on the V2 ablation) → classifier.

9. Calibration
CalibratedClassifierCV wraps every model so probabilities are trustworthy, not just directionally correct. Calibration method/cv are explicit parameters (defaulting to isotonic for logreg/histgb/extratrees, sigmoid for SVM). Calibration curves (predicted vs. observed frequency per condition) lie close to the diagonal across the achievable probability range for every condition.

10. Ensemble
Four methods are computed per condition, and the best one for that condition is kept:

Weighted average — per-condition weights optimized to minimize log loss.
Stacked meta-model — a Logistic Regression trained on the 4 base models' OOF probabilities, validated with its own CV.
Rank average — averages each model's rank (not raw probability) per condition, then rescales to [0, 1]; more robust when models disagree on calibration scale.
Classifier Chain (new) — models each condition using the other four conditions' predictions as extra input features, averaged over 5 random label orderings; the only method here that explicitly uses label co-occurrence. See the experiment above for what actually won per condition.
All four are compared using OOF ROC-AUC, and the winning method is selected per condition rather than globally.

11. Evaluation
Per-condition and macro-averaged: ROC-AUC, PR-AUC, log loss, Brier score. Final macro ROC-AUC (auto-selected ensemble): 0.676. Best single base model: ExtraTrees, macro ROC-AUC 0.6635.

12. Uncertainty Quantification — Conformal Prediction (new)
A raw probability doesn't say how confident to be about any one case, even when well-calibrated on average. Split conformal prediction calibrates a per-condition threshold on OOF data so a stated coverage level (90%) is a genuine guarantee rather than an average-case statement. At 90% coverage, the fraction of OOF cases the model cannot confidently call either way ranges from 56.4% (Effusion) to 67.5% (Nodule) per condition. Framed as a triage tool rather than an autonomous diagnosis: the model confidently clears roughly a third to just under half of cases per condition and flags the rest for review.

13. Final Model + Test Predictions
All 4 models retrained on full training data, plus the classifier chain ensemble refit on full data. The per-condition ensemble method/weights from validation are reused at test time; where rank averaging won OOF, the pipeline falls back to weighted averaging at test time instead, since rank is relative to whatever batch of images is scored together and can't be reproduced honestly across a different test batch. The same preprocessing/feature extraction is applied to test images (no re-fitting on test data). The 90%-coverage conformal thresholds are applied to test predictions too, adding an uncertain flag per condition. Predictions saved to predictions.csv. Sanity checks compare prediction averages to real prevalence, and check test-vs-train feature distribution shift (KS test).

14. Extra Insight Analyses
Exploratory visualizations for interpretability:

Condition co-occurrence, both as a heatmap and as a network graph (NetworkX, Jaccard-weighted edges).
Conditions-per-patient distribution.
Left-right symmetry breakdown (segmentation-based), plus a raw-pixel laterality check as a cross-check that the segmentation-based symmetry signal isn't an artifact — Effusion stands out as the most laterally asymmetric condition in the dataset (only 39% bilateral vs. 65–88% for every other condition).
A raw-pixel brightness-shift meter per condition vs. normal scans, as a quick sanity signal before trusting the hand-crafted features.
Which feature family each condition relies on most (mutual information family radar), plus a permutation-importance view and, new this version, SHAP summary plots on the final ExtraTrees models — a direction-aware complement showing not just which features matter but which way they push each prediction.
A PCA view of how visually distinct each condition is from a normal scan (no condition forms a clean 2D cluster — the real signal lives in higher-dimensional combinations).
A composite asymmetry-severity index (symmetry + costophrenic + lung-shape features combined) checked against how many conditions a patient has at once — a weak, noisy but directionally-positive trend, reported honestly as suggestive rather than conclusive.
The feature-space outlier ablation above, kept here as a documented, evidence-based decision not to filter training data on this signal.
15. Tools & Libraries
OpenCV, scikit-image, SciPy, PyWavelets, scikit-learn, imbalanced-learn, iterative-stratification, shap, NumPy, pandas, Matplotlib, Seaborn, NetworkX, Plotly/Kaleido, UpSetPlot. No TensorFlow/PyTorch, no CNNs, no pretrained models.

16. How to Run
Open the notebook in Google Colab.
Run the setup cell to install packages and mount Google Drive.
Set ZIP_FILENAME to your dataset zip on Drive.
Run all cells top to bottom — CSV and image folders are auto-detected.
Extracted features are cached to Drive (keyed by feature_set), so re-running won't recompute from scratch.
Final predictions, all figures, and the conformal-flagged predictions.csv are saved to the configured output folder on Drive.
