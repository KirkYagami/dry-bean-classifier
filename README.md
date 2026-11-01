# Dry Bean Variety Classification

**Live app:** https://dry-bean-classifier-2026ak05520.streamlit.app


## a. Problem statement

Dry beans are sold by variety, and sorting them by eye is slow and inconsistent.
Given 16 shape measurements extracted from a photo of a single bean grain, predict
which of **7 registered dry-bean varieties** it belongs to. This is a **multi-class
classification** problem.

Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian
Naive Bayes and Support Vector Machine) are trained on the same data, compared on six
evaluation metrics, and served through an interactive Streamlit app that scores
uploaded test data with the saved models.

## b. Dataset description

| | |
|---|---|
| **Source** | UCI Machine Learning Repository: [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) (Koklu & Ozkan, 2020, *Computers and Electronics in Agriculture* 174) |
| **Instances** | 13,611 bean images; **13,543** after removing 68 exact duplicate rows (all 68 were HOROZ) |
| **Features** | **16**, all numeric, no missing values |
| **Target** | `Class`, 7 varieties |
| **Split** | Stratified 80 % train (10,834) / 20 % test (2,709), `random_state=42` |

**Features.** Twelve are dimensional measurements of the grain outline: `Area`,
`Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation` (aspect ratio),
`Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `roundness` and
`Compactness`. The other four are shape factors (`ShapeFactor1`–`ShapeFactor4`)
derived from those measurements. Feature scales differ a lot (`Area` is around
20,000–255,000 while `ShapeFactor2` is around 0.0006–0.0037). Many features are
almost duplicates of each other: 15 of the 120 feature pairs have |correlation| > 0.95,
for example `Area`/`ConvexArea` at 0.9999.

**Class balance (full dataset after de-duplication):** DERMASON 3,546 · SIRA 2,636 ·
SEKER 2,027 · HOROZ 1,860 · CALI 1,630 · BARBUNYA 1,322 · BOMBAY 522. The classes are
imbalanced: BOMBAY is about 4 % of the data and DERMASON about 26 %, which is why the
split is stratified and precision/recall/F1 are macro-averaged.

`test_data.csv` in this repo is the 2,709-row hold-out test split (16 features +
`Class`). It is the file to upload in the app.

## c. GitHub repository link

https://github.com/KirkYagami/dry-bean-classifier

```
├── app.py               Streamlit app (loads saved models, never retrains)
├── train_models.py      trains + tunes all five models, writes model/ and test_data.csv
├── evaluation.py        shared metric code used by both scripts
├── requirements.txt
├── test_data.csv        hold-out test set (upload this in the app)
├── data/
│   └── Dry_Bean_Dataset.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metrics.csv      test-set metrics for all five models
```

Each `.joblib` file is a complete scikit-learn `Pipeline`. It includes the fitted
`StandardScaler` where the model needs one, so no separate preprocessing artifact
is required.

## d. Models used

All five models are trained on the same training split. Hyper-parameters are chosen
by 5-fold stratified cross-validation on the training split only, scored by macro F1.
Each model is then evaluated **once** on the untouched test split.

| Model | Preprocessing | Tuned hyper-parameters (selected) |
|---|---|---|
| Logistic Regression | StandardScaler | C = 1000 |
| Decision Tree | none | criterion = entropy, max_depth = 10, min_samples_leaf = 10 |
| KNN | StandardScaler | k = 21, weights = distance |
| Naive Bayes (Gaussian) | none | var_smoothing = 1e-11 |
| SVM (RBF kernel) | StandardScaler | C = 10, gamma = scale; probabilities via sigmoid calibration |

**Metric definitions (multi-class):** AUC is one-vs-rest ROC AUC, macro-averaged over
the 7 classes. Precision, Recall and F1 are macro-averaged, so every variety counts
equally regardless of its size. MCC is the multi-class Matthews correlation
coefficient.

### Comparison table (test set, 2,709 beans)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9173 | **0.9949** | 0.9289 | 0.9283 | 0.9285 | 0.9001 |
| Decision Tree | 0.9022 | 0.9749 | 0.9141 | 0.9126 | 0.9132 | 0.8817 |
| KNN | 0.9192 | 0.9912 | 0.9334 | 0.9294 | 0.9309 | 0.9023 |
| Naive Bayes | 0.8605 | 0.9839 | 0.8729 | 0.8652 | 0.8661 | 0.8321 |
| Support Vector Machine | **0.9240** | 0.9904 | **0.9360** | **0.9344** | **0.9351** | **0.9080** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression achieved 91.73% accuracy. Its linear boundaries worked well with the scaled features, suggesting that a linear model captures much of the separation between varieties. It had the highest AUC (0.9949), meaning its probabilities ranked beans well for each variety against the rest, even though its top predictions were less accurate than SVM and KNN. |
| Decision Tree | Decision Tree achieved 90.22% accuracy and had the lowest AUC (0.9749). It gives every bean in a leaf the same probability, which creates ties and can limit its ranking performance. Its one-feature-at-a-time splits also need several steps to approximate diagonal or curved boundaries. The selected depth limit of 10 helps control overfitting, but SIRA remained difficult, with 82.9% recall. |
| KNN | KNN achieved 91.92% accuracy and was second on F1 and MCC. It classifies beans using nearby training examples, so scaling helps prevent large-valued features such as Area from dominating the distances. With 21 neighbours and distance weighting, it had the highest CALI recall (96.3%). A trade-off is that it keeps the training examples, so prediction can become more expensive as the dataset grows. |
| Naive Bayes | Naive Bayes had the lowest accuracy (86.05%), and BARBUNYA recall was only 64.5%. It assumes features are independent within each class, but several measurements contain similar information: 15 of 120 feature pairs had absolute correlation above 0.95, including Area and ConvexArea at 0.9999. This suggests repeated size information could hurt the model, although overall correlations alone do not prove dependence within each class. Its AUC (0.9839) still exceeded the tree's, showing that ranking and top-label accuracy can tell different stories. |
| Support Vector Machine | SVM achieved the highest accuracy (92.40%) and made the fewest errors. Its RBF kernel can model curved boundaries, while the margin and regularisation help balance fitting the data with generalising to new beans. It led on precision, recall, F1 and MCC. Its AUC (0.9904) was below Logistic Regression and KNN. The probabilities come from a separate sigmoid calibration step, but this alone does not establish why its AUC was lower. |
| Overall, Winner for your dataset? | I would choose SVM because it performed best on five of the six metrics. Its lead over KNN and Logistic Regression was small on this test split, and Logistic Regression had the highest AUC. All five models made repeated mistakes between SIRA and DERMASON, and between BARBUNYA and CALI. This suggests these pairs are difficult to distinguish with the available shape measurements, but it does not prove that another model or better features could not improve the results. |

## Streamlit app

- **Test data:** upload a CSV with the 16 feature columns and `Class` (the bundled
  `test_data.csv` loads by default and can be downloaded from the sidebar)
- **Model selection:** dropdown with all five classifiers
- **Evaluation metrics:** the six metrics for the selected model
- **Confusion matrix and classification report:** heatmap (counts or % of each actual
  variety) and per-variety precision / recall / F1
- **Comparison:** all five models on the uploaded data, plus a per-metric chart
- **Predictions:** row-level predictions with confidence, filterable to the errors and
  downloadable as CSV

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_models.py      # optional: re-creates model/ and test_data.csv (~30 s)
streamlit run app.py
```
