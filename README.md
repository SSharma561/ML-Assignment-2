# ML Assignment 2 – Breast Cancer Classification with 5 ML Models

**Live Streamlit app:** <Streamlit-app>
**GitHub repository:** <GITHUB-LInk>

---

## a. Problem statement

Breast cancer is one of the most common cancers in women, and catching malignant tumours early improves outcomes. The goal of this project is to **predict whether a breast tumour is Malignant (1) or Benign (0)** from 30 numeric features computed from a digitised image of a fine-needle aspirate (FNA) of a breast mass.

Five classification models (Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine) are trained on the same dataset. Each is evaluated with Accuracy, AUC, Precision, Recall, F1 and MCC. The models are then served through an interactive Streamlit web app deployed on Streamlit Community Cloud.

## b. Dataset description

| Property | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository – https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| Instances | 569 (requirement: ≥ 500) |
| Features | 30 numeric, real-valued (requirement: ≥ 12) |
| Target | `diagnosis` – 1 = Malignant (212 rows, 37.3%), 0 = Benign (357 rows, 62.7%) |
| Problem type | Binary classification |
| Missing values | None |

For each cell nucleus, ten characteristics are measured: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Each one is reported three ways, as the **mean**, the **standard error** and the **worst** (mean of the three largest values). That gives 10 × 3 = 30 features, for example `mean_radius`, `radius_error` and `worst_radius`.

**Preprocessing**
- Column names cleaned (spaces → underscores); target re-coded so that 1 = Malignant.
- Stratified 80/20 train-test split (`random_state=42`): **455 train / 114 test** rows. The test split is saved as `test_data.csv`.
- `StandardScaler` is fitted on the training data only. It sits inside each model's `Pipeline`, so it is saved in every `.pkl` file.

## c. GitHub repository

<PASTE-YOUR-GITHUB-REPO-LINK-HERE>

```
ml-assignment-2/
│-- app.py                  # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv           # 114-row test split (features + diagnosis)
│-- data/breast_cancer.csv  # full dataset
│-- src/
│   │-- download_data.py    # creates data/breast_cancer.csv
│   │-- eda.py              # quick EDA + plots in reports/
│   │-- preprocessing.py    # load / split / save test data
│   │-- models.py           # the 5 model pipelines
│   │-- metrics.py          # Accuracy, AUC, Precision, Recall, F1, MCC
│   └-- train_models.py     # trains + saves all models
│-- model/
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- svm.pkl
│   │-- feature_names.json
│   └-- metrics.csv
└-- reports/                # EDA plots
```

**Run locally**
```bash
pip install -r requirements.txt
python src/download_data.py
python src/train_models.py
streamlit run app.py
```

## d. Models used and comparison

All metrics are computed on the held-out **test set (114 rows)**. The positive class is Malignant (1).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a simple linear model: 96.5% accuracy and the **highest AUC (0.996)**, so it ranks malignant cases above benign ones almost perfectly. This suggests the two classes are close to linearly separable once the features are scaled. It missed 3 of the 42 malignant cases (recall 0.929). |
| Decision Tree | The **weakest model** overall (accuracy 92.1%, lowest AUC 0.945 and lowest MCC 0.830). Its recall of 0.833 means it missed 7 malignant tumours, which is costly in a medical setting. A single tree with axis-parallel splits is high-variance and cannot capture smooth boundaries well, even with `max_depth=5`. |
| KNN | Good performance (95.6% accuracy, F1 0.938), helped by feature scaling, which distance-based KNN depends on. It is slightly behind LR and SVM because 30 dimensions dilute distance information. It is also slower at prediction time, since it compares each sample with all training points. |
| Naive Bayes | Same accuracy as the Decision Tree (92.1%) but a much better AUC (0.989), so its probability ranking is good while its 0.5 threshold decisions are weaker. The independence assumption is clearly violated here: radius, perimeter and area are highly correlated. As a result it has the lowest precision (0.923) and the lowest MCC. |
| Support Vector Machine | **Best model overall**: highest Accuracy (97.4%), F1 (0.963) and MCC (0.944), and **precision of 1.0** (no benign tumour was wrongly flagged as malignant). The RBF kernel finds a maximum-margin boundary that generalises well on this small, scaled dataset. Its recall equals LR's (0.929). |

**Overall winner: Support Vector Machine (RBF kernel).** It has the best Accuracy, F1 and MCC, and MCC is the most reliable single metric when classes are imbalanced (37/63). Logistic Regression is a very close second and has the best AUC, so it would be a good choice when interpretability matters. The Decision Tree is the weakest model for this dataset.

## e. Streamlit app features

1. **Dataset upload (CSV):** upload test data with all 30 features and the `diagnosis` column. A sample `test_data.csv` can be downloaded from the sidebar.
2. **Model selection dropdown:** choose any of the 5 saved models.
3. **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
4. **Confusion matrix and classification report:** both shown for the selected model.
5. **Compare all models:** a tab showing a side-by-side table and bar chart of all 5 models.

The app only **loads** the saved `.pkl` models. It never retrains on the uploaded data.