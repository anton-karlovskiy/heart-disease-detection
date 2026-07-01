# Predicting Heart Disease using Machine Learning

An end-to-end machine learning **proof of concept** that predicts whether or not a
patient has heart disease based on their clinical parameters.

> *Given clinical parameters about a patient, can we predict whether or not they have heart disease?*

This is a **binary classification** problem: given a number of features (health
characteristics) about a person, the model predicts one of two classes — heart
disease (`1`) or no heart disease (`0`).

## Project structure

```
heart-disease-detection/
├── README.md
├── requirements.txt
├── data/
│   └── heart-disease.csv        # dataset (used by the main notebook via ../data/)
├── images/                      # figures referenced by the main notebook
│   ├── sklearn-cross-validation.png
│   └── sklearn-ml-map-cheatsheet-heart-disease-ensemble.png
└── notebooks/
    ├── end-to-end-heart-disease-classification.ipynb        # main (detailed) notebook
    ├── end-to-end-heart-disease-classification-video.ipynb  # video walkthrough version
    └── heart-disease.csv        # dataset copy (video notebook loads it same-dir)
```

Both notebooks are included unmodified:

- **`end-to-end-heart-disease-classification.ipynb`** — the full, detailed notebook
  with expanded explanations, EDA, evaluation and hyperparameter tuning.
- **`end-to-end-heart-disease-classification-video.ipynb`** — the version that follows
  along with the course video.

The dataset ships in two locations so each notebook runs as-is: `data/heart-disease.csv`
for the main notebook (`../data/heart-disease.csv`) and `notebooks/heart-disease.csv`
for the video notebook (`heart-disease.csv`).

## The data

The dataset originates from the [Cleveland database](https://archive.ics.uci.edu/dataset/45/heart+disease)
of the UCI Machine Learning Repository, downloaded in a formatted way from
[Kaggle](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset).
It contains 303 patient records with 13 features and 1 target column.

| Feature   | Description |
|-----------|-------------|
| age       | Age in years |
| sex       | Sex (1 = male, 0 = female) |
| cp        | Chest pain type (0–3) |
| trestbps  | Resting blood pressure (mm Hg) |
| chol      | Serum cholesterol (mg/dl) |
| fbs       | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| restecg   | Resting electrocardiographic results (0–2) |
| thalach   | Maximum heart rate achieved |
| exang     | Exercise-induced angina (1 = yes, 0 = no) |
| oldpeak   | ST depression induced by exercise relative to rest |
| slope     | Slope of the peak exercise ST segment (0–2) |
| ca        | Number of major vessels (0–3) colored by fluoroscopy |
| thal      | Thalium stress result |
| **target**| **Presence of heart disease (1 = yes, 0 = no)** |

## Approach

The project follows a 6-step machine learning modelling framework: problem
definition, data, evaluation, features, modelling, and experimentation. It covers:

- **Exploratory data analysis (EDA)** — understanding the dataset.
- **Model training** — comparing `LogisticRegression`, `KNeighborsClassifier`
  and `RandomForestClassifier`.
- **Hyperparameter tuning** — with `RandomizedSearchCV` and `GridSearchCV`.
- **Cross-validation** — via `cross_val_score`.
- **Model evaluation** — confusion matrix, classification report, ROC curve,
  precision, recall and F1-score.

## Getting started

1. Create and activate a virtual environment (optional but recommended).
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter and open a notebook:

   ```bash
   jupyter notebook
   ```

   Open `notebooks/end-to-end-heart-disease-classification.ipynb` (detailed) or
   `notebooks/end-to-end-heart-disease-classification-video.ipynb` (video walkthrough)
   and run the cells top to bottom.

## Credits

Based on the "Complete A.I. Machine Learning and Data Science: Zero to Mastery"
course material by [Daniel Bourke](https://github.com/mrdbourke/zero-to-mastery-ml).
