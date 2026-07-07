# Diabetes Prediction

A machine learning project that compares multiple classification algorithms for predicting diabetes outcomes using the Pima Indians Diabetes dataset.

---

## Overview

This project trains and evaluates four classic ML classifiers on clinical features like glucose, BMI, and blood pressure to predict whether a patient has diabetes. It includes preprocessing, model comparison, hyperparameter tuning, and visualization — all inside a Google Colab notebook.

---

## Dataset

- **File:** `diabetes.csv`
- **Target column:** `Outcome` (1 = diabetic, 0 = not diabetic)
- **Features used:**

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (µU/mL) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |

---

## Workflow

1. **Load Data** — Upload CSV via Google Colab or mount from Google Drive
2. **Preprocessing** — Fill missing values (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with column means
3. **Train/Test Split** — 80/20 split with `random_state=14`
4. **Feature Scaling** — StandardScaler applied to all features
5. **Model Training** — Four classifiers trained and evaluated
6. **Hyperparameter Tuning** — GridSearchCV on Random Forest
7. **Visualization** — Bar chart comparing model accuracies

---

## Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| SVM (SVC) | Kernel-based classifier |
| Decision Tree | Interpretable tree model |
| Random Forest | Ensemble method; best candidate for tuning |

---

## Hyperparameter Tuning

GridSearchCV is applied to **Random Forest** with 5-fold cross-validation over:

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
```

---

## Evaluation Metrics

- Accuracy Score
- Classification Report (Precision, Recall, F1)
- Confusion Matrix
- Bar chart of model accuracy comparison

---

## Getting Started

### Run on Google Colab

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Upload `diabetes.csv` when prompted, or mount your Google Drive
3. Run all cells in order

### Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## Project Structure

```
diabetes-prediction/
├── diabetes_prediction.ipynb   # Main notebook
├── diabetes.csv                # Dataset
└── README.md
```

---

## Author

**Shubh** • [GitHub](https://github.com/neural-shubh)
