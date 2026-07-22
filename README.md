# 🩺 DiabetesIQ — Early Risk Detector

A machine learning web app that predicts early-stage diabetes risk based on symptoms. Built with Random Forest, converted to ONNX, and runs entirely in the browser — no backend, no server, instant predictions.

🌐 **[Live Demo → neural-shubh.github.io/diabetes-prediction](https://neural-shubh.github.io/diabetes-prediction)**

---

## 🖼 Preview

> Enter your age, gender, and 14 symptoms → get an instant diabetes risk assessment with confidence score.

---

## ⚙️ How It Works

```
User fills symptom form on the web app
        ↓
ONNX Runtime Web runs the model in the browser
        ↓
Scaler normalizes input → Random Forest predicts
        ↓
Result: Low Risk ✓ or High Risk ⚠ with probability score
```

---

## 📋 Dataset

**Early Stage Diabetes Risk Prediction Dataset**
- 520 patients, 16 features, binary classification
- Source: UCI Machine Learning Repository

| Feature | Type |
|---|---|
| Age | Numeric |
| Gender | Male / Female |
| Polyuria | Yes / No |
| Polydipsia | Yes / No |
| Sudden Weight Loss | Yes / No |
| Weakness | Yes / No |
| Polyphagia | Yes / No |
| Genital Thrush | Yes / No |
| Visual Blurring | Yes / No |
| Itching | Yes / No |
| Irritability | Yes / No |
| Delayed Healing | Yes / No |
| Partial Paresis | Yes / No |
| Muscle Stiffness | Yes / No |
| Alopecia | Yes / No |
| Obesity | Yes / No |

**Target:** `Positive` (diabetic risk) / `Negative` (no risk)

---

## 🤖 Model

| Detail | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Tuning | GridSearchCV (5-fold CV) |
| Export | ONNX (runs client-side via ONNX Runtime Web) |
| Input | 16 features (age, gender, 14 symptoms) |
| Output | Positive / Negative + probability score |

---

## 🧪 ML Pipeline

1. Load Early Stage Diabetes dataset
2. Encode categorical features (Yes/No → 1/0, Male/Female → 1/0)
3. Train/test split (80/20)
4. StandardScaler normalization
5. Train & compare Logistic Regression, SVM, Decision Tree, Random Forest
6. GridSearchCV hyperparameter tuning on Random Forest
7. Export best model + scaler to ONNX

---

## 📁 Project Structure

```
diabetes-prediction/
├── diabetes_prediction.ipynb   # Training notebook (Google Colab)
├── diabetes_model.pkl          # Trained Random Forest model
├── scaler.pkl                  # Fitted StandardScaler
├── app.py                      # Flask app (local use)
├── templates/index.html        # Flask template
├── requirements.txt
└── docs/                       # GitHub Pages web app
    ├── index.html              # Frontend (HTML/CSS/JS + ONNX)
    └── models/
        ├── diabetes_model.onnx
        └── scaler.onnx
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/neural-shubh/diabetes-prediction.git
cd diabetes-prediction
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`

---

## 🛠 Tech Stack

- **ML:** Scikit-learn, Pandas, NumPy, Matplotlib
- **Export:** skl2onnx
- **Frontend:** HTML, CSS, JavaScript, ONNX Runtime Web
- **Hosting:** GitHub Pages
- **Notebook:** Google Colab

---

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. It does not replace professional medical advice. Please consult a doctor for a proper diagnosis.

---
