# 🚨 Real-Time Fraud Detection System (Cost-Sensitive Machine Learning)

A production-style machine learning system that detects fraudulent credit card transactions in real time while minimizing financial loss under extreme class imbalance.

Instead of optimizing for accuracy, this system is designed around **business cost impact**, where missing fraud is significantly more expensive than incorrectly flagging legitimate transactions.

---

## ✨ Key Highlights

* 🧠 Best model: **XGBoost (cost-optimized)**
* 🎯 Business-driven threshold tuning (not default 0.5)
* 📊 Handles extreme class imbalance (~0.17% fraud rate)
* ⚡ Real-time fraud risk scoring via Streamlit dashboard
* 🔍 Compares supervised models + anomaly detection baseline
* 📉 Optimized for financial loss, not just classification metrics

---

## 🧠 Business Problem

Fraud detection is a high-stakes classification problem with extreme class imbalance:

* Fraud cases are rare (~0.17%)
* False negatives (missed fraud) are extremely costly
* False positives increase operational workload

### Objective:

Build a system that minimizes **total expected financial loss**, not just prediction error.

---
## 🚀 Live Deployment

A working end-to-end fraud detection dashboard is deployed here:

👉 https://fraud-detection-system-000.streamlit.app/

You can:
- Upload transaction data (CSV)
- View fraud risk scores in real time
- Adjust decision threshold
- See flagged transactions instantly

---

## 📦 Dataset

* Credit Card Fraud Detection Dataset (European transactions)
* ~284,000 transactions
* Features: `Time`, `Amount`, and PCA-transformed variables `V1–V28`
* Target: `Class` (0 = legitimate, 1 = fraud)

📁 Dataset location:

```text
data/raw/creditcard.csv
```

> Dataset is excluded from the repository due to size and external availability.

---

## 🧠 Models

### Supervised Models

* Logistic Regression (class-weighted)
* Random Forest (balanced sampling)
* XGBoost (scale_pos_weight tuned)

### Unsupervised Baseline

* Isolation Forest (anomaly detection)

---

## ⚙️ Cost-Sensitive Learning

Instead of optimizing accuracy, the system minimizes:

```text
Total Cost =
False Positives × Investigation Cost +
False Negatives × Fraud Loss
```

### Default cost assumptions:

* False Positive Cost: 25
* False Negative Cost: 2500

Configurable in:

```text
src/fraud_detection/config.py
```

---

## 🎯 Threshold Optimization

Rather than using a default 0.5 cutoff, the system selects the optimal decision threshold by:

* Evaluating cost across all thresholds
* Minimizing total financial loss
* Balancing precision and recall under business constraints

### Optimal threshold (XGBoost):

`0.39`

---

## 📊 Evaluation Metrics

Due to severe class imbalance, evaluation focuses on:

* Precision
* Recall
* F1-score
* PR-AUC (primary metric)
* ROC-AUC
* Confusion Matrix
* Business cost at selected threshold

---

## 🏆 Model Performance

| Model               | PR-AUC | ROC-AUC | Recall | Precision |
| ------------------- | ------ | ------- | ------ | --------- |
| XGBoost (Best)      | ~0.86  | ~0.98   | ~0.88  | ~0.63     |
| Random Forest       | ~0.86  | ~0.95   | ~0.89  | ~0.60     |
| Logistic Regression | ~0.71  | ~0.97   | ~0.88  | ~0.48     |

---

## 🏗️ System Architecture

```text
Raw Data
   ↓
Data Preprocessing
   ↓
Stratified Train/Test Split
   ↓
Model Training (3 supervised + 1 baseline)
   ↓
Cost-Based Threshold Optimization
   ↓
Best Model Selection
   ↓
Streamlit Dashboard (real-time scoring)
```

---

## 📁 Project Structure

```text
Fraud_Detection/
│
├── data/                 # Raw dataset
├── models/               # Saved model artifacts
├── notebooks/           # EDA + experimentation
├── reports/             # Visualizations & figures
├── src/fraud_detection/ # Core ML pipeline
├── streamlit_app/       # Dashboard UI
├── tests/               # Unit tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

---

## 🧪 Train Model

```bash
python -m fraud_detection.train --data-path data/raw/creditcard.csv
```

### Outputs:

```text
models/fraud_xgboost.joblib
models/threshold.joblib
models/metrics.json
```

---

## 📊 Run Dashboard (Streamlit)

```bash
streamlit run streamlit_app/app.py
```

### Features:

* Upload transaction CSV
* Fraud risk scoring in real time
* Adjustable decision threshold
* Ranked suspicious transactions

---

## 💡 Key Learnings

* Fraud detection requires **cost-sensitive modeling, not accuracy optimization**
* PR-AUC is more informative than ROC-AUC in extreme imbalance
* Threshold tuning has major business impact
* End-to-end ML systems require reproducible pipelines + deployment layer

---

## 🚀 Future Improvements

* SHAP explainability for fraud decisions
* Real-time streaming pipeline (Kafka-style simulation)
* Model monitoring and drift detection
* API layer for scalable production deployment (optional extension)

---

## 👩‍💻 Author

**Anshika Bhanot**
MSc Business Analytics | Machine Learning & Data Science Enthusiast

### Focus Areas:

* Machine learning for imbalanced classification problems
* Business-driven evaluation and cost-sensitive learning
* End-to-end ML pipeline design
* Deployment using Streamlit

---

## ⭐ Notes

If you find this project useful or interesting, feel free to star the repository.
