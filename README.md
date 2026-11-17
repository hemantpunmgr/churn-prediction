# Predictive Customer Churn & Retention Modeling in Retail Banking

![Orange Data Mining](https://img.shields.io/badge/Tool-Orange_Data_Mining_3.41-orange)
![ML Task](https://img.shields.io/badge/Task-Binary_Classification-blue)
![Domain](https://img.shields.io/badge/Domain-Retail_Banking-green)

## 📌 Project Overview
In Australia's competitive retail banking landscape, low switching barriers and aggressive digital challengers make customer retention a high-priority commercial lever. Blanket retention campaigns waste marketing budget on loyal customers while failing to intervene before at-risk accounts disengage.

This project implements an end-to-end **supervised machine learning pipeline** in **Orange Data Mining** to predict customer attrition, surface key behavioral churn drivers, and guide cost-effective retention targeting.

### Key Highlights
* **Optimal Production Model:** **Logistic Regression** achieved **89.4% accuracy**, **0.860 F1-score**, and **0.282 MCC**, balancing predictive performance with regulatory interpretability.
* **Primary Churn Drivers:** Permutation feature importance revealed that historical engagement metrics (`poutcome`, `pdays`, `campaign`) outweigh static customer demographics (`age`, `job`, `marital`).
* **Projected Commercial Value:** Lifting annual retention by **3–5%** via targeted outreach is estimated to preserve **$500K–$1M AUD** in recurring customer value.

---

## 📊 Dataset & Preprocessing
* **Source:** UCI Bank Marketing benchmark dataset (4,521 records, 17 attributes) used as an APRA/privacy-compliant proxy for retail banking engagement.
* **Target Variable (`y`):** Binary classification (`Yes` = Churn/Disengaged [11.5%], `No` = Retained [88.5%]).
* **Feature Selection:** Removed 4 post-hoc/administrative contact fields (`duration`, `contact`, `day`, `month`) to prevent data leakage, retaining **12 pre-campaign predictors** across demographics, financial profile (`balance`, `housing`, `loan`, `default`), and campaign engagement.
* **Validation Setup:** 70/30 stratified train-test split (3,165 train / 1,356 holdout test) paired with 10-fold stratified cross-validation.

---

## 🤖 Model Benchmarking Results

| Model | Configuration | AUC | Accuracy | F1-Score | Precision | Recall | MCC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | No regularisation | 0.673 | **0.894** | **0.860** | **0.879** | **0.894** | **0.282** |
| **Neural Network** | 1 hidden layer (5 neurons), ReLU, Adam | 0.663 | 0.884 | 0.851 | 0.849 | 0.884 | 0.195 |
| **Random Forest** | 500 trees, max depth 3, 3 splits | **0.687** | 0.885 | 0.831 | 0.783 | 0.885 | 0.000 |
| **Decision Tree** | Binary, max depth 3, min leaf 4 | 0.596 | 0.883 | 0.836 | 0.831 | 0.883 | 0.087 |
| **k-Nearest Neighbours** | $k=5$, Euclidean, uniform weights | 0.547 | 0.876 | 0.835 | 0.817 | 0.876 | 0.069 |

### Why Logistic Regression?
1. **Imbalanced-Class Discrimination:** Outperformed all peers on **F1-score (0.860)** and **MCC (0.282)** across an 88.5% / 11.5% skewed target.
2. **Auditability:** Transparent coefficient weights satisfy financial governance and explainability requirements.
3. **Deployment Simplicity:** Lightweight inference integrates seamlessly into batch CRM scoring workflows.

---

## 🔍 Explainability & Error Analysis
* **Feature Importance:** `poutcome` (previous campaign outcome) was the strongest predictor, followed by `housing`, `loan`, `pdays`, and `campaign`.
* **Confusion Matrix (Holdout Set, $n = 1,356$):** At the default `0.50` threshold, Logistic Regression captured **1,182 True Negatives** and **23 True Positives** (with 18 False Positives and 133 False Negatives).
* **Next-Step Optimization:** Lowering the decision threshold below `0.50` using cost-sensitive tuning (weighing retention offer cost against customer lifetime value) and adding longitudinal transaction velocity features will significantly increase minority-class recall.

---

## 🚀 How to Reproduce
1. Install [Orange Data Mining (v3.41+)](https://orangedatamining.com/) along with the **Explain** add-on (`Options -> Add-ons -> Orange3-Explain`).
2. Open `workflows/banking_churn_pipeline.ows`.
3. Point the `CSV File Import` widget to `data/bank_marketing_sample.csv` and inspect results across `Test and Score`, `Feature Importance`, and `ROC Analysis`.
