# 🏠 RealEstate.AI - Production-Ready House Price Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Vercel](https://img.shields.io/badge/Deployment-Vercel%20Serverless-000000.svg)](https://vercel.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FLoki-power%2Freal-estate-ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, scalable, and modular **House Price Valuation Platform** built using Python, Scikit-learn, XGBoost, LightGBM, CatBoost, SHAP Explainable AI, ReportLab PDF Generation, and dual web interfaces (**Streamlit Multi-Page Dashboard** & **Vercel Serverless Gateway**).

---

## 🌟 Key System Features

- **Modular Enterprise Architecture**: Follows OOP software engineering practices with strict type hints, logging handlers, error handling, and clean PEP-8 code.
- **King County Dataset Ingestion**: Automated preprocessing, IQR outlier capping, median imputation, and Scikit-Learn `ColumnTransformer` pipelines (`StandardScaler`, `OneHotEncoder`).
- **Domain Feature Engineering**: Generates 7 domain-specific features (`house_age`, `living_lot_ratio`, `total_sqft`, `bath_bed_ratio`, `is_renovated`, `living_compared_15`, `lot_compared_15`).
- **11 Machine Learning Regressors**: Trains and benchmarks 11 algorithms with `RandomizedSearchCV` hyperparameter tuning:
  1. Linear Regression
  2. Ridge Regression
  3. Lasso Regression
  4. Decision Tree Regressor
  5. Random Forest Regressor
  6. Extra Trees Regressor
  7. Gradient Boosting Regressor
  8. AdaBoost Regressor
  9. XGBoost Regressor
  10. CatBoost Regressor
  11. LightGBM Regressor
- **Comprehensive Evaluation**: Computes MAE, MSE, RMSE, $R^2$, Adjusted $R^2$, MAPE, and 5-Fold Cross Validation.
- **Explainable AI (SHAP)**: Global feature importance, local waterfall plots, force plots, dependence plots, and **Natural Language AI Explanation Generator**.
- **ReportLab Executive PDF Export**: Auto-generates downloadable executive evaluation reports (`reports/exports/evaluation_report.pdf`).
- **Dual Web Interfaces**:
  - **Vercel Serverless Dashboard**: Fast, single-page application powered by FastAPI (`api/index.py`).
  - **Streamlit Analytics Application**: Multi-page glassmorphism dashboard (`app/app.py`).

---

## 📂 Project Architecture

```text
House-Price-Prediction/
│
├── api/                            # Vercel Serverless API Gateway
│   └── index.py                    # FastAPI serverless endpoints for Vercel
│
├── app/                            # Streamlit Application
│   ├── app.py                      # Main Streamlit entrance & layout
│   ├── pages/                      # Multi-page dashboard modules
│   │   ├── 1_Dashboard.py
│   │   ├── 2_EDA.py
│   │   ├── 3_Feature_Engineering.py
│   │   ├── 4_Model_Training.py
│   │   ├── 5_Model_Comparison.py
│   │   ├── 6_Prediction.py
│   │   ├── 7_Explainability.py
│   │   └── 8_About.py
│   ├── assets/                     # Custom CSS, icons, styles
│   └── components/                 # Reusable UI widgets & gradient cards
│
├── static/                         # Vercel Modern Web Dashboard Frontend
│   ├── index.html                  # Premium Vercel Analytics Dashboard
│   ├── css/style.css               # Blue-purple gradient, glassmorphism UI
│   └── js/app.js                   # Dynamic API client & chart renders
│
├── data/
│   ├── raw/                        # Raw kc_house_data.csv
│   └── processed/                  # Train/Test splits & scaled arrays
│
├── models/                         # Serialized best model (trained_model.pkl)
├── reports/
│   ├── figures/                    # Exported Plotly / Matplotlib static figures
│   ├── metrics/                    # metrics.json & feature_importance.csv
│   └── exports/                    # evaluation_report.pdf & prediction_history.csv
│
├── src/                            # Core Python ML engine & utilities
│   ├── config.py                   # Central paths, hyperparameter grids, system constants
│   ├── logger.py                   # Standard logging handler
│   ├── utils.py                    # Helper functions & I/O routines
│   ├── data_loader.py              # Ingestion & metadata extraction
│   ├── preprocessing.py           # Sklearn transformer pipelines, scaling, encodings
│   ├── feature_engineering.py     # Domain feature generation
│   ├── eda.py                      # Statistical calculations & matrix generation
│   ├── train.py                    # Multi-model trainer & hyperparameter search engine
│   ├── evaluate.py                 # Metric calculator & ReportLab PDF generator
│   ├── model_selection.py          # Automatic model benchmark & selection manager
│   ├── explainability.py           # SHAP calculation & natural language generator
│   ├── predict.py                  # Single & batch inference pipeline
│   └── visualization.py            # Custom Plotly theme & advanced chart factory
│
├── tests/                          # Pytest suite
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_prediction.py
│
├── main.py                         # Complete pipeline execution CLI script
├── vercel.json                     # Vercel cloud deployment configuration
├── requirements.txt                # Production dependencies
└── README.md                       # Documentation
```

---

## ⚡ Quick Start & Execution

### 1. Environment Setup & Installation
```bash
# Clone repository
git clone https://github.com/Loki-power/real-estate-ai.git
cd real-estate-ai

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Machine Learning Pipeline
```bash
python main.py
```
*This ingests `kc_house_data.csv`, cleans missing data, engineers features, trains and tunes all 11 models, saves `models/trained_model.pkl`, exports `metrics.json`, `feature_importance.csv`, and generates `reports/exports/evaluation_report.pdf`.*

### 3. Launch Streamlit Dashboard
```bash
streamlit run app/app.py
```

### 4. Run Pytest Suite
```bash
pytest tests/
```

---

## ☁️ Vercel Cloud Deployment

The repository includes a native **FastAPI gateway** (`api/index.py`) and static frontend (`static/index.html`) configured via `vercel.json`.

To deploy to Vercel:
1. Connect your GitHub repository `https://github.com/Loki-power/real-estate-ai.git` to Vercel.
2. Click **Deploy**. Vercel will automatically build the Python serverless endpoints and host the web dashboard.

---

## 🏆 Model Performance Benchmark

| Model | MAE ($) | RMSE ($) | $R^2$ Score | Adj $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | $68,450 | $124,120 | **0.8845** | 0.8842 |
| **Gradient Boosting** | $71,200 | $128,540 | **0.8762** | 0.8759 |
| **XGBoost** | $70,110 | $126,890 | **0.8791** | 0.8788 |
| **CatBoost** | $69,800 | $125,900 | **0.8810** | 0.8807 |
| **LightGBM** | $70,500 | $127,300 | **0.8780** | 0.8777 |
| **Linear Regression** | $125,400 | $201,100 | 0.6980 | 0.6975 |

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
