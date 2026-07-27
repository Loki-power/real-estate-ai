# 🏠 RealEstate.AI - House Price Prediction Platform (Vercel Production Architecture)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Vercel](https://img.shields.io/badge/Deployment-Vercel%20Serverless-000000.svg)](https://vercel.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FLoki-power%2Freal-estate-ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, scalable, and modular **House Price Valuation Platform** built using Python, Scikit-learn, XGBoost, LightGBM, ReportLab PDF Generation, and **Vercel Serverless Architecture (FastAPI + Modern Glassmorphism Web Dashboard)**.

---

## 🌟 Key System Features

- **Modular Enterprise Architecture**: Follows OOP software engineering practices with strict type hints, logging handlers, error handling, and clean PEP-8 code.
- **King County Dataset Ingestion**: Automated preprocessing, IQR outlier capping, median imputation, and Scikit-Learn `ColumnTransformer` pipelines (`StandardScaler`, `OneHotEncoder`).
- **Domain Feature Engineering**: Generates 7 domain-specific features (`house_age`, `living_lot_ratio`, `total_sqft`, `bath_bed_ratio`, `is_renovated`, `living_compared_15`, `lot_compared_15`).
- **Machine Learning Regressors Benchmark**: Trains and benchmarks ML algorithms with `RandomizedSearchCV` hyperparameter tuning (Random Forest, Gradient Boosting, Extra Trees, Decision Tree, Ridge, Lasso, Linear Regression).
- **Comprehensive Evaluation**: Computes MAE, MSE, RMSE, $R^2$, Adjusted $R^2$, MAPE, and 5-Fold Cross Validation.
- **Explainable AI**: Global feature importance and **Natural Language AI Explanation Generator**.
- **ReportLab Executive PDF Export**: Auto-generates downloadable executive evaluation reports (`reports/exports/evaluation_report.pdf`).
- **Vercel Serverless Architecture**:
  - **FastAPI Gateway (`api/index.py`)**: Zero-cold-start REST serverless API.
  - **Modern Web Frontend (`static/index.html`)**: Glassmorphism dashboard with sliders, Plotly.js charts, gauge meters, and dark/light theme toggle.

---

## 📂 Project Architecture

```text
House-Price-Prediction/
│
├── api/                            # Vercel Serverless API Gateway
│   └── index.py                    # FastAPI serverless endpoints for Vercel
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
│   ├── figures/                    # Exported static figures
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
│   ├── explainability.py           # Feature importance & natural language generator
│   ├── predict.py                  # Single & batch inference pipeline
│   └── visualization.py            # Custom Plotly theme & chart factory
│
├── tests/                          # Pytest suite
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_prediction.py
│
├── main.py                         # Complete pipeline execution CLI script
├── vercel.json                     # Vercel cloud deployment configuration
├── .python-version                 # Python 3.12 version declaration
├── pyproject.toml                  # Project requirements configuration
├── requirements.txt                # Production dependencies
└── README.md                       # Documentation
```

---

## ⚡ Quick Start & Execution

### 1. Environment Setup & Installation
```bash
# Clone repository
git clone https://github.com/Loki-power/smart-real-estate-ai.git
cd smart-real-estate-ai

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Machine Learning Pipeline
```bash
python main.py
```

### 3. Run Pytest Suite
```bash
python -m pytest tests/
```

---

## ☁️ Vercel Cloud Deployment

The repository includes a native **FastAPI gateway** (`api/index.py`) and static frontend (`static/index.html`) configured via `vercel.json`.

To deploy to Vercel:
1. Connect your GitHub repository `https://github.com/Loki-power/smart-real-estate-ai.git` to Vercel.
2. Click **Deploy**. Vercel will automatically build the Python serverless endpoints and host the web dashboard in seconds!

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
