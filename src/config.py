"""
Configuration settings, file paths, model parameters, and hyperparameter grids
for the House Price Prediction system.
"""

from pathlib import Path
from typing import Dict, Any, List

# --- Directory Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "kc_house_data.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "trained_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
EXPORTS_DIR = REPORTS_DIR / "exports"

METRICS_JSON_PATH = METRICS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = METRICS_DIR / "feature_importance.csv"
EVALUATION_REPORT_PDF_PATH = EXPORTS_DIR / "evaluation_report.pdf"
PREDICTION_HISTORY_CSV_PATH = EXPORTS_DIR / "prediction_history.csv"

# Ensure directories exist (safe for read-only serverless filesystems like Vercel)
for path_dir in [PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR, EXPORTS_DIR]:
    try:
        path_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        pass

# --- Feature Column Definitions ---
TARGET_COL = "price"
ID_COL = "id"
DATE_COL = "date"

RAW_NUMERICAL_COLS: List[str] = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
    "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"
]

RAW_CATEGORICAL_COLS: List[str] = [
    "waterfront", "view", "condition", "grade"
]

ENGINEERED_COLS: List[str] = [
    "house_age", "is_renovated", "living_lot_ratio",
    "total_sqft", "bath_bed_ratio", "living_compared_15", "lot_compared_15"
]

ALL_FEATURE_COLS: List[str] = RAW_NUMERICAL_COLS + RAW_CATEGORICAL_COLS + ENGINEERED_COLS

# --- UI Theme Color Palette (Blue-Purple Glassmorphism Gradient) ---
THEME_COLORS = {
    "primary": "#6366f1",        # Indigo/Purple
    "secondary": "#3b82f6",      # Bright Blue
    "accent": "#8b5cf6",         # Deep Purple
    "background_dark": "#0f172a",# Dark Slate
    "card_dark": "#1e293b",      # Slate Card
    "text_dark": "#f8fafc",      # Light text
    "background_light": "#f8fafc",
    "card_light": "#ffffff",
    "text_light": "#0f172a",
    "gradient": "linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #2563eb 100%)",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444"
}

# --- Hyperparameter Search Grids for 11 Regressors ---
MODEL_PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    "Linear Regression": {},
    "Ridge Regression": {
        "model__alpha": [0.1, 1.0, 10.0, 100.0]
    },
    "Lasso Regression": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0]
    },
    "Decision Tree": {
        "model__max_depth": [5, 10, 15, None],
        "model__min_samples_split": [2, 5, 10]
    },
    "Random Forest": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5]
    },
    "Extra Trees": {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [10, 20, None]
    },
    "Gradient Boosting": {
        "model__n_estimators": [50, 100, 150],
        "model__learning_rate": [0.03, 0.1, 0.2],
        "model__max_depth": [3, 5, 7]
    },
    "AdaBoost": {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.05, 0.1, 0.5]
    },
    "XGBoost": {
        "model__n_estimators": [50, 100, 150],
        "model__learning_rate": [0.03, 0.1, 0.2],
        "model__max_depth": [3, 6, 8]
    },
    "CatBoost": {
        "model__iterations": [100, 200],
        "model__learning_rate": [0.03, 0.1],
        "model__depth": [4, 6]
    },
    "LightGBM": {
        "model__n_estimators": [50, 100, 150],
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [5, 10, -1]
    }
}
