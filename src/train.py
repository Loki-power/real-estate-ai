"""
Multi-model training engine supporting 11 regressors with automated hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# 11 Regressors
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor

# Gradient boosting frameworks
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

from src.config import MODEL_PARAM_GRIDS
from src.logger import logger


class ModelTrainer:
    """Class to initialize, tune, and train 11 machine learning regressors."""

    def __init__(self):
        self.models_dict = self._init_base_models()

    def _init_base_models(self) -> Dict[str, Any]:
        """Initializes the dictionary of regressor model instances."""
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=42),
            "Lasso Regression": Lasso(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
            "Extra Trees": ExtraTreesRegressor(random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(random_state=42)
        }

        if XGBRegressor is not None:
            models["XGBoost"] = XGBRegressor(random_state=42, verbosity=0)
        
        if LGBMRegressor is not None:
            models["LightGBM"] = LGBMRegressor(random_state=42, verbose=-1)

        if CatBoostRegressor is not None:
            models["CatBoost"] = CatBoostRegressor(random_state=42, verbose=0)

        return models

    def train_and_tune(
        self,
        model_name: str,
        preprocessor: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_tuning: bool = True
    ) -> Pipeline:
        """
        Builds a full Pipeline (preprocessor + estimator) and optionally performs RandomizedSearchCV.
        
        Args:
            model_name (str): Name of the regressor.
            preprocessor (Any): ColumnTransformer instance.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            use_tuning (bool): Whether to perform hyperparameter tuning.
            
        Returns:
            Pipeline: Fitted Scikit-Learn Pipeline object.
        """
        if model_name not in self.models_dict:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(self.models_dict.keys())}")

        base_estimator = self.models_dict[model_name]
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", base_estimator)
        ])

        param_grid = MODEL_PARAM_GRIDS.get(model_name, {})

        if use_tuning and param_grid:
            logger.info(f"Tuning hyperparameters for {model_name}...")
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_grid,
                n_iter=5,
                scoring="neg_root_mean_squared_error",
                cv=3,
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            logger.info(f"Best params for {model_name}: {search.best_params_}")
            return search.best_estimator_
        else:
            logger.info(f"Fitting base pipeline for {model_name}...")
            pipe.fit(X_train, y_train)
            return pipe
