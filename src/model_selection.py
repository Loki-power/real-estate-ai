"""
Automated model selection engine for benchmarking 11 regressors and saving the best pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from src.train import ModelTrainer
from src.evaluate import Evaluator
from src.config import BEST_MODEL_PATH, METRICS_JSON_PATH
from src.utils import save_joblib, save_json
from src.logger import logger


class ModelSelector:
    """Automates model training, benchmarking, comparison, and best model selection."""

    def __init__(self, trainer: ModelTrainer, evaluator: Evaluator):
        self.trainer = trainer
        self.evaluator = evaluator

    def run_benchmark(
        self,
        preprocessor: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Tuple[str, Any, pd.DataFrame, Dict[str, Any]]:
        """
        Trains and compares all 11 regressors, returning the champion model.
        
        Returns:
            Tuple[str, Any, pd.DataFrame, Dict[str, Any]]: (best_model_name, best_pipeline, leaderboard_df, all_metrics_dict)
        """
        results = []
        trained_pipelines = {}
        all_metrics_dict = {}

        num_features = X_train.shape[1]

        for model_name in self.trainer.models_dict.keys():
            logger.info(f"--- Benchmarking Model: {model_name} ---")
            try:
                # Fit model
                pipeline = self.trainer.train_and_tune(
                    model_name=model_name,
                    preprocessor=preprocessor,
                    X_train=X_train,
                    y_train=y_train,
                    use_tuning=True
                )

                # Predict on test set
                y_pred = pipeline.predict(X_test)

                # Calculate metrics
                metrics = self.evaluator.evaluate_predictions(y_test, y_pred, num_features=num_features)
                metrics["Model"] = model_name

                results.append(metrics)
                trained_pipelines[model_name] = pipeline
                all_metrics_dict[model_name] = metrics

            except Exception as e:
                logger.error(f"Failed to benchmark model {model_name}: {e}")

        # Build Leaderboard DataFrame
        leaderboard_df = pd.DataFrame(results).sort_values(by="R2_Score", ascending=False).reset_index(drop=True)
        logger.info(f"Model Benchmark Complete. Leaderboard:\n{leaderboard_df[['Model', 'MAE', 'RMSE', 'R2_Score']]}")

        # Pick best model
        best_model_name = leaderboard_df.iloc[0]["Model"]
        best_pipeline = trained_pipelines[best_model_name]
        best_metrics = all_metrics_dict[best_model_name]

        # Save artifacts
        save_joblib(best_pipeline, BEST_MODEL_PATH)
        save_json(all_metrics_dict, METRICS_JSON_PATH)

        return best_model_name, best_pipeline, leaderboard_df, all_metrics_dict
