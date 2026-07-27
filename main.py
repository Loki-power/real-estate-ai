"""
Main execution CLI script for the end-to-end House Price Prediction machine learning pipeline.
Executes ingestion, preprocessing, feature engineering, 11-model benchmark tuning,
evaluation, PDF report generation, and artifact export.
"""

import sys
from pathlib import Path

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.evaluate import Evaluator
from src.model_selection import ModelSelector
from src.explainability import Explainer
from src.logger import logger


def main():
    """Runs the full automated machine learning pipeline."""
    logger.info("=========================================================")
    logger.info("Starting House Price Prediction Pipeline Execution")
    logger.info("=========================================================")

    # 1. Ingestion
    loader = DataLoader()
    raw_df = loader.load_data()
    metadata = loader.get_metadata()
    logger.info(f"Loaded dataset with {metadata['total_rows']} rows and {metadata['total_columns']} columns.")

    # 2. Preprocessing & Cleaning
    preprocessor_mgr = DataPreprocessor()
    clean_df = preprocessor_mgr.clean_data(raw_df)
    clean_df = preprocessor_mgr.handle_outliers_iqr(clean_df)

    # 3. Feature Engineering
    engineer = FeatureEngineer()
    engineered_df = engineer.transform(clean_df)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = preprocessor_mgr.prepare_train_test_split(engineered_df)

    # 5. Build ColumnTransformer Pipeline
    column_preprocessor = preprocessor_mgr.build_pipeline(X_train)

    # 6. Train & Benchmark 11 Regressors
    trainer = ModelTrainer()
    evaluator = Evaluator()
    selector = ModelSelector(trainer=trainer, evaluator=evaluator)

    logger.info("Benchmarking 11 machine learning regression algorithms...")
    best_model_name, best_pipeline, leaderboard_df, all_metrics = selector.run_benchmark(
        preprocessor=column_preprocessor,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    logger.info(f"BEST MODEL SELECTED: {best_model_name}")

    # 7. Generate PDF Report
    best_metrics = all_metrics[best_model_name]
    pdf_path = evaluator.generate_pdf_report(
        best_model_name=best_model_name,
        leaderboard_df=leaderboard_df,
        best_metrics=best_metrics
    )
    logger.info(f"PDF Evaluation Report created at: {pdf_path}")

    # 8. SHAP Feature Importance Export
    explainer = Explainer(best_pipeline)
    sample_shap_df = X_test.sample(min(200, len(X_test)), random_state=42)
    explainer.fit_shap(sample_shap_df)
    explainer.export_feature_importance(sample_shap_df)

    logger.info("=========================================================")
    logger.info("Pipeline Execution Completed Successfully!")
    logger.info("=========================================================")


if __name__ == "__main__":
    main()
