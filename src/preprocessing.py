"""
Data preprocessing engine implementing Scikit-Learn Pipelines for feature scaling,
encoding, outlier handling, and train-test splitting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.config import (
    RAW_NUMERICAL_COLS, RAW_CATEGORICAL_COLS, ENGINEERED_COLS,
    TARGET_COL, ID_COL, DATE_COL, PREPROCESSOR_PATH
)
from src.utils import save_joblib, load_joblib
from src.logger import logger


class DataPreprocessor:
    """Class responsible for data cleaning, outlier handling, and pipeline creation."""

    def __init__(self, target_col: str = TARGET_COL):
        self.target_col = target_col
        self.pipeline: Optional[ColumnTransformer] = None
        self.numerical_cols: List[str] = RAW_NUMERICAL_COLS + ENGINEERED_COLS
        self.categorical_cols: List[str] = RAW_CATEGORICAL_COLS

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicates, drops unnecessary ID/date columns, and fills missing values.
        
        Args:
            df (pd.DataFrame): Input raw DataFrame.
            
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df_clean = df.copy()
        
        # Deduplicate
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        dedup_len = len(df_clean)
        logger.info(f"Removed {initial_len - dedup_len} duplicate rows.")

        # Impute missing values if any
        if df_clean.isnull().sum().sum() > 0:
            for col in df_clean.select_dtypes(include=[np.number]).columns:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            for col in df_clean.select_dtypes(include=["object"]).columns:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

        return df_clean

    def handle_outliers_iqr(self, df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
        """
        Clips extreme numerical outliers using the Interquartile Range (IQR) method.
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame.
            factor (float): Multiplier for IQR bounds.
            
        Returns:
            pd.DataFrame: DataFrame with extreme outliers capped.
        """
        df_out = df.copy()
        clip_cols = ["price", "sqft_living", "sqft_lot", "bedrooms"]
        for col in clip_cols:
            if col in df_out.columns:
                q1 = df_out[col].quantile(0.25)
                q3 = df_out[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - factor * iqr
                upper_bound = q3 + factor * iqr
                df_out[col] = df_out[col].clip(lower=max(0, lower_bound), upper=upper_bound)
        logger.info("Outlier capping complete using IQR thresholding.")
        return df_out

    def build_pipeline(self, feature_df: pd.DataFrame) -> ColumnTransformer:
        """
        Creates a Scikit-Learn ColumnTransformer pipeline for scaling and encoding.
        
        Args:
            feature_df (pd.DataFrame): DataFrame containing feature columns.
            
        Returns:
            ColumnTransformer: Fitted or unfitted transformer pipeline.
        """
        num_cols = [c for c in self.numerical_cols if c in feature_df.columns]
        cat_cols = [c for c in self.categorical_cols if c in feature_df.columns]

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols)
            ],
            remainder="drop"
        )

        self.pipeline = preprocessor
        return preprocessor

    def prepare_train_test_split(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits dataset into feature matrix X and target array y, followed by train/test split.
        
        Args:
            df (pd.DataFrame): Dataset including target and engineered features.
            test_size (float): Proportion of testing data.
            random_state (int): Seed.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
        """
        drop_cols = [self.target_col, ID_COL, DATE_COL, "sale_year", "sale_month"]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        X = df[feature_cols]
        y = df[self.target_col]

        logger.info(f"Preparing split. Features count: {X.shape[1]}, Total samples: {len(X)}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples.")
        return X_train, X_test, y_train, y_test
