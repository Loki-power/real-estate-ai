"""
Data loader module for ingesting, validating, and profiling the King County House dataset.
Includes pre-computed fallback metadata for Vercel serverless deployments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from src.config import RAW_DATA_PATH, TARGET_COL, ID_COL, DATE_COL
from src.logger import logger


class DataLoader:
    """Class responsible for dataset ingestion, schema validation, and summary metrics."""

    def __init__(self, data_path: Path = RAW_DATA_PATH):
        self.data_path = Path(data_path)
        self.df: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """
        Loads the raw CSV dataset from data_path into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Loaded King County house sales dataset.
        """
        if not self.data_path.exists():
            logger.warning(f"Raw data file not found at: {self.data_path}. Returning empty DataFrame for serverless runtime.")
            return pd.DataFrame()
        
        logger.info(f"Loading raw dataset from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Dataset successfully loaded. Shape: {self.df.shape}")
        return self.df

    def get_metadata(self) -> Dict[str, Any]:
        """
        Calculates summary metadata of the loaded dataset or returns pre-computed stats for Vercel.
        
        Returns:
            Dict[str, Any]: Summary statistics dictionary.
        """
        if self.df.empty and self.data_path.exists():
            self.load_data()

        if not self.df.empty:
            metadata = {
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
                "columns": list(self.df.columns),
                "data_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
                "missing_values": self.df.isnull().sum().to_dict(),
                "total_missing": int(self.df.isnull().sum().sum()),
                "duplicate_rows": int(self.df.duplicated().sum()),
                "unique_houses": int(self.df[ID_COL].nunique()) if ID_COL in self.df.columns else len(self.df),
                "target_col": TARGET_COL,
                "target_mean": float(self.df[TARGET_COL].mean()) if TARGET_COL in self.df.columns else 540088.14,
                "target_median": float(self.df[TARGET_COL].median()) if TARGET_COL in self.df.columns else 450000.0,
                "target_min": float(self.df[TARGET_COL].min()) if TARGET_COL in self.df.columns else 75000.0,
                "target_max": float(self.df[TARGET_COL].max()) if TARGET_COL in self.df.columns else 7700000.0,
                "target_std": float(self.df[TARGET_COL].std()) if TARGET_COL in self.df.columns else 367127.2,
            }
            return metadata

        # Pre-computed King County dataset metadata fallback for Vercel serverless environment
        return {
            "total_rows": 21613,
            "total_columns": 21,
            "columns": [
                "id", "date", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
                "floors", "waterfront", "view", "condition", "grade", "sqft_above",
                "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long",
                "sqft_living15", "sqft_lot15"
            ],
            "total_missing": 0,
            "duplicate_rows": 0,
            "unique_houses": 21436,
            "target_col": TARGET_COL,
            "target_mean": 540088.14,
            "target_median": 450000.0,
            "target_min": 75000.0,
            "target_max": 7700000.0,
            "target_std": 367127.2
        }
