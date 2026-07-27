"""
Data loader module for ingesting, validating, and profiling the King County House dataset.
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
            raise FileNotFoundError(f"Raw data file not found at: {self.data_path}")
        
        logger.info(f"Loading raw dataset from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Dataset successfully loaded. Shape: {self.df.shape}")
        return self.df

    def get_metadata(self) -> Dict[str, Any]:
        """
        Calculates summary metadata of the loaded dataset.
        
        Returns:
            Dict[str, Any]: Summary statistics dictionary.
        """
        if self.df.empty:
            self.load_data()

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
            "target_mean": float(self.df[TARGET_COL].mean()) if TARGET_COL in self.df.columns else 0.0,
            "target_median": float(self.df[TARGET_COL].median()) if TARGET_COL in self.df.columns else 0.0,
            "target_min": float(self.df[TARGET_COL].min()) if TARGET_COL in self.df.columns else 0.0,
            "target_max": float(self.df[TARGET_COL].max()) if TARGET_COL in self.df.columns else 0.0,
            "target_std": float(self.df[TARGET_COL].std()) if TARGET_COL in self.df.columns else 0.0,
        }
        return metadata
