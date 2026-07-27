"""
EDA analytics engine providing statistical calculations and matrix generation for the KC House dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.config import TARGET_COL
from src.logger import logger


class EDAEngine:
    """Calculates statistics, correlation matrices, and distribution metrics for EDA."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_summary_table(self) -> pd.DataFrame:
        """Generates summary statistics table with mean, std, min, median, max, skewness."""
        num_df = self.df.select_dtypes(include=[np.number])
        stats = num_df.describe().T[["mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median"})
        stats["skewness"] = num_df.skew()
        stats["kurtosis"] = num_df.kurt()
        stats["missing"] = self.df[num_df.columns].isnull().sum()
        return stats.round(2)

    def get_correlation_matrix((self)) -> pd.DataFrame:
        """Returns correlation matrix of all numeric columns."""
        num_df = self.df.select_dtypes(include=[np.number])
        if "id" in num_df.columns:
            num_df = num_df.drop(columns=["id"])
        return num_df.corr()

    def get_top_features_correlation(self, target_col: str = TARGET_COL, top_n: int = 10) -> pd.Series:
        """Returns top_n features most correlated with the target variable."""
        corr = self.get_correlation_matrix()
        if target_col in corr.columns:
            target_corr = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
            return target_corr.head(top_n)
        return pd.Series(dtype=float)

    def get_zipcode_analytics(self) -> pd.DataFrame:
        """Aggregates average price, count, and sqft by zipcode."""
        if "zipcode" not in self.df.columns or TARGET_COL not in self.df.columns:
            return pd.DataFrame()
        
        agg = self.df.groupby("zipcode").agg(
            avg_price=(TARGET_COL, "mean"),
            median_price=(TARGET_COL, "median"),
            house_count=(TARGET_COL, "count"),
            avg_sqft=("sqft_living", "mean")
        ).reset_index().sort_values(by="avg_price", ascending=False)
        return agg.round(2)
