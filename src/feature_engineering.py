"""
Feature engineering pipeline module for generating domain-specific housing attributes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from src.logger import logger


class FeatureEngineer:
    """Class to transform raw features and add engineered domain variables."""

    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies domain feature transformations to a given DataFrame.
        
        Args:
            df (pd.DataFrame): Raw or preprocessed King County DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame augmented with new engineered features.
        """
        df_out = df.copy()

        # Parse date column if present
        if "date" in df_out.columns:
            try:
                sale_datetime = pd.to_datetime(df_out["date"])
                df_out["sale_year"] = sale_datetime.dt.year
                df_out["sale_month"] = sale_datetime.dt.month
            except Exception:
                df_out["sale_year"] = 2015
                df_out["sale_month"] = 1
        else:
            df_out["sale_year"] = 2015
            df_out["sale_month"] = 1

        # Calculate house age at sale time
        yr_built = df_out.get("yr_built", pd.Series(1975, index=df_out.index))
        yr_renovated = df_out.get("yr_renovated", pd.Series(0, index=df_out.index))
        
        # Effective built year is renovated year if > 0 else yr_built
        effective_built_yr = np.where(yr_renovated > 0, yr_renovated, yr_built)
        df_out["house_age"] = np.maximum(0, df_out["sale_year"] - effective_built_yr)

        # Renovation binary flag
        df_out["is_renovated"] = np.where(yr_renovated > 0, 1, 0)

        # Ratios & Aggregations
        bedrooms = df_out.get("bedrooms", pd.Series(3, index=df_out.index)).replace(0, 1)
        bathrooms = df_out.get("bathrooms", pd.Series(2, index=df_out.index))
        sqft_living = df_out.get("sqft_living", pd.Series(2000, index=df_out.index))
        sqft_lot = df_out.get("sqft_lot", pd.Series(5000, index=df_out.index)).replace(0, 1)
        sqft_above = df_out.get("sqft_above", pd.Series(1500, index=df_out.index))
        sqft_basement = df_out.get("sqft_basement", pd.Series(500, index=df_out.index))
        sqft_living15 = df_out.get("sqft_living15", pd.Series(2000, index=df_out.index)).replace(0, 1)
        sqft_lot15 = df_out.get("sqft_lot15", pd.Series(5000, index=df_out.index)).replace(0, 1)

        df_out["living_lot_ratio"] = sqft_living / sqft_lot
        df_out["total_sqft"] = sqft_above + sqft_basement
        df_out["bath_bed_ratio"] = bathrooms / bedrooms
        df_out["living_compared_15"] = sqft_living / sqft_living15
        df_out["lot_compared_15"] = sqft_lot / sqft_lot15

        logger.info(f"Engineered 7 new features. Total columns: {len(df_out.columns)}")
        return df_out
