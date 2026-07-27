"""
Unit tests for FeatureEngineer domain transformations.
"""

import pytest
import pandas as pd
from src.feature_engineering import FeatureEngineer


def test_feature_engineer_transformations():
    """Verify engineered domain columns are created."""
    sample_df = pd.DataFrame([{
        "date": "20141013T000000",
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 2000,
        "sqft_lot": 5000,
        "sqft_above": 1500,
        "sqft_basement": 500,
        "yr_built": 1990,
        "yr_renovated": 0,
        "sqft_living15": 1800,
        "sqft_lot15": 4800
    }])

    fe = FeatureEngineer()
    df_out = fe.transform(sample_df)

    assert "house_age" in df_out.columns
    assert "is_renovated" in df_out.columns
    assert "living_lot_ratio" in df_out.columns
    assert "total_sqft" in df_out.columns
    assert df_out["total_sqft"].iloc[0] == 2000
    assert df_out["is_renovated"].iloc[0] == 0
