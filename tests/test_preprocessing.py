"""
Unit tests for DataPreprocessor class.
"""

import pytest
import pandas as pd
from src.preprocessing import DataPreprocessor


def test_clean_data_deduplication():
    """Verify deduplication removes duplicate rows."""
    sample_df = pd.DataFrame([
        {"price": 100, "bedrooms": 2},
        {"price": 100, "bedrooms": 2}
    ])

    dp = DataPreprocessor()
    df_clean = dp.clean_data(sample_df)
    assert len(df_clean) == 1
