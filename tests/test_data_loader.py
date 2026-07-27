"""
Unit tests for DataLoader ingestion module.
"""

import pytest
import pandas as pd
from src.data_loader import DataLoader
from src.config import RAW_DATA_PATH


def test_data_loader_file_exists():
    """Verify raw dataset file exists."""
    assert RAW_DATA_PATH.exists()


def test_data_loader_schema():
    """Verify dataset loads and contains required columns."""
    loader = DataLoader(RAW_DATA_PATH)
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "price" in df.columns
    assert "bedrooms" in df.columns
    assert "sqft_living" in df.columns


def test_data_loader_metadata():
    """Verify metadata dictionary output."""
    loader = DataLoader(RAW_DATA_PATH)
    meta = loader.get_metadata()
    assert meta["total_rows"] > 20000
    assert meta["target_col"] == "price"
