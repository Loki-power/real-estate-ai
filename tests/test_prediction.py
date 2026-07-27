"""
Unit test suite for Predictor inference pipeline.
"""

import pytest
import pandas as pd
from src.predict import Predictor
from src.config import BEST_MODEL_PATH


def test_predictor_single_sample():
    """Verify single prediction dictionary generation."""
    if not BEST_MODEL_PATH.exists():
        pytest.skip("Model artifact trained_model.pkl not yet trained.")

    predictor = Predictor()
    input_dict = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 2000,
        "sqft_lot": 5000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1500,
        "sqft_basement": 500,
        "yr_built": 1980,
        "yr_renovated": 0,
        "zipcode": 98052,
        "lat": 47.560,
        "long": -122.213,
        "sqft_living15": 1800,
        "sqft_lot15": 4800
    }

    result = predictor.predict_single(input_dict)
    assert "predicted_price" in result
    assert result["predicted_price"] > 0
    assert "category" in result
    assert "explanation" in result
