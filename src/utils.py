"""
Utility functions for file I/O, serialization, data formatting, and prediction history storage.
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.logger import logger
from src.config import PREDICTION_HISTORY_CSV_PATH


def save_joblib(obj: Any, file_path: Path) -> None:
    """Save an object using joblib serialization."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Successfully saved Joblib object to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save Joblib object to {file_path}: {e}")
        raise


def load_joblib(file_path: Path) -> Any:
    """Load an object using joblib deserialization."""
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        obj = joblib.load(file_path)
        logger.info(f"Successfully loaded Joblib object from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load Joblib object from {file_path}: {e}")
        raise


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save dictionary to JSON file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Successfully saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return {}


def format_currency(value: Union[float, int]) -> str:
    """Format numeric value as USD currency string (e.g. $540,000)."""
    return f"${value:,.2f}" if isinstance(value, (int, float)) else str(value)


def format_number(value: Union[float, int], decimals: int = 2) -> str:
    """Format float or integer with thousands separator."""
    return f"{value:,.{decimals}f}" if isinstance(value, (int, float)) else str(value)


def save_prediction_record(record: Dict[str, Any]) -> None:
    """Append a single prediction record to prediction_history.csv."""
    try:
        PREDICTION_HISTORY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_new = pd.DataFrame([record])
        if PREDICTION_HISTORY_CSV_PATH.exists():
            df_new.to_csv(PREDICTION_HISTORY_CSV_PATH, mode="a", header=False, index=False)
        else:
            df_new.to_csv(PREDICTION_HISTORY_CSV_PATH, mode="w", header=True, index=False)
        logger.info("Appended prediction record to history.")
    except Exception as e:
        logger.error(f"Error saving prediction record: {e}")
