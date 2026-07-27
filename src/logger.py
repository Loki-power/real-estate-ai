"""
Logging configuration for the House Price Prediction application (Vercel Serverless Ready).
"""

import logging
import sys
from pathlib import Path
from src.config import REPORTS_DIR

LOG_FILE_PATH = REPORTS_DIR / "system.log"


def get_logger(name: str = "HousePriceApp") -> logging.Logger:
    """
    Get a configured logger instance that writes formatted log messages to stdout
    and optionally to system.log if filesystem is writable.
    
    Args:
        name (str): Name of the logger domain/module.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Stream Handler (stdout - works everywhere including Vercel Serverless)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler (safely handled for read-only serverless filesystems)
        try:
            file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (OSError, PermissionError):
            pass

    return logger

# Module-default logger instance
logger = get_logger("HousePriceApp")
