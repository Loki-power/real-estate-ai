"""
Explainable AI (XAI) engine using SHAP with graceful fallback to tree/linear feature importances.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

from src.logger import logger
from src.config import FEATURE_IMPORTANCE_PATH


class Explainer:
    """Computes feature importances and natural language narrative descriptions."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.model = pipeline.named_steps["model"]
        self.explainer: Optional[Any] = None

    def _get_transformed_features(self, X_sample: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transforms raw input features using the preprocessor pipeline and returns feature names."""
        X_trans = self.preprocessor.transform(X_sample)
        
        feature_names = []
        if hasattr(self.preprocessor, "get_feature_names_out"):
            try:
                feature_names = list(self.preprocessor.get_feature_names_out())
            except Exception:
                feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]
        else:
            feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]

        clean_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]
        return X_trans, clean_names

    def fit_shap(self, X_train_sample: pd.DataFrame) -> None:
        """Fits SHAP explainer if available."""
        if not HAS_SHAP:
            logger.info("SHAP package not present. Falling back to native tree/model feature importances.")
            return

        try:
            X_trans, _ = self._get_transformed_features(X_train_sample)
            if hasattr(self.model, "feature_importances_") or "Tree" in type(self.model).__name__ or "Forest" in type(self.model).__name__ or "Boost" in type(self.model).__name__:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                self.explainer = shap.Explainer(self.model, X_trans)

            logger.info("Fitted SHAP explainer successfully.")
        except Exception as e:
            logger.warning(f"SHAP fit fallback: {e}")

    def export_feature_importance(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """Calculates global feature importances and exports feature_importance.csv."""
        X_trans, clean_names = self._get_transformed_features(X_sample)

        importances = np.zeros(len(clean_names))

        if HAS_SHAP and self.explainer is not None:
            try:
                shap_vals = self.explainer(X_trans)
                shap_matrix = shap_vals.values if hasattr(shap_vals, "values") else np.array(shap_vals)
                importances = np.abs(shap_matrix).mean(axis=0)
            except Exception:
                pass

        if np.all(importances == 0):
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                importances = np.abs(self.model.coef_)
            else:
                importances = np.ones(len(clean_names)) / len(clean_names)

        df_imp = pd.DataFrame({
            "feature": clean_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        df_imp.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        logger.info(f"Exported global feature importance to {FEATURE_IMPORTANCE_PATH}")
        return df_imp

    def generate_natural_language_explanation(self, instance_dict: Dict[str, Any], predicted_price: float) -> str:
        """
        AI Feature: Converts house attributes into a natural language narrative explanation.
        """
        reasons = []

        sqft = instance_dict.get("sqft_living", 2000)
        grade = instance_dict.get("grade", 7)
        waterfront = instance_dict.get("waterfront", 0)
        bathrooms = instance_dict.get("bathrooms", 2)
        bedrooms = instance_dict.get("bedrooms", 3)
        yr_built = instance_dict.get("yr_built", 1980)

        if sqft > 2800:
            reasons.append(f"large living area ({sqft:,} sq ft)")
        elif sqft < 1400:
            reasons.append(f"compact interior area ({sqft:,} sq ft)")

        if grade >= 9:
            reasons.append(f"exceptional construction grade ({grade}/13)")
        elif grade <= 6:
            reasons.append(f"modest construction grade ({grade}/13)")

        if waterfront == 1:
            reasons.append("premium waterfront view")

        if bathrooms >= 3.5:
            reasons.append(f"luxurious bathroom count ({bathrooms} baths)")

        if yr_built >= 2005:
            reasons.append(f"modern construction (built in {yr_built})")

        price_level = "premium" if predicted_price > 650000 else "standard" if predicted_price > 350000 else "affordable"

        if reasons:
            drivers_str = ", ".join(reasons[:-1]) + f", and {reasons[-1]}" if len(reasons) > 1 else reasons[0]
            narrative = f"The estimated house valuation of ${predicted_price:,.2f} reflects a {price_level} property category. The price is primarily driven by its {drivers_str}."
        else:
            narrative = f"The estimated house valuation of ${predicted_price:,.2f} is well-aligned with regional market standards for properties with {bedrooms} beds and {bathrooms} baths."

        return narrative
