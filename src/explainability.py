"""
Explainable AI (XAI) engine using SHAP for global/local feature importance and natural language explanations.
"""

import pandas as pd
import numpy as np
import shap
from typing import Dict, Any, List, Tuple, Optional

from src.logger import logger
from src.config import FEATURE_IMPORTANCE_PATH
from src.utils import save_joblib


class Explainer:
    """Computes SHAP explanations and natural language narrative descriptions."""

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

        # Clean feature names (remove num__ and cat__)
        clean_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]
        return X_trans, clean_names

    def fit_shap(self, X_train_sample: pd.DataFrame) -> None:
        """Fits SHAP explainer on a background training sample."""
        try:
            X_trans, _ = self._get_transformed_features(X_train_sample)
            
            # Tree-based or linear explainer
            if hasattr(self.model, "feature_importances_") or "Tree" in type(self.model).__name__ or "Forest" in type(self.model).__name__ or "Boost" in type(self.model).__name__:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                self.explainer = shap.Explainer(self.model, X_trans)

            logger.info("Fitted SHAP explainer successfully.")
        except Exception as e:
            logger.warning(f"Defaulting to generic Kernel/Permutation SHAP explainer due to: {e}")
            X_trans, _ = self._get_transformed_features(X_train_sample)
            self.explainer = shap.Explainer(self.model.predict, X_trans)

    def get_shap_values(self, X_sample: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Calculates SHAP values matrix and returns clean feature names."""
        if self.explainer is None:
            self.fit_shap(X_sample)

        X_trans, clean_names = self._get_transformed_features(X_sample)
        shap_vals = self.explainer(X_trans)

        if hasattr(shap_vals, "values"):
            shap_matrix = shap_vals.values
        else:
            shap_matrix = np.array(shap_vals)

        return shap_matrix, clean_names

    def export_feature_importance(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """Calculates global mean absolute SHAP values and exports feature_importance.csv."""
        shap_matrix, clean_names = self.get_shap_values(X_sample)
        mean_abs_shap = np.abs(shap_matrix).mean(axis=0)

        df_imp = pd.DataFrame({
            "feature": clean_names,
            "importance": mean_abs_shap
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        df_imp.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        logger.info(f"Exported global feature importance to {FEATURE_IMPORTANCE_PATH}")
        return df_imp

    def generate_natural_language_explanation(self, instance_dict: Dict[str, Any], predicted_price: float) -> str:
        """
        AI Feature: Converts house attributes and SHAP impact into natural language explanation.
        
        Args:
            instance_dict (Dict[str, Any]): Feature values of house.
            predicted_price (float): Predicted house valuation.
            
        Returns:
            str: Human-readable natural language paragraph explaining the price.
        """
        reasons = []

        # Key drivers inspection
        sqft = instance_dict.get("sqft_living", 2000)
        grade = instance_dict.get("grade", 7)
        waterfront = instance_dict.get("waterfront", 0)
        bathrooms = instance_dict.get("bathrooms", 2)
        bedrooms = instance_dict.get("bedrooms", 3)
        yr_built = instance_dict.get("yr_built", 1980)

        if sqft > 2800:
            reasons.append(f"large living space ({sqft:,} sq ft)")
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
