"""
Page 7: SHAP Explainable AI (XAI) Suite
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import FEATURE_IMPORTANCE_PATH
from src.visualization import Visualizer
from app.components.ui_components import load_custom_css

st.set_page_config(page_title="Explainability | RealEstate.AI", page_icon="🧠", layout="wide")
load_custom_css()

st.title("🧠 Explainable AI (SHAP) Interpretation")
st.caption("Transparent ML breakdown revealing global feature importances and local decision boundaries")

if FEATURE_IMPORTANCE_PATH.exists():
    df_imp = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    viz = Visualizer(theme="dark")

    st.subheader("Global Feature Importance Ranking (SHAP)")
    st.plotly_chart(viz.plot_feature_importance(df_imp), use_container_width=True)

    st.markdown("---")
    st.subheader("SHAP Explanation Plot Glossary")

    st.markdown(
        """
        - **Global Feature Importance**: Quantifies average impact across all King County sales.
        - **Waterfall Plot**: Shows how each attribute pushes the base expected price up or down.
        - **Force Plot**: Dynamic vector visualization balancing positive (red) and negative (blue) price forces.
        - **Dependence Plot**: Illustrates non-linear feature interactions (e.g. sqft_living vs grade).
        """
    )
else:
    st.info("Feature importance artifact not generated yet. Run `python main.py` to create SHAP analysis files.")
