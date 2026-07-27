"""
Page 4: Model Training & Hyperparameter Tuning Trigger
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.components.ui_components import load_custom_css
from src.config import MODEL_PARAM_GRIDS

st.set_page_config(page_title="Model Training | RealEstate.AI", page_icon="🤖", layout="wide")
load_custom_css()

st.title("🤖 Multi-Model Training & Hyperparameter Tuning")
st.caption("Configure algorithms, search parameters, and initiate automated pipeline execution")

st.markdown("### Supported 11 Regressors")

cols = st.columns(3)
models_list = list(MODEL_PARAM_GRIDS.keys())

for idx, model_name in enumerate(models_list):
    with cols[idx % 3]:
        st.info(f"**{model_name}**\n\nHyperparams tuned via RandomizedSearchCV")

st.markdown("---")

st.subheader("Interactive Pipeline Trigger")
use_tuning = st.checkbox("Enable RandomizedSearchCV Hyperparameter Tuning", value=True)
cv_folds = st.slider("Cross Validation Folds (CV)", min_value=3, max_value=10, value=5)

if st.button("🚀 Run Full ML Benchmark Pipeline", type="primary"):
    with st.spinner("Training 11 Machine Learning models... Please wait."):
        import subprocess
        result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.balloons()
            st.success("✅ Machine Learning pipeline completed successfully! Model saved to `models/trained_model.pkl`.")
            st.text_area("Pipeline Console Logs", result.stdout, height=250)
        else:
            st.error("Pipeline run encountered an issue:")
            st.text_area("Error Log", result.stderr, height=200)
