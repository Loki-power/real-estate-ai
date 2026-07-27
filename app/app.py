"""
Main Streamlit application entry point for the House Price Prediction Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import DataLoader
from src.config import THEME_COLORS
from app.components.ui_components import load_custom_css, render_kpi_card

# Configure Streamlit page layout
st.set_page_config(
    page_title="RealEstate.AI - House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css()

# Sidebar Navigation Header
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h2 style="color: #6366f1; margin-bottom: 0;">🏠 RealEstate.AI</h2>
        <p style="font-size: 0.8rem; color: #94a3b8;">King County Valuation System</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Theme & Global Filters
st.sidebar.subheader("⚙️ Global Settings")
theme_mode = st.sidebar.radio("Theme Mode", ["Dark Mode 🌙", "Light Mode ☀️"], index=0)

# Main Hero Header
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #2563eb 100%); padding: 32px; border-radius: 20px; color: white; margin-bottom: 24px; box-shadow: 0 20px 40px rgba(79, 70, 229, 0.3);">
        <h1 style="font-size: 2.6rem; font-weight: 800; margin-bottom: 8px; color: white;">Production-Ready House Price Analytics</h1>
        <p style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 0;">Automated ML Engine, 11 Benchmark Regressors, SHAP Explainable AI & Interactive Valuation Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load metadata
try:
    loader = DataLoader()
    raw_df = loader.load_data()
    meta = loader.get_metadata()

    # KPI Banner Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Total Dataset Rows", f"{meta['total_rows']:,}", "King County Sales", "purple")
    with c2:
        render_kpi_card("Average House Price", f"${meta['target_mean']:,.0f}", "Mean Valuation", "blue")
    with c3:
        render_kpi_card("Median House Price", f"${meta['target_median']:,.0f}", "50th Percentile", "emerald")
    with c4:
        render_kpi_card("Total Features", f"{meta['total_columns']}", "Numerical & Categorical", "purple")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("📌 Project Architectural Workflow")
        st.markdown(
            """
            This platform uses an **end-to-end modular Python architecture** covering:
            - **Data Pipeline**: Outlier removal (IQR), median imputation, and Scikit-Learn `ColumnTransformer`.
            - **Domain Feature Engineering**: Age calculation, total square footage, living/lot ratio, renovation flag.
            - **Multi-Model Suite**: 11 algorithms trained (Linear, Ridge, Lasso, Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, CatBoost, LightGBM).
            - **Explainable AI (SHAP)**: Waterfall & Force plots detailing positive/negative price drivers.
            - **Vercel & Streamlit Dual Deployment**: REST API powered by FastAPI alongside interactive Streamlit dashboards.
            """
        )

    with col_right:
        st.subheader("🚀 Quick Actions")
        st.markdown("Use the **Sidebar Menu** to explore pages:")
        st.info("📊 **EDA**: Explore correlations & maps\n\n🤖 **Prediction**: Estimate house valuation\n\n💡 **Explainability**: SHAP AI breakdown\n\n🏆 **Comparison**: Model leaderboard")

except Exception as e:
    st.warning(f"Note: Dataset loaded with basic view ({e}). Run `python main.py` to generate complete models.")
