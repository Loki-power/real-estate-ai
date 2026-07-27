"""
Page 8: Architectural & Project Documentation
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.components.ui_components import load_custom_css

st.set_page_config(page_title="About Project | RealEstate.AI", page_icon="ℹ️", layout="wide")
load_custom_css()

st.title("ℹ️ About House Price Prediction Platform")
st.caption("Production-grade software architecture, machine learning methodology, and Vercel cloud deployment")

st.markdown(
    """
    ### 🏛️ System Architecture
    
    ```text
    House-Price-Prediction/
    ├── api/                 # FastAPI serverless endpoints for Vercel
    ├── app/                 # Interactive Streamlit multi-page dashboard
    ├── static/              # Vercel single-page glassmorphism web frontend
    ├── data/                # Raw & processed King County dataset
    ├── models/              # Joblib serialized best pipeline
    ├── reports/             # PDF evaluation reports, JSON metrics, CSV exports
    ├── src/                 # Modular Python ML engine
    ├── tests/               # Pytest unit testing suite
    └── vercel.json          # Vercel cloud routing configuration
    ```

    ### 🛠️ Key Technology Stack
    - **Language**: Python 3.10+
    - **ML Frameworks**: Scikit-Learn, XGBoost, LightGBM, CatBoost
    - **Explainability**: SHAP (SHapley Additive exPlanations)
    - **Visualizations**: Plotly, Seaborn
    - **Reports**: ReportLab PDF Generator
    - **Web Applications**: Streamlit & FastAPI
    - **Cloud Platform**: Vercel Serverless Architecture
    """
)
