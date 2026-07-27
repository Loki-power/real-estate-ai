"""
Page 3: Feature Engineering Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.visualization import Visualizer
from app.components.ui_components import load_custom_css

st.set_page_config(page_title="Feature Engineering | RealEstate.AI", page_icon="⚙️", layout="wide")
load_custom_css()

st.title("⚙️ Domain Feature Engineering Explorer")
st.caption("Inspect newly created features and their correlation impact on house prices")

loader = DataLoader()
raw_df = loader.load_data()
fe = FeatureEngineer()
engineered_df = fe.transform(raw_df)

st.success("Successfully generated 7 engineered domain features!")

# Display Engineered Feature Samples
eng_cols = ["house_age", "is_renovated", "living_lot_ratio", "total_sqft", "bath_bed_ratio", "living_compared_15", "lot_compared_15"]
st.subheader("Newly Created Feature Samples")
st.dataframe(engineered_df[["price"] + eng_cols].head(10), use_container_width=True)

st.markdown("---")
st.subheader("Feature Correlation with Target Price ($)")

corr_series = engineered_df[eng_cols + ["price"]].corr()["price"].drop("price").sort_values(ascending=False)
fig = px.bar(
    x=corr_series.values,
    y=corr_series.index,
    orientation="h",
    labels={"x": "Correlation Coefficient with Price", "y": "Engineered Feature"},
    color=corr_series.values,
    color_continuous_scale="Purples"
)
fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)
