"""
Page 2: Exploratory Data Analysis (EDA) Suite
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_loader import DataLoader
from src.eda import EDAEngine
from src.visualization import Visualizer
from app.components.ui_components import load_custom_css

st.set_page_config(page_title="EDA | RealEstate.AI", page_icon="🔍", layout="wide")
load_custom_css()

st.title("🔍 Exploratory Data Analysis (EDA)")
st.caption("Interactive analysis of features, distributions, correlations, and geography")

loader = DataLoader()
df = loader.load_data()
eda = EDAEngine(df)
viz = Visualizer(theme="dark")

tabs = st.tabs([
    "🔥 Correlation Analysis",
    "🗺️ Geographic Map",
    "📊 Price Distributions & KDE",
    "📦 Box & Violin Plots",
    "🎯 Feature vs. Price Scatters",
    "📋 Summary Statistics & Missing Values"
])

with tabs[0]:
    st.subheader("Feature Correlation Matrix")
    corr_df = eda.get_correlation_matrix()
    st.plotly_chart(viz.plot_correlation_heatmap(corr_df), use_container_width=True)

with tabs[1]:
    st.subheader("Geographic Price Distribution (King County)")
    st.plotly_chart(viz.plot_geographic_map(df), use_container_width=True)

with tabs[2]:
    st.subheader("Price Distribution & Logarithmic KDE")
    st.plotly_chart(viz.plot_price_distribution(df), use_container_width=True)

with tabs[3]:
    col_a, col_b = st.columns(2)
    with col_a:
        group_col = st.selectbox("Select Grouping Feature for Boxplot", ["grade", "waterfront", "view", "bedrooms", "condition"])
        st.plotly_chart(viz.plot_boxplot(df, group_col), use_container_width=True)
    with col_b:
        violin_col = st.selectbox("Select Grouping Feature for Violin Plot", ["condition", "grade", "waterfront", "floors"])
        st.plotly_chart(viz.plot_violin(df, violin_col), use_container_width=True)

with tabs[4]:
    scatter_feat = st.selectbox("Select Feature for Scatter Plot", ["sqft_living", "sqft_above", "sqft_lot", "yr_built", "zipcode"])
    st.plotly_chart(viz.plot_feature_vs_price_scatter(df, scatter_feat), use_container_width=True)

with tabs[5]:
    st.subheader("Dataset Summary Statistics")
    summary_df = eda.get_summary_table()
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Missing Value Matrix")
    missing = df.isnull().sum().to_frame("Missing Count")
    missing["Percentage"] = (missing["Missing Count"] / len(df)) * 100
    st.dataframe(missing.T, use_container_width=True)
