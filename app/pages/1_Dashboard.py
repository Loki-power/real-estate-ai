"""
Page 1: Overview Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_loader import DataLoader
from src.visualization import Visualizer
from app.components.ui_components import load_custom_css, render_kpi_card

st.set_page_config(page_title="Dashboard | RealEstate.AI", page_icon="📈", layout="wide")
load_custom_css()

st.title("📈 Executive Overview Dashboard")
st.caption("Real-time summary of King County house sales dataset and model performance")

loader = DataLoader()
df = loader.load_data()
viz = Visualizer(theme="dark")

# Key Indicators
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card("Total Sales Records", f"{len(df):,}", "Transactions", "purple")
with col2:
    render_kpi_card("Mean Price", f"${df['price'].mean():,.0f}", "USD ($)", "blue")
with col3:
    render_kpi_card("Max Sale Price", f"${df['price'].max():,.0f}", "Waterfront Estate", "emerald")
with col4:
    render_kpi_card("Avg Living SqFt", f"{df['sqft_living'].mean():,.0f} sq ft", "Living Space", "purple")

st.markdown("---")

c_left, c_right = st.columns(2)
with c_left:
    st.plotly_chart(viz.plot_price_distribution(df), use_container_width=True)

with c_right:
    st.plotly_chart(viz.plot_geographic_map(df), use_container_width=True)
