"""
Page 5: Model Comparison & Performance Leaderboard
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import METRICS_JSON_PATH
from src.utils import load_json
from src.visualization import Visualizer
from app.components.ui_components import load_custom_css, render_kpi_card

st.set_page_config(page_title="Model Leaderboard | RealEstate.AI", page_icon="🏆", layout="wide")
load_custom_css()

st.title("🏆 Regressor Leaderboard & Model Comparison")
st.caption("Benchmark metrics comparing MAE, MSE, RMSE, R², Adjusted R², and MAPE across all 11 models")

metrics_dict = load_json(METRICS_JSON_PATH)

if not metrics_dict:
    st.info("No trained metrics found yet. Click 'Run Full ML Benchmark Pipeline' in Model Training or run `python main.py`.")
else:
    df_metrics = pd.DataFrame(list(metrics_dict.values())).sort_values(by="R2_Score", ascending=False).reset_index(drop=True)
    
    # Champion Card
    top_row = df_metrics.iloc[0]
    st.subheader(f"🥇 Champion Regressor: {top_row['Model']}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("R² Accuracy Score", f"{top_row['R2_Score']:.4f}", "Variance Explained", "purple")
    with col2:
        render_kpi_card("RMSE", f"${top_row['RMSE']:,.0f}", "Root Mean Sq Error", "blue")
    with col3:
        render_kpi_card("MAE", f"${top_row['MAE']:,.0f}", "Mean Absolute Error", "emerald")
    with col4:
        render_kpi_card("MAPE", f"{top_row['MAPE']:.2f}%", "Percentage Error", "purple")

    st.markdown("---")

    st.subheader("Leaderboard Table")
    st.dataframe(df_metrics, use_container_width=True)

    viz = Visualizer(theme="dark")
    st.plotly_chart(viz.plot_model_leaderboard(df_metrics), use_container_width=True)
