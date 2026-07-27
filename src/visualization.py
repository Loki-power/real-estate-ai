"""
Visualization suite powered by Plotly for interactive, styled charts matching dark/light mode themes.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional

from src.config import THEME_COLORS, TARGET_COL


class Visualizer:
    """Plotly chart generator with blue-purple glassmorphism visual styling."""

    def __init__(self, theme: str = "dark"):
        self.theme = theme
        self.bg_color = THEME_COLORS["background_dark"] if theme == "dark" else THEME_COLORS["background_light"]
        self.card_color = THEME_COLORS["card_dark"] if theme == "dark" else THEME_COLORS["card_light"]
        self.text_color = THEME_COLORS["text_dark"] if theme == "dark" else THEME_COLORS["text_light"]
        self.template = "plotly_dark" if theme == "dark" else "plotly_white"

    def _apply_layout(self, fig: go.Figure, title: str, x_title: str = "", y_title: str = "") -> go.Figure:
        """Applies consistent styling, fonts, and dark/light paper backgrounds."""
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=18, color=self.text_color, family="Inter, sans-serif")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=self.text_color, family="Inter, sans-serif"),
            xaxis=dict(title=x_title, gridcolor="rgba(255,255,255,0.1)" if self.theme == "dark" else "rgba(0,0,0,0.05)"),
            yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,0.1)" if self.theme == "dark" else "rgba(0,0,0,0.05)"),
            template=self.template,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    def plot_correlation_heatmap(self, corr_df: pd.DataFrame) -> go.Figure:
        """Generates correlation heatmap."""
        fig = px.imshow(
            corr_df,
            text_auto=".2f",
            color_continuous_scale="Viridis" if self.theme == "dark" else "RdBu_r",
            aspect="auto"
        )
        return self._apply_layout(fig, "Feature Correlation Heatmap")

    def plot_price_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Generates house price distribution histogram with logarithmic toggle."""
        fig = px.histogram(
            df,
            x=TARGET_COL,
            nbins=50,
            marginal="box",
            color_discrete_sequence=[THEME_COLORS["primary"]],
            opacity=0.85
        )
        return self._apply_layout(fig, "King County House Price Distribution", "Price ($)", "Count")

    def plot_geographic_map(self, df: pd.DataFrame) -> go.Figure:
        """Generates interactive geographic scatter map of house locations colored by price."""
        sample_df = df.sample(min(5000, len(df)), random_state=42)
        fig = px.scatter_mapbox(
            sample_df,
            lat="lat",
            lon="long",
            color=TARGET_COL,
            size="sqft_living",
            color_continuous_scale="Plasma",
            size_max=15,
            zoom=9,
            hover_data=["bedrooms", "bathrooms", "grade", "zipcode"],
            mapbox_style="carto-darkmatter" if self.theme == "dark" else "carto-positron"
        )
        return self._apply_layout(fig, "Geographic House Price Distribution (King County)")

    def plot_feature_vs_price_scatter(self, df: pd.DataFrame, feature_name: str) -> go.Figure:
        """Interactive scatter plot of feature vs price."""
        sample_df = df.sample(min(3000, len(df)), random_state=42)
        fig = px.scatter(
            sample_df,
            x=feature_name,
            y=TARGET_COL,
            color="grade",
            trendline="ols",
            color_continuous_scale="Viridis"
        )
        return self._apply_layout(fig, f"{feature_name.replace('_', ' ').title()} vs. House Price", feature_name, "Price ($)")

    def plot_boxplot(self, df: pd.DataFrame, category_col: str) -> go.Figure:
        """Generates price boxplot grouped by categorical feature (e.g. grade, waterfront)."""
        fig = px.box(
            df,
            x=category_col,
            y=TARGET_COL,
            color=category_col,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        return self._apply_layout(fig, f"Price Distribution by {category_col.title()}", category_col, "Price ($)")

    def plot_violin(self, df: pd.DataFrame, category_col: str) -> go.Figure:
        """Generates price violin plot grouped by categorical feature."""
        fig = px.violin(
            df,
            x=category_col,
            y=TARGET_COL,
            color=category_col,
            box=True,
            points="all"
        )
        return self._apply_layout(fig, f"Price Violin Density by {category_col.title()}", category_col, "Price ($)")

    def plot_feature_importance(self, importance_df: pd.DataFrame) -> go.Figure:
        """Generates horizontal feature importance bar chart."""
        df_sorted = importance_df.sort_values(by="importance", ascending=True).tail(15)
        fig = px.bar(
            df_sorted,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Purples"
        )
        return self._apply_layout(fig, "Top 15 Predictive Features (SHAP / Feature Importance)", "Importance Score", "Feature")

    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """Scatter plot of Actual vs. Predicted house prices."""
        df_eval = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
        sample = df_eval.sample(min(2000, len(df_eval)), random_state=42)
        fig = px.scatter(
            sample,
            x="Actual",
            y="Predicted",
            opacity=0.6,
            color_discrete_sequence=[THEME_COLORS["secondary"]]
        )
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", name="Perfect Fit (y=x)", line=dict(color=THEME_COLORS["warning"], dash="dash"))
        )
        return self._apply_layout(fig, "Actual vs. Predicted House Prices", "Actual Price ($)", "Predicted Price ($)")

    def plot_model_leaderboard(self, metrics_df: pd.DataFrame) -> go.Figure:
        """Bar chart comparing R2 and RMSE across trained ML models."""
        fig = px.bar(
            metrics_df,
            x="Model",
            y="R2_Score",
            color="RMSE",
            text_auto=".3f",
            color_continuous_scale="Viridis_r"
        )
        return self._apply_layout(fig, "Model Leaderboard Comparison (R² Score & RMSE)", "Model", "R² Score")
