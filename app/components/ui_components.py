"""
Reusable Streamlit UI components: Gradient Cards, Glassmorphism Panels,
KPI Metric Badges, Gauge Meters, and Custom CSS loaders.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any

from src.config import THEME_COLORS


def load_custom_css():
    """Injects premium blue-purple glassmorphism CSS styles into Streamlit app."""
    css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 25px 50px rgba(99, 102, 241, 0.25);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Gradient Cards */
    .gradient-card-purple {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 16px;
        padding: 24px;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.3);
    }
    
    .gradient-card-blue {
        background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%);
        border-radius: 16px;
        padding: 24px;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.3);
    }
    
    .gradient-card-emerald {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-radius: 16px;
        padding: 24px;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
    }

    /* KPI Metric Styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_kpi_card(title: str, value: str, subtext: str = "", card_type: str = "purple"):
    """Renders styled metric card."""
    card_class = f"gradient-card-{card_type}" if card_type in ["purple", "blue", "emerald"] else "glass-card"
    html = f"""
    <div class="{card_class}">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        <div style="font-size: 0.85rem; opacity: 0.8;">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_price_gauge(price: float, min_val: float = 100000, max_val: float = 2000000) -> go.Figure:
    """Renders Plotly price gauge meter chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=price,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>Valuation Meter</b>", 'font': {'size': 18, 'color': '#f8fafc'}},
        delta={'reference': 540088, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
            'bar': {'color': "#6366f1"},
            'bgcolor': "rgba(30, 41, 59, 0.8)",
            'borderwidth': 2,
            'bordercolor': "rgba(255, 255, 255, 0.2)",
            'steps': [
                {'range': [min_val, 350000], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [350000, 650000], 'color': 'rgba(59, 130, 246, 0.3)'},
                {'range': [650000, 1200000], 'color': 'rgba(139, 92, 246, 0.3)'},
                {'range': [1200000, max_val], 'color': 'rgba(239, 68, 68, 0.3)'}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#f8fafc", 'family': "Inter, sans-serif"},
        height=280,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
