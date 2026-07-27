"""
Page 6: Interactive House Price Prediction Engine & What-If Simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.predict import Predictor
from app.components.ui_components import load_custom_css, render_price_gauge, render_kpi_card

st.set_page_config(page_title="Valuation Engine | RealEstate.AI", page_icon="🎯", layout="wide")
load_custom_css()

st.title("🎯 Interactive House Valuation Engine")
st.caption("Adjust sliders and property attributes to generate real-time AI valuation & explainability insights")

predictor = Predictor()

col_input, col_result = st.columns([1.1, 1.4])

with col_input:
    st.subheader("🏠 Property Attributes")
    
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.slider("Bathrooms", min_value=0.5, max_value=8.0, value=2.25, step=0.25)
    sqft_living = st.slider("Living Area (Sq Ft)", min_value=300, max_value=12000, value=2100, step=50)
    sqft_lot = st.slider("Lot Size (Sq Ft)", min_value=500, max_value=100000, value=7500, step=100)
    floors = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=2)

    st.markdown("---")
    waterfront = st.selectbox("Waterfront Property View", [0, 1], format_func=lambda x: "Yes 🌊" if x == 1 else "No 🏞️")
    view = st.slider("View Quality Grade (0 - 4)", min_value=0, max_value=4, value=0)
    condition = st.slider("Building Condition (1 - 5)", min_value=1, max_value=5, value=3)
    grade = st.slider("Construction Grade (1 - 13)", min_value=1, max_value=13, value=7)
    
    st.markdown("---")
    sqft_above = st.number_input("SqFt Above Ground", value=1600, step=50)
    sqft_basement = st.number_input("SqFt Basement", value=500, step=50)
    yr_built = st.slider("Year Built", min_value=1900, max_value=2024, value=1985)
    yr_renovated = st.selectbox("Year Renovated (0 if none)", [0] + list(range(1950, 2024)), index=0)
    zipcode = st.number_input("Zipcode", value=98052, step=1)
    lat = st.number_input("Latitude", value=47.560, format="%.4f")
    long = st.number_input("Longitude", value=-122.213, format="%.4f")
    sqft_living15 = st.number_input("SqFt Living 15 (Neighbors)", value=1900)
    sqft_lot15 = st.number_input("SqFt Lot 15 (Neighbors)", value=7500)

    input_dict = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "zipcode": zipcode,
        "lat": lat,
        "long": long,
        "sqft_living15": sqft_living15,
        "sqft_lot15": sqft_lot15
    }

with col_result:
    st.subheader("💡 Estimated Market Valuation")

    try:
        res = predictor.predict_single(input_dict)
        pred_price = res["predicted_price"]

        # Card Result Box
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); padding: 28px; border-radius: 20px; text-align: center; color: white; margin-bottom: 20px; box-shadow: 0 15px 35px rgba(124, 58, 237, 0.4);">
                <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">Predicted Market Price</div>
                <div style="font-size: 3.2rem; font-weight: 800; margin: 8px 0;">${pred_price:,.2f}</div>
                <span class="badge-pill" style="background-color: {res['badge_color']}; color: white;">{res['category']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            render_kpi_card("Model Confidence", f"{res['confidence_score']}%", "Estimated Prediction Interval", "purple")
        with c2:
            render_kpi_card("vs. Regional Avg ($540k)", f"{res['diff_from_avg_pct']:+,.1f}%", "King County Baseline", "blue" if res['diff_from_avg_pct'] > 0 else "emerald")

        st.plotly_chart(render_price_gauge(pred_price), use_container_width=True)

        st.subheader("🧠 AI Natural Language Explanation")
        st.info(res["explanation"])

        st.markdown("---")
        st.subheader("📥 Export Valuation Summary")
        
        df_export = pd.DataFrame([res])
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Prediction Record (CSV)",
            data=csv_data,
            file_name="house_price_prediction.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}. Ensure trained model exists by running `python main.py`.")
