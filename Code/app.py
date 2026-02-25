# app.py  (Final Clean Version)

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
from dotenv import load_dotenv
import os
import plotly.express as px

# ---------------------------------------------------
# ENV CONFIG
# ---------------------------------------------------

login_user = os.path.expanduser("~")
load_dotenv("dev.env")

clean_csv_output_path = os.getenv("OUTPUT_DATASET_PATH")
model_dir = os.getenv("MODELS_DIRECTORY")

dataset_path = os.path.join(login_user, clean_csv_output_path)
model_path = os.path.join(login_user, model_dir, "crop_production_xgb.pkl")

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Agricultural Production Forecasting System",
    layout="wide"
)

st.title("🌾 Agricultural Production Forecasting & Risk Monitoring System")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(dataset_path)
    return df

@st.cache_resource
def load_model():
    return pickle.load(open(model_path, "rb"))

df = load_data()
model = load_model()

# ---------------------------------------------------
# SIDEBAR MENU
# ---------------------------------------------------

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Overview",
            "Regional Analysis",
            "Crop Analysis",
            "Production Forecast",
            "Risk Monitoring"
        ],
        icons=["house", "globe", "flower1", "cpu", "exclamation-triangle"],
        default_index=0
    )

# =====================================================
# OVERVIEW
# =====================================================

if selected == "Overview":

    col1, col2 = st.columns(2)
    col1.metric("Total Records", len(df))
    col2.metric("Total Features", df.shape[1])

# =====================================================
# REGIONAL ANALYSIS
# =====================================================

elif selected == "Regional Analysis":

    st.subheader("🌍 Top Producing Regions")

    area_prod = (
        df.groupby("area")["production_tons"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )

    st.bar_chart(area_prod)

    yearly = df.groupby("year")["production_tons"].sum()
    st.line_chart(yearly)

# =====================================================
# CROP ANALYSIS
# =====================================================

elif selected == "Crop Analysis":

    st.subheader("🌾 Top Crops")

    crop_prod = (
        df.groupby("item")["production_tons"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )

    st.bar_chart(crop_prod)

# =====================================================
# PRODUCTION FORECAST
# =====================================================

elif selected == "Production Forecast":

    area = st.selectbox("Region", sorted(df["area"].unique()))
    crop = st.selectbox("Crop", sorted(df["item"].unique()))
    year = st.number_input("Year", min_value=1990, max_value=2035, value=2025)
    area_harvested = st.number_input("Area Harvested (ha)", min_value=0.0)
    yield_value = st.number_input("Yield (kg per ha)", min_value=0.0)

    if st.button("Predict"):

        input_df = pd.DataFrame({
            "area": [area],
            "item": [crop],
            "year": [year],
            "area_harvested_ha": [area_harvested],
            "yield_kg_per_ha": [yield_value]
        })

        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        prediction = model.predict(input_encoded)[0]

        st.success(f"Predicted Production: {prediction:,.2f} tons")

# =====================================================
# RISK MONITORING
# =====================================================

elif selected == "Risk Monitoring":

    area = st.selectbox("Region", sorted(df["area"].unique()))
    crop = st.selectbox("Crop", sorted(df["item"].unique()))

    crop_data = df[(df["area"] == area) & (df["item"] == crop)]

    if not crop_data.empty:

        avg_production = crop_data["production_tons"].mean()
        latest_year = crop_data["year"].max()

        latest_production = crop_data[
            crop_data["year"] == latest_year
        ]["production_tons"].values[0]

        st.metric("Historical Average", round(avg_production, 2))
        st.metric("Latest Production", round(latest_production, 2))

        if latest_production < avg_production:
            st.error("⚠ Production below historical average.")
        else:
            st.success("Production Stable")

        fig = px.line(
            crop_data,
            x="year",
            y="production_tons",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)