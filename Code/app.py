import streamlit as st
import pandas as pd
import pickle
from dotenv import load_dotenv
import os

login_user = os.path.expanduser("~")
load_dotenv("dev.env")

clean_csv_output_path = os.getenv("OUTPUT_DATASET_PATH")
model_dir = os.getenv("MODELS_DIRECTORY")

dataset_path = os.path.join(login_user, clean_csv_output_path)
model_path = os.path.join(login_user, model_dir, "crop_production_xgb.pkl")

st.set_page_config(
    page_title="Crop Production Forecasting",
    layout="centered"
)

st.title("🌾 Crop Production Forecasting")

@st.cache_data
def load_data():
    return pd.read_csv(dataset_path)

@st.cache_resource
def load_model():
    return pickle.load(open(model_path, "rb"))

df = load_data()
model = load_model()

st.subheader("Enter Crop Details")

area = st.selectbox("Select Region", sorted(df["area"].unique()))
crop = st.selectbox("Select Crop", sorted(df["item"].unique()))
year = st.number_input("Year", min_value=1990, max_value=2035, value=2025)
area_harvested = st.number_input("Area Harvested (ha)", min_value=0.0)
yield_value = st.number_input("Yield (kg per ha)", min_value=0.0)


if st.button("Predict Production"):
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

    st.success(f"🌾 Predicted Production: {prediction:,.2f} tons")

   