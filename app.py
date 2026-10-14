from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Used Car Price Prediction")
st.title("Used Car Price Prediction")
st.write("Estimate a vehicle's listing price in US dollars.")


@st.cache_resource
def load_pipeline():
    return joblib.load(ROOT / "used_car_pipeline.joblib")


pipeline = load_pipeline()

with open(ROOT / "app_details.json") as file:
    details = json.load(file)

categories = details["categories"]

with st.form("vehicle_form"):

    brand = st.selectbox("Brand", categories["brand"])
    model = st.selectbox("Model", categories["model"])

    model_year = st.number_input(
        "Model year",
        min_value=1974,
        max_value=2024,
        value=2020
    )

    mileage = st.number_input(
        "Mileage in miles",
        min_value=0,
        value=50000,
        step=1000
    )

    fuel_type = st.selectbox(
        "Fuel type", categories["fuel_type"]
    )

    accident = st.selectbox(
        "Accident history", categories["accident"]
    )

    clean_title = st.selectbox(
        "Clean title", categories["clean_title"]
    )

    transmission_type = st.selectbox(
        "Transmission type",
        categories["transmission_type"]
    )

    st.write("Leave unavailable engine or gear details as unknown.")

    horsepower_known = st.checkbox("Horsepower is known")
    horsepower = st.number_input(
        "Horsepower", min_value=1.0, value=200.0
    )

    litres_known = st.checkbox("Engine size is known")
    engine_litres = st.number_input(
        "Engine size in litres", min_value=0.1, value=2.0
    )

    cylinders_known = st.checkbox("Cylinder count is known")
    cylinders = st.number_input(
        "Cylinders", min_value=1, value=4
    )

    gears_known = st.checkbox("Gear count is known")
    gears = st.number_input(
        "Gears", min_value=1, value=6
    )

    submitted = st.form_submit_button("Predict price")

if submitted:

    vehicle = pd.DataFrame([{
        "model_year": model_year,
        "milage": mileage,
        "horsepower": horsepower if horsepower_known else np.nan,
        "engine_litres": engine_litres if litres_known else np.nan,
        "cylinders": cylinders if cylinders_known else np.nan,
        "gears": gears if gears_known else np.nan,
        "brand": brand,
        "model": model,
        "fuel_type": fuel_type,
        "accident": accident,
        "clean_title": clean_title,
        "transmission_type": transmission_type
    }])

    prediction = float(pipeline.predict(vehicle)[0])

    if np.isfinite(prediction) and prediction > 0:
        st.metric("Estimated listing price", f"${prediction:,.2f}")
    else:
        st.warning("Invalid prediction. Check the vehicle details.")

st.subheader("Held-out test performance")

metrics = details["metrics"]

st.write(f"MAE: ${metrics['MAE']:,.2f}")
st.write(f"RMSE: ${metrics['RMSE']:,.2f}")
st.write(f"R²: {metrics['R2']:.3f}")

st.caption(
    "This estimates historical listing prices, not guaranteed selling "
    "prices. Rare vehicles and extreme prices may have larger errors. "
    "MAE is an average error, not a prediction interval."
)
