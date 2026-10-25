
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

st.title("🏠 House Price Prediction")
st.write("Mee house details enter chesi estimated price telusukondi.")

model_path = Path("artifacts/house_price_model.joblib")
columns_path = Path("artifacts/model_columns.joblib")
data_path = Path("dataset/Housing.csv")

if not model_path.exists() or not columns_path.exists():
    st.error("Model files dorakaledu. Mundhu train.py run cheyyi.")
    st.stop()

if not data_path.exists():
    st.error("dataset/Housing.csv file dorakaledu.")
    st.stop()

model = joblib.load(model_path)
model_columns = joblib.load(columns_path)
df = pd.read_csv(data_path)

# Original input columns: price column ni remove cheyyali
original_features = [
    col for col in df.columns if col != "price"
]

st.subheader("Enter House Details")

inputs = {}

for col in original_features:
    if pd.api.types.is_numeric_dtype(df[col]):
        median_value = df[col].median()
        inputs[col] = st.number_input(
            col.replace("_", " ").title(),
            min_value=0.0,
            value=float(median_value) if pd.notna(median_value) else 0.0
        )
    else:
        options = df[col].dropna().astype(str).unique().tolist()

        if not options:
            options = ["Unknown"]

        inputs[col] = st.selectbox(
            col.replace("_", " ").title(),
            options
        )

if st.button("Predict House Price"):
    # User inputs ni model ki tagina format loki convert cheyyali
    input_df = pd.DataFrame([inputs])
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Training time lo unna columns anni maintain cheyyali
    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    prediction = model.predict(input_df)[0]

    st.success(f"Estimated House Price: {prediction:,.2f}")
    st.caption(
        "Idi machine learning model estimate maatrame; "
        "actual market price veruga undavachu."
    )