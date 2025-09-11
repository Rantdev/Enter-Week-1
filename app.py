"""
app.py - Streamlit app for predicting suitability and yield.

Usage:
    streamlit run app.py

Place the trained models in ./models:
    models/suitability_pipeline.joblib
    models/yield_pipeline.joblib
"""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import json

MODELS_DIR = Path("models")
SUIT_MODEL = MODELS_DIR / "suitability_pipeline.joblib"
YIELD_MODEL = MODELS_DIR / "yield_pipeline.joblib"
META_PATH = MODELS_DIR / "metadata.json"

st.set_page_config(page_title="Agriculture Suitability & Yield Predictor", layout="wide")

# Column mapping from raw CSV to model-friendly names
COLUMN_MAP = {
    "Farm_Area(acres)": "Farm_Area_acres",
    "Fertilizer_Used(tons)": "Fertilizer_Used_tons",
    "Pesticide_Used(kg)": "Pesticide_Used_kg",
    "Water_Usage(cubic meters)": "Water_Usage_cubic_meters",
}

@st.cache_data
def load_artifacts():
    if not SUIT_MODEL.exists() or not YIELD_MODEL.exists() or not META_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Run preprocess_train.py to create 'models' folder.")
    clf = joblib.load(SUIT_MODEL)
    reg = joblib.load(YIELD_MODEL)
    meta = json.loads(META_PATH.read_text())
    return clf, reg, meta

def predict(df):
    clf, reg, meta = load_artifacts()

    # 🔹 Rename columns if they come from the raw dataset
    df = df.rename(columns=COLUMN_MAP)

    features = meta["numeric_features"] + meta["categorical_features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None

    X = df[features]
    suit_pred = clf.predict(X)
    yield_pred = reg.predict(X)
    df_out = df.copy()
    df_out["Predicted_Suitability"] = ["Suitable" if p == 1 else "Not Suitable" for p in suit_pred]
    df_out["Predicted_Yield_tons"] = yield_pred
    return df_out

def main():
    st.title("🌾 Agriculture Suitability & Yield Predictor")
    st.markdown("Upload a CSV file with the same columns as the training data. The app will output predicted suitability and predicted yield.")

    try:
        clf, reg, meta = load_artifacts()
        st.sidebar.write("Detected features:")
        st.sidebar.write("Numeric:", meta["numeric_features"])
        st.sidebar.write("Categorical:", meta["categorical_features"])
    except Exception as e:
        st.sidebar.error(str(e))
        return

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Preview")
        st.dataframe(df.head(10))
        if st.button("Run Predictions"):
            df_out = predict(df)
            if df_out is not None:
                st.subheader("Predictions")
                st.dataframe(df_out.head(20))
                csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", csv, file_name="predictions.csv")

    st.markdown("---")
    st.markdown("**Quick demo**: Generate a small synthetic CSV using `generate_data.py` and train using `preprocess_train.py` if you don't have your own data.")

if __name__ == "__main__":
    main()
