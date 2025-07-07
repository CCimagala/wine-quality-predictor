
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("wine_model.pkl")

st.title("🍷 Wine Quality Classifier")
st.write("Determine if a red wine is of good quality (score ≥ 7).")

# --- Option 1: Manual Input ---
st.header("🔬 Predict a Single Sample")
st.write("Adjust the sliders to set wine sample values.")

features = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]

input_vals = []
defaults = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]

cols = st.columns(2)
for i, feat in enumerate(features):
    val = cols[i % 2].number_input(feat.capitalize(), value=defaults[i])
    input_vals.append(val)

if st.button("Predict Single Sample"):
    data = np.array([input_vals])
    prediction = model.predict(data)[0]
    confidence = model.predict_proba(data)[0][1]
    
    if prediction == 1:
        st.success(f"✅ Good Quality Wine (Confidence: {confidence:.2%})")
    else:
        st.error(f"❌ Not Good Quality (Confidence: {confidence:.2%})")

# --- Option 2: Batch CSV Upload ---
st.header("📁 Predict from CSV File")
st.write("Upload a CSV file with wine chemical data (same columns as the dataset).")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        pred = model.predict(df)
        conf = model.predict_proba(df)[:, 1]
        df['Prediction'] = np.where(pred == 1, "Good", "Not Good")
        df['Confidence'] = np.round(conf * 100, 2).astype(str) + '%'
        st.write(df)
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
