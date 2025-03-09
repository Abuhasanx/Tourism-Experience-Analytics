import streamlit as st
import joblib
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
import xgboost 

# Load trained model and encoders
model_path = r"D:\1final ds\predxgb.pkl"
ohe_path = r"D:\1final ds\preddump.pkl"
target_enc_path = r"D:\1final ds\predrating_target.pkl"
scaler_path = r"D:\1final ds\predscalar.pkl"

# Load saved models
best_xgb = joblib.load(model_path)
ohe = joblib.load(ohe_path)
target_enc = joblib.load(target_enc_path)
scaler = joblib.load(scaler_path)

# Streamlit app styling
st.markdown("""
    <style>
        .block-container { padding-top: 0rem !important; }
        body { background-color: #f0f2f6; }
        .main {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
        }
        .title {
            text-align: center;
            color: #1f77b4;
            font-size: 30px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: gray;
            font-size: 18px;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 8px 20px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover { background-color: #135a91; }
    </style>
""", unsafe_allow_html=True)

# Create the main layout
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<p class='title'>🏝️ TOURISM RATING PREDICTION APP</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter details below to predict the tourism rating.</p>", unsafe_allow_html=True)

# User input fields
VisitYear = st.number_input("📅 Visit Year", min_value=2000, max_value=2100, step=1)
VisitMonth = st.number_input("📆 Visit Month", min_value=1, max_value=12, step=1)
VisitModeName = st.text_input("🚗 Visit Mode Name", placeholder="e.g., Friends")
AttractionId = st.number_input("🏞️ Attraction ID", min_value=1, step=1)

Attraction = st.text_input("🎡 Attraction Name", placeholder="e.g., Sacred Monkey Forest Sanctuary")
AttractionType = st.text_input("🏛️ Attraction Type", placeholder="e.g., Beaches")
CountryId = st.number_input("🌍 Country ID", min_value=1, step=1)
RegionId = st.number_input("🌍 Region ID", min_value=1, step=1)

# Prediction button
if st.button("🔍 Predict Rating"):
    if not all([VisitYear, VisitMonth, VisitModeName, AttractionId, Attraction, AttractionType, CountryId, RegionId]):
        st.error("⚠️ Please provide all inputs before predicting!")
    else:
        # Prepare input data
        input_data = pd.DataFrame({
            "VisitYear": [VisitYear],
            "VisitMonth": [VisitMonth],
            "VisitModeName": [VisitModeName],
            "AttractionId": [AttractionId],
            "Attraction": [Attraction],
            "AttractionType": [AttractionType],
            "CountryId": [CountryId],
            "RegionId": [RegionId]
        })

        # Encoding categorical variables
        encoded_features = ohe.transform(input_data[["VisitModeName", "AttractionType"]])
        encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(["VisitModeName", "AttractionType"]))

        # Encode the target variable
        input_data["Attraction"] = target_enc.transform(input_data["Attraction"])

        # Drop original categorical columns
        input_data.drop(columns=["VisitModeName", "AttractionType"], inplace=True)

        # Merge with encoded features
        input_data = pd.concat([input_data, encoded_df], axis=1)

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Predict using the trained model
        prediction = best_xgb.predict(input_scaled)

        # Display prediction
        st.success(f"⭐ Predicted Tourism Rating: {prediction[0]:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

# Preview dataset
st.subheader("📊 Preview of Tourism Data")
df_path = r"D:\1final ds\final_ds2.csv"
try:
    df = pd.read_csv(df_path)
    st.dataframe(df.head(10))  # Display only the first 10 rows
except FileNotFoundError:
    st.error("❌ Error: Dataset file not found. Please check the file path.")
