import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Water Potability", layout="centered")
st.title("Water Potability")
st.write(
    "Input water metrics to evaluate if a specific source is safe or unsafe for human consumption "
    "using Random Forest Classifier model."
)

FEATURE_NAMES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

@st.cache_resource
def load_water_model():
    try:
        return joblib.load("rf_model.pkl")
    except FileNotFoundError:
        st.error("Error: 'rf_model.pkl' not found. Please place your model file in the current directory.")
        return None

model = load_water_model()
if model is not None:
    st.markdown("### Enter Water Quality Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ph = st.number_input("pH level (0 - 14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        chloramines = st.number_input("Chloramines (mg/L)", min_value=0.0, max_value=15.0, value=7.1, step=0.1)
        organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, max_value=30.0, value=14.2, step=0.1)
        
    with col2:
        hardness = st.number_input("Hardness (mg/L)", min_value=40.0, max_value=330.0, value=196.3, step=1.0)
        sulfate = st.number_input("Sulfate (mg/L)", min_value=120.0, max_value=490.0, value=333.7, step=1.0)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, max_value=130.0, value=66.4, step=0.1)
        
    with col3:
        solids = st.number_input("Total Dissolved Solids (ppm)", min_value=300.0, max_value=60000.0, value=22000.0, step=100.0)
        conductivity = st.number_input("Conductivity (μS/cm)", min_value=180.0, max_value=750.0, value=426.2, step=1.0)
        turbidity = st.number_input("Turbidity (NTU)", min_value=1.0, max_value=7.0, value=3.9, step=0.1)

    if st.button("Analyze Potability", type="primary"):
        user_input_array = np.array([[
            ph, hardness, solids, chloramines, sulfate, 
            conductivity, organic_carbon, trihalomethanes, turbidity
        ]])
        
        prediction = model.predict(user_input_array)[0]
        probabilities = model.predict_proba(user_input_array)[0]
        
        st.subheader("Prediction Results")
        
        if prediction == 1:
            st.success("✅ Result: Potable means safe for drinking.")
        else:
            st.error("❌ Result: Not Potable means unsafe for drinking.")
            
        st.write("**Model Prediction Percentage:**")
        st.write(f"Unsafe (Class 0): `{probabilities[0]:.2%}`")
        st.progress(float(probabilities[0]))
        st.write(f"Safe / Potable (Class 1): `{probabilities[1]:.2%}`")
        st.progress(float(probabilities[1]))
            