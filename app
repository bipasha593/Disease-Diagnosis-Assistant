import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model and encoder
model_data = joblib.load("disease_model.pkl")
disease_mapping = joblib.load("label_encoder.pkl")

# Load dataset
df = pd.read_csv("disease_symptoms.csv")
symptoms = df.columns[:-1]  # Exclude 'Disease' column

# Function to predict disease based on symptom similarity
def predict_disease(symptom_inputs):
    X_train = model_data["X_train"]
    y_train = model_data["y_train"]
    int_to_disease = model_data["int_to_disease"]

    # Find the disease with the highest symptom match score
    match_scores = np.sum(X_train * symptom_inputs, axis=1)  # Element-wise match count
    best_match_index = np.argmax(match_scores)

    predicted_disease = int_to_disease[y_train[best_match_index]]
    return predicted_disease

# Streamlit UI
st.set_page_config(page_title="Disease Diagnosis Assistant", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Predict Disease"])

# Home Page
if page == "Home":
    st.markdown("<h1 style='text-align: center; font-size: 50px;'>🩺 Disease Diagnosis Assistant</h1>",
                unsafe_allow_html=True)

    # Centering text using HTML & CSS
    st.markdown("""
        <div style="text-align: center; font-size: 20px; margin-top: 20px;">
            <p><b>Welcome to the Disease Diagnosis Assistant!</b></p>
            <p>Go to the navigation bar and click <b>Predict Disease</b> to diagnose.</p>
        </div>
    """, unsafe_allow_html=True)

# Prediction Page
elif page == "Predict Disease":
    st.title("🔍 Predict Your Disease")
    st.write("Select the symptoms you are experiencing and click 'Predict'.")

    # User selects symptoms
    user_symptoms = [st.checkbox(symptom, False) for symptom in symptoms]

    # Convert boolean inputs to 1 (selected) and 0 (not selected)
    symptom_inputs = np.array(user_symptoms).reshape(1, -1)

    # Predict button
    if st.button("Predict"):
        predicted_disease = predict_disease(symptom_inputs)
        st.success(f"🩺 Based on your symptoms, you may have **{predicted_disease}**.") 
