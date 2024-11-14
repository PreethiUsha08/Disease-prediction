import pickle
import pandas as pd
import streamlit as st
import xgboost as xgb

# Load the trained XGBoost model from the pickle file
with open('xgboost_model (2).pkl', 'rb') as file:
    model = pickle.load(file)

# Define a function to make a prediction based on input features
def predict_disease( age, sex, albumin, alkaline_phosphatase, alanine_aminotransferase,
    aspartate_aminotransferase, bilirubin, cholinesterase, cholesterol,
    creatinina, gamma_glutamyl_transferase, protein):
    """
    Predicts the disease presence based on input features.
    Parameters:
        input_data (dict): A dictionary of features with values for each required input.
    Returns:
        str: The predicted category (e.g., 'no_disease', 'severe_disease', 'hepatitis', 'fibrosis', 'cirrhosis').
    """
   if sex == "male":
       sex = 1
   else:
       sex = 0

    # Convert input data into a DataFrame
    features = np.array([age, sex, albumin, alkaline_phosphatase, alanine_aminotransferase,
    aspartate_aminotransferase, bilirubin, cholinesterase, cholesterol,
    creatinina, gamma_glutamyl_transferase, protein])

   
    # Use the model to make a prediction
    prediction = model.predict(features)
    
    # Map prediction to disease category
    disease_mapping = {
        3: 'No Disease',
        4: 'Suspect_Disease',
        2: 'Hepatitis',
        1: 'Fibrosis',
        0: 'Cirrhosis'
    }
    
    return disease_mapping.get(prediction[0], "Unknown")

# Streamlit UI elements
st.title("Disease Prediction")
st.write("Enter the following details to predict the disease category:")

# Input fields for user to enter data
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["m", "f"])
albumin = st.number_input("Albumin", min_value=0.0, value=38.5)
alkaline_phosphatase = st.number_input("Alkaline Phosphatase", min_value=0.0, value=52.5)
alanine_aminotransferase = st.number_input("Alanine Aminotransferase", min_value=0.0, value=7.7)
aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0.0, value=22.1)
bilirubin = st.number_input("Bilirubin", min_value=0.0, value=7.5)
cholinesterase = st.number_input("Cholinesterase", min_value=0.0, value=6.93)
cholesterol = st.number_input("Cholesterol", min_value=0.0, value=3.23)
creatinina = st.number_input("Creatinina", min_value=0, value=106)
gamma_glutamyl_transferase = st.number_input("Gamma Glutamyl Transferase", min_value=0.0, value=12.1)
protein = st.number_input("Protein", min_value=0.0, value=69.0)

# Button to trigger prediction
if st.button("Predict"):
    input_data = {
        "age": age,
        "sex": sex,
        "albumin": albumin,
        "alkaline_phosphatase": alkaline_phosphatase,
        "alanine_aminotransferase": alanine_aminotransferase,
        "aspartate_aminotransferase": aspartate_aminotransferase,
        "bilirubin": bilirubin,
        "cholinesterase": cholinesterase,
        "cholesterol": cholesterol,
        "creatinina": creatinina,
        "gamma_glutamyl_transferase": gamma_glutamyl_transferase,
        "protein": protein
    }
    
    # Make prediction
    result = predict_disease(input_data)
    
    # Display result
    st.success(f"Prediction: {result}")
