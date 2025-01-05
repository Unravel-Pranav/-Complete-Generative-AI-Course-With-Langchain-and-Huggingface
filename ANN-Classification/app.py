import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the trained model
model = load_model('model.keras')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

# User input 
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
gender = st.selectbox('Gender', label_encoder_gender.classes_)
credit_score = st.number_input('Credit Score', min_value=0, max_value=850)
tenure = st.number_input('Tenure', min_value=0, max_value=10, step=1)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, step=1)
has_cr_card = st.selectbox('Has Credit Card', options=[0, 1])
is_active_member = st.selectbox('Is Active Member', options=[0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]  # Assumes model outputs a probability for churn

# Display the prediction
st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn')
