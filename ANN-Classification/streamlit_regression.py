import pickle
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model 
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the trained regression model
model = load_model('regression_model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title('Estimated Salary Prediction')

# User input 
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
gender = st.selectbox('Gender', label_encoder_gender.classes_)
credit_score = st.number_input('Credit Score', min_value=0, max_value=850)
exited=st.selectbox('Exited',[0,1])
tenure = st.slider('Tenure', [0,10])
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, step=1)
has_cr_card = st.selectbox('Has Credit Card', options=[0, 1])
is_active_member = st.selectbox('Is Active Member', options=[0, 1])

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
    'Exited': [exited]
})
# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data (ensure scaler was fitted without 'EstimatedSalary')
input_data_scaled = scaler.transform(input_data)

# Predict the estimated salary
predicted_salary = model.predict(input_data_scaled)
predicted_salary_value = predicted_salary[0][0]  # Assuming model outputs a single value

# Display the predicted estimated salary
st.write(f'Estimated Salary: ${predicted_salary_value:.2f}')
