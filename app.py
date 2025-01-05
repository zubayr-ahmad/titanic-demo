import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Set page title
st.title('Titanic Survival Prediction')

# Create input fields
st.subheader('Enter Passenger Information')

gender = st.radio('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=0, max_value=100, value=30)
p_class = st.selectbox('Passenger Class', [1, 2, 3])

# Convert gender to numeric
gender_numeric = 0 if gender == 'Male' else 1

# Create predict button
if st.button('Predict Survival'):
    # Prepare input data
    input_data = np.array([[gender_numeric, age, p_class]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Show prediction
    if prediction == 1:
        st.success(f'This passenger would likely SURVIVE with {probability:.1%} probability')
    else:
        st.error(f'This passenger would likely NOT SURVIVE with {(1-probability):.1%} probability')

# Add information about the model
st.sidebar.header('About')
st.sidebar.info('This app predicts the probability of survival on the Titanic based on passenger characteristics using a Random Forest model.')