import streamlit as st
import pickle

# Load the trained model from the pickle file
with open('data.pkl', 'rb') as file:
    model = pickle.load(file)   

data1 = st.slider(
    "Enter a value",
    min_value=1.0,
    max_value=200.0,
    step=0.2,
    key="data1_slider"
)


df = [data1]

if st.button('submit'):
    prediction = model.predict([[data1]])
    st.write(f'The predicted value is: {prediction[0]}')