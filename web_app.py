
import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from streamlit_lottie import st_lottie
import json

# Load the model
pipeline = pk.load(open('ML_final.pkl', 'rb'))

st.title('Car Price Prediction Model')

# lottie_animation = load_lottie_file("C:\Users\99bis\Downloads\Animation - 1716398637329.json")

st.markdown("""
### About this Application
This is a web application where you can predict the reselling price of a car depending on various factors such as the car brand, manufacturing year, kilometers driven, fuel type, seller type, transmission type, owner type, mileage, engine capacity, maximum power, and the number of seats.

This web application is based on combination of 3 machine learning algorithm with an accuracy of 92.5 percent.

Simply input the details of the car in the sidebar, and click on the **Predict** button to get the estimated reselling price.
""")

# st_lottie(lottie_animation, height=200, key="header_animation")

# page_bg_img = '''
# <style>
# body {
# background-image: url("C:\Users\99bis\Downloads\pexels-trace-707046.jpg");
# background-size: cover;
# }
# </style>
# '''
# Load dataset
cars_data = pd.read_csv(r"C:\Users\99bis\Used Car Price prediction ML\Dataset\Cardetails.csv")

def get_brand_name(car_name):
    car_name = car_name.split()[0]
    return car_name.strip(' ')

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Sidebar inputs
st.sidebar.header('Input Features')
name = st.sidebar.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.sidebar.slider("Car Manufactured Year", 1994, 2020)
km_driven = st.sidebar.slider("No. of kilometers driven", 10, 500000)
fuel = st.sidebar.selectbox('Select Fuel Type', cars_data['fuel'].unique())
seller_type = st.sidebar.selectbox('Seller Type', cars_data['seller_type'].unique())
transmission = st.sidebar.selectbox('Transmission Type', cars_data['transmission'].unique())
owner = st.sidebar.selectbox("Owner Type", cars_data['owner'].unique())
mileage = st.sidebar.slider("Car Mileage kmpl", 1, 40)
engine = st.sidebar.slider("Car Engine in CC", 500, 5000)
max_power = st.sidebar.slider("Max Power of Car in bhp", 0, 250)
seats = st.sidebar.slider("Enter No. of seats", 2, 20)

# Prediction button
if st.sidebar.button("Predict"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )
    
    st.subheader('Input Data')
    st.write(input_data_model)
    
    predict = pipeline.predict(input_data_model)
    
    st.subheader('Predicted Car Price')
    st.write(f"{predict[0]:,.2f}")
