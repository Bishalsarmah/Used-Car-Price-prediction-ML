import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

pipeline = pk.load(open('ML_final.pkl','rb'))

st.header('Car Price Prediction Model')

cars_data = pd.read_csv(r"C:\Users\99bis\Used Car Price prediction ML\Dataset\Cardetails.csv")


def get_brand_name(car_name):
    car_name = car_name.split()[0]
    return car_name.strip(' ')

cars_data['name'] = cars_data['name'].apply(get_brand_name)

name = st.selectbox('Select Car Brand',cars_data['name'].unique())

year = st.slider("Car Manufactured Year",1994,2020)

km_driven = st.slider("No. of kilometers driven",10,500000)

fuel = st.selectbox('Select Fuel Type ',cars_data['fuel'].unique())

seller_type = st.selectbox('Seller Type',cars_data['seller_type'].unique())

transmission = st.selectbox('Transmission Type',cars_data['transmission'].unique())

owner = st.selectbox("Owner Type",cars_data['owner'].unique())

mileage = st.slider("Car Mileage kmpl ",1,40)

engine = st.slider("Car Engine in CC ",500,5000)

max_power = st.slider("Max Power of Car in bhp",0,250)

seats = st.slider("Enter No. of seats",2,20)



if st.button("Predict"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )
    st.write(input_data_model)
    predict = pipeline.predict(input_data_model)
    st.write(f"Predicted Car Price: {predict[0]}")