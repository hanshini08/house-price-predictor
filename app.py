import streamlit as st
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# Load dataset
housing = fetch_california_housing(as_frame=True)
data = housing.frame

# Select features
X = data[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']]
y = data['MedHouseVal']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Page settings
st.set_page_config(page_title="Smart House Price Predictor", page_icon="🏠")

st.title("🏠 Smart House Price Predictor")
st.write("Enter property details below to estimate house price.")

income = st.number_input("Median Income", min_value=0.0, step=0.1)
age = st.number_input("House Age", min_value=0)
rooms = st.number_input("Average Rooms", min_value=0.0, step=0.1)
bedrooms = st.number_input("Average Bedrooms", min_value=0.0, step=0.1)
population = st.number_input("Population", min_value=0)

if st.button("Predict Price"):
    user_input = np.array([[income, age, rooms, bedrooms, population]])
    prediction = model.predict(user_input)
    predicted_price = prediction[0] * 100000

    st.success(f"Estimated House Price: ${predicted_price:,.2f}")

    if predicted_price > 500000:
        st.info("Premium Property – High Investment Value")
    elif predicted_price > 250000:
        st.info("Mid-Range Property – Good Investment")
    else:
        st.warning("Budget Property – Affordable Option")
