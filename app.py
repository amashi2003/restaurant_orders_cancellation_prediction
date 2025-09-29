# app.py

import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Load trained model and training columns
# -----------------------
model = joblib.load('random_forest_order_cancellation.pkl')
model_columns = joblib.load('model_columns.pkl')  # Columns used in training

st.title("Restaurant Order Cancellation Prediction")
st.write("Predict if an order will be canceled based on order details.")

# -----------------------
# User Inputs
# -----------------------
st.header("Enter Order Details:")

# Numeric Inputs
order_amount = st.number_input("Order Amount ($)", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
number_of_items = st.number_input("Number of Items", min_value=1, max_value=50, value=5, step=1)
distance_km = st.number_input("Distance (km)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
customer_rating = st.number_input("Customer Rating (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=10, value=0, step=1)
delivery_duration_min = st.number_input("Delivery Duration (minutes)", min_value=1, max_value=200, value=30, step=1)

# Categorical Inputs
delivery_type = st.selectbox("Delivery Type", ["home delivery", "pick-up"])
payment_method = st.selectbox("Payment Method", ["cash", "debit card", "credit card", "apple pay", "other"])
day_of_week = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
restaurant = st.selectbox("Restaurant", ["Chipotle", "Dominos", "McDonalds", "PizzaHut", "Subway", "TacoBell"])
city = st.selectbox("City", ["Houston", "Los Angeles", "New York", "Philadelphia", "Phoenix", "San Antonio"])

# -----------------------
# Preprocess input for model
# -----------------------
input_dict = {
    'order_amount': order_amount,
    'number_of_items': number_of_items,
    'distance_km': distance_km,
    'customer_rating': customer_rating,
    'previous_cancellations': previous_cancellations,
    'delivery_duration_min': delivery_duration_min
}

# One-hot encode categorical variables

# Delivery type
input_dict['delivery_type_home delivery'] = 1 if delivery_type == "home delivery" else 0

# Payment method
for method in ["debit card", "credit card", "apple pay", "other"]:
    input_dict[f'payment_method_{method}'] = 1 if payment_method == method else 0

# Day of week
for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
    input_dict[f'day_of_week_{day}'] = 1 if day_of_week == day else 0

# Restaurants
for r in ["Chipotle", "Dominos", "McDonalds", "PizzaHut", "Subway", "TacoBell"]:
    input_dict[f'restaurant_{r}'] = 1 if restaurant == r else 0

# Cities
for c in ["Houston", "Los Angeles", "New York", "Philadelphia", "Phoenix", "San Antonio"]:
    input_dict[f'city_{c}'] = 1 if city == c else 0

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# -----------------------
# Fix missing/unseen columns
# -----------------------
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training
input_df = input_df[model_columns]

# -----------------------
# Prediction
# -----------------------
if st.button("Predict Cancellation"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # probability of cancellation

    if prediction == 1:
        st.error(f"Prediction: Order will be CANCELED")
    else:
        st.success(f"Prediction: Order will NOT be canceled")

    st.info(f"Probability of cancellation: {probability:.2f}")
