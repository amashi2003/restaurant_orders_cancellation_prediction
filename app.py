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
# User Inputs (numeric + essential categorical)
# -----------------------
st.header("Enter Order Details:")

# Numeric Inputs
order_amount = st.number_input("Order Amount ($)", min_value=0.0, value=50.0, step=1.0)
number_of_items = st.number_input("Number of Items", min_value=1, max_value=50, value=5, step=1)
distance_km = st.number_input("Distance (km)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
customer_rating = st.number_input("Customer Rating (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=10, value=0, step=1)
delivery_duration_min = st.number_input("Delivery Duration (minutes)", min_value=1, max_value=200, value=30, step=1)

# Categorical Inputs (used for prediction)
delivery_type = st.selectbox("Delivery Type", ["home delivery", "pick-up"])
day_of_week = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

# -----------------------
# Build input DataFrame
# -----------------------
input_dict = {
    'order_amount': order_amount,
    'number_of_items': number_of_items,
    'distance_km': distance_km,
    'customer_rating': customer_rating,
    'previous_cancellations': previous_cancellations,
    'delivery_duration_min': delivery_duration_min
}

# One-hot encode essential categorical variables
# Delivery type
input_dict['delivery_type_home delivery'] = 1 if delivery_type == "home delivery" else 0

# Day of week
for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
    input_dict[f'day_of_week_{day}'] = 1 if day_of_week == day else 0

# Fill remaining columns (removed categorical features) with 0
for col in model_columns:
    if col not in input_dict:
        input_dict[col] = 0

# Convert to DataFrame and reorder columns
input_df = pd.DataFrame([input_dict])[model_columns]

# -----------------------
# Prediction
# -----------------------
if st.button("Predict Cancellation"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("Prediction: Order will be CANCELED")
    else:
        st.success("Prediction: Order will NOT be canceled")

    st.info(f"Probability of cancellation: {probability:.2f}")
