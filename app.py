# app.py

import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="Restaurant Order Cancellation Prediction",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Custom CSS Styling
# -----------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: bold;
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    .input-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 5px solid #667eea;
    }
    
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        border: 2px solid #e9ecef;
    }
    
    .success-card {
        border-color: #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    
    .error-card {
        border-color: #dc3545;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 4px solid #667eea;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #ced4da;
    }
    
    .stSelectbox > div > div > div {
        background-color: white;
        color: #495057;
    }
    
    .stSelectbox > div > div > div > div {
        background-color: white;
        color: #495057;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ced4da;
    }
    
    .stSelectbox label {
        color: #495057;
        font-weight: 500;
    }
    
    /* Ensure selectbox dropdown is visible */
    .stSelectbox > div > div > div[data-baseweb="select"] {
        background-color: white;
        border: 1px solid #ced4da;
        border-radius: 8px;
    }
    
    /* Style the selected value display */
    .stSelectbox > div > div > div > div[data-baseweb="select"] > div {
        background-color: white;
        color: #495057;
        padding: 0.5rem;
    }
    
    /* Ensure dropdown options are visible */
    .stSelectbox > div > div > div[data-baseweb="select"] > div[role="listbox"] {
        background-color: white;
        border: 1px solid #ced4da;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Load trained model and training columns
# -----------------------
model = joblib.load('random_forest_order_cancellation.pkl')
model_columns = joblib.load('model_columns.pkl')  # Columns used in training

# -----------------------
# Header Section
# -----------------------
st.markdown("""
<div class="main-header">
    <h1>🍽️ Restaurant Order Cancellation Prediction</h1>
    <p>Advanced AI-powered system to predict order cancellations and optimize restaurant operations</p>
</div>
""", unsafe_allow_html=True)

# -----------------------
# User Inputs Section
# -----------------------
#st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown("### 📝 Order Information")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💰 Order Details")
    order_amount = st.number_input(
        "💰 Order Amount ($)", 
        min_value=0.0, 
        value=0.0, 
        step=1.0,
        help="Total amount of the order in dollars"
    )
    number_of_items = st.number_input(
        "📦 Number of Items", 
        min_value=1, 
        max_value=50, 
        value=1, 
        step=1,
        help="Total number of items in the order"
    )
    distance_km = st.number_input(
        "🚚 Distance (km)", 
        min_value=0.0, 
        max_value=50.0, 
        value=0.0, 
        step=0.1,
        help="Delivery distance in kilometers"
    )

with col2:
    st.markdown("#### ⭐ Customer & Delivery Info")
    customer_rating = st.number_input(
        "⭐ Customer Rating (1-5)", 
        min_value=1.0, 
        max_value=5.0, 
        value=1.0, 
        step=0.1,
        help="Customer's average rating"
    )
    previous_cancellations = st.number_input(
        "❌ Previous Cancellations", 
        min_value=0, 
        max_value=10, 
        value=0, 
        step=1,
        help="Number of previous cancellations by this customer"
    )
    delivery_duration_min = st.number_input(
        "⏱️ Delivery Duration (minutes)", 
        min_value=1, 
        max_value=200, 
        value=1, 
        step=1,
        help="Estimated delivery time in minutes"
    )

# Additional inputs in a new row
st.markdown("#### 🚚 Delivery Preferences")
col3, col4 = st.columns(2)

with col3:
    delivery_type = st.selectbox(
        "🚚 Delivery Type", 
        ["", "home delivery", "pick-up"],
        help="Choose delivery method"
    )

with col4:
    day_of_week = st.selectbox(
        "📅 Day of Week", 
        ["", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        help="Day when the order is placed"
    )

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Prediction Section
# -----------------------
st.markdown("### 🔮 Prediction Analysis")

# Add a centered prediction button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🚀 Predict Cancellation Risk", use_container_width=True)

if predict_button:
    # Show loading spinner
    with st.spinner('🤖 Analyzing order data...'):
        # Build input DataFrame
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

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    # Display results with enhanced UI
    st.markdown("---")
    
    # Create columns for results
    col_result1, col_result2 = st.columns([2, 1])
    
    with col_result1:
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-card error-card">
                <h2 style="color: #dc3545; margin-top: 0;">⚠️ HIGH RISK</h2>
                <h3 style="color: #dc3545;">Order Likely to be CANCELED</h3>
                <p style="font-size: 1.1rem; margin-bottom: 0;">Based on the provided order details, this order has a high probability of being canceled.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card success-card">
                <h2 style="color: #28a745; margin-top: 0;">✅ LOW RISK</h2>
                <h3 style="color: #28a745;">Order Likely to be COMPLETED</h3>
                <p style="font-size: 1.1rem; margin-bottom: 0;">Based on the provided order details, this order has a low probability of being canceled.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_result2:
        # Probability visualization
        risk_level = "High" if probability > 0.5 else "Low"
        risk_color = "#dc3545" if probability > 0.5 else "#28a745"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {risk_color}; margin-top: 0;">📊 Risk Level</h3>
            <h2 style="color: {risk_color}; font-size: 2rem; margin: 0.5rem 0;">{risk_level}</h2>
            <div style="background: #e9ecef; border-radius: 10px; padding: 0.5rem; margin: 1rem 0;">
                <div style="background: {risk_color}; height: 20px; border-radius: 10px; width: {probability*100}%;"></div>
            </div>
            <p style="font-size: 1.2rem; font-weight: bold; margin: 0;">{probability:.1%}</p>
            <p style="color: #6c757d; margin: 0;">Cancellation Probability</p>
            <hr style="margin: 1rem 0; border: none; border-top: 1px solid #dee2e6;">
            <h4 style="color: {risk_color}; margin: 0.5rem 0;">📈 Detailed Analysis</h4>
            <p style="font-size: 0.9rem; color: #6c757d; margin: 0.2rem 0;">• Probability: <strong>{probability:.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional insights section
    st.markdown("### 📈 Order Analysis Insights")
    
    insight_cols = st.columns(3)
    
    with insight_cols[0]:
        if order_amount > 100:
            st.success("💰 High-value order")
        elif order_amount < 20:
            st.warning("💰 Low-value order")
        else:
            st.info("💰 Medium-value order")
    
    with insight_cols[1]:
        if customer_rating >= 4.5:
            st.success("⭐ Excellent customer")
        elif customer_rating <= 2.5:
            st.error("⭐ Poor customer rating")
        else:
            st.info("⭐ Average customer rating")
    
    with insight_cols[2]:
        if previous_cancellations == 0:
            st.success("✅ No cancellation history")
        elif previous_cancellations >= 3:
            st.error("❌ High cancellation history")
        else:
            st.warning("⚠️ Some cancellation history")
