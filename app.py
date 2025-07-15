import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("sales_predictor.pkl")

# Title
st.title("📊 SmartInsight Pro - Sales Prediction Dashboard")

# Sidebar input
st.sidebar.header("Enter Input Features")
inventory = st.sidebar.number_input("Inventory Level", 0, 1000, 100)
unit_price = st.sidebar.number_input("Unit Price", 0.0, 1000.0, 50.0)
discount = st.sidebar.slider("Discount (%)", 0, 100, 10)
month = st.sidebar.selectbox("Month", list(range(1, 13)))
weekday = st.sidebar.selectbox("Weekday", list(range(0, 7)))  # 0=Monday

# Prediction Button
if st.sidebar.button("Predict Sales"):
    input_df = pd.DataFrame([{
        'inventory_level': inventory,
        'unit_price': unit_price,
        'discount': discount,
        'month': month,
        'weekday': weekday
    }])
    
    input_df.rename(columns={'unit_price': 'price'}, inplace=True)


    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted Units Sold: **{round(prediction)}**")

    # Chart: bar chart of input features
    st.subheader("🔎 Input Feature Breakdown")
    fig = px.bar(
        x=input_df.columns,
        y=input_df.values[0],
        labels={"x": "Feature", "y": "Value"},
        title="Input Features Used for Prediction"
    )
    st.plotly_chart(fig)

    # Optional: show pie or line chart
    st.subheader("📈 Prediction as Pie Slice")
    fig2 = px.pie(
        names=["Predicted", "Remaining Stock"],
        values=[prediction, max(inventory - prediction, 0)],
        title="Predicted Units vs Remaining Inventory"
    )
    st.plotly_chart(fig2)
      # Add to results table
    st.subheader("🔢 Predicted Data Table")
    input_df["predicted_sales"] = round(prediction)
    st.dataframe(input_df)
