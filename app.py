import streamlit as st
import pandas as pd
import numpy as np
from inferencing.predict_invoice_flag import predict_flag_invoice
from inferencing.predict_freight import predict_freight_cost



st.set_page_config(
    page_title="Vendor Invoice Intelligence",
    page_icon="📦",
    layout="wide"
)

st.markdown("""" 
    # Vendor Invoice Intelligence Portal
    ### AI-Driven Freight Cost Prediction & Invoice Risk Flagging
    This internal analytics portal leverages machine learning to
    - **Forecast freight costs accurately**
    - **Detect, risky or abnormal vendor invoices**
    - **Reduce'financial leakaqe and manual workload**
""")


st.divider()

st.sidebar.title("🔍Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    ["Freight Cost Prediction","Invoice Manual Approval Flag"]
)


st.sidebar.markdown("""
**Business Impact**
- Improved cost forecasting
- Reduced invoice fraud & anomalies
- Faster finance operations
"""
)


if (selected_model=='Freight Cost Prediction'):
    
    st.markdown("""
        **Objective:**
        Predict freight cost for a vendor invoice using **Quantity** and **Invoice Dollars** to support budgeting, forecasting, and vendor negotiations.
    """)

    with st.form("Freight Form"):
        col1,col2 = st.columns(2)

        with col1:
            quantity = st.number_input("Quantity",min_value=1,value=1200)
        
        with col2:
            dollars = st.number_input("invoice dollars ",min_value=1.0,value=18500.0)

        submit_freight = st.form_submit_button("Predict Freight Cost")

    if submit_freight:
        input_data = {
            "Dollars":[dollars],
            "Quantity":[quantity]
        }
        prediction = predict_freight_cost(input_data)

        st.success("prediction completed Successfully ")

        st.metric(
            label="Estimated Freight Cost",
            value=f"{prediction['Predicted_Freight'][0]:.2f}"
        )

else :
    st.subheader("Invoice Manual Approval Flag")

    st.markdown("""
        **Objective:**
        Predict whether a vendor invoice should be **flagged for manual approval** based on abnormal cost, freight, or delivery patterns.
    """)

    with st.form("invoice_flag_form"):
        col1,col2,col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input("Invoice Quantity",min_value=1,value=50)
            freight = st.number_input("Freight Cost",min_value=0.0,value=1.73)
        with col2:
            invoice_dollars = st.number_input("Invoice Dollars",min_value=1.0,value=352.95)
            total_item_quantity = st.number_input("Total Item Quantity",min_value=1,value=162)
        with col3:
            total_item_dollars = st.number_input("Total Item Dollars",min_value=1.0,value=2476.0)
        
        submit_flag = st.form_submit_button("Predict Flag")
    

    if submit_flag:
        input_data = {
            "invoice_quantity":[invoice_quantity],
            "invoice_dollars":[invoice_dollars],
            "Freight":[freight],
            "total_item_quantity":[total_item_quantity],
            "total_item_dollars":[total_item_dollars]
        }
        flag_prediction = predict_flag_invoice(input_data)

        is_flagged = bool(flag_prediction[0])


        if is_flagged:
            st.error("Invoice require manual Approval")
        else :
            st.success("Invoice is safe for auto-approval")


        