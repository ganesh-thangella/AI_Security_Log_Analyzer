import streamlit as st
import pandas as pd
from model import train_model, load_model, predict
import os

st.set_page_config(page_title="AI Security Log Analyzer")

st.title("🔐 AI Security Log Analyzer (SOC + ML)")

uploaded_file = st.file_uploader("Upload Log File (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Logs")
    st.dataframe(df)

    if st.button("Train Model"):
        model = train_model(df)
        st.success("Model Trained Successfully!")

    if os.path.exists("anomaly_model.pkl"):
        model = load_model()
        result_df = predict(model, df)

        st.subheader("Analyzed Logs")
        st.dataframe(result_df)

        anomalies = result_df[result_df['anomaly'] == -1]

        st.subheader("🚨 Detected Threats")
        st.dataframe(anomalies)

        st.metric("Total Logs", len(result_df))
        st.metric("Anomalies Detected", len(anomalies))