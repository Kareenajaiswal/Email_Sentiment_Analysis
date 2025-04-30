# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sentiment_logic import analyze_sentiment

st.set_page_config(page_title="Email Sentiment Analyzer", layout="centered")

st.title("📬 Email Sentiment Analyzer")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if "email_body" not in df.columns:
        st.error("CSV must contain a 'body' column with email content.")
    else:
        st.success("File uploaded successfully!")

        df["Sentiment"] = df["email_body"].apply(analyze_sentiment)

        st.dataframe(df[["email_body", "Sentiment"]], height=300)

        # Plot sentiment distribution
        sentiment_counts = df["Sentiment"].value_counts()
        fig, ax = plt.subplots()
        colors = {"positive": "orange", "negative": "red", "neutral": "green"}
        sentiment_counts.plot(kind="bar", color=[colors.get(s, "blue") for s in sentiment_counts.index], ax=ax)
        plt.title("Sentiment Distribution")
        plt.ylabel("Number of Emails")
        st.pyplot(fig)

        # Download result
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV with Sentiment", csv, "emails_with_sentiment.csv", "text/csv")
