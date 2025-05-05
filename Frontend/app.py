import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Page config
st.set_page_config(page_title="Email Sentiment Analysis", layout="wide")

# Load CSV
def load_csv(file):
    data = pd.read_csv(file)
    sentiment_col = None
    for col in data.columns:
        if col.strip().lower() == 'sentiment':
            sentiment_col = col
            break
    if sentiment_col:
        data[sentiment_col] = data[sentiment_col].str.lower()
        data.rename(columns={sentiment_col: 'sentiment'}, inplace=True)
    else:
        st.error("❌ Sentiment column not found in the uploaded file.")
        return None
    return data

# Preprocess features
def preprocess_data(X):
    label_encoder = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = label_encoder.fit_transform(X[col].astype(str))
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled

# Train models
def train_models(data):
    X = data.drop('sentiment', axis=1)
    y = data['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
    X = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    accuracies, conf_matrices = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, y_pred)
        conf_matrices[name] = confusion_matrix(y_test, y_pred)

    return accuracies, conf_matrices

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'],
                cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{model_name} Confusion Matrix')
    st.pyplot(fig)

# MAIN APP
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ("Dashboard", "Model Performance"))

    if "data" not in st.session_state:
        st.session_state["data"] = None

    if page == "Dashboard":
        st.markdown("<h1 style='text-align: center; color: #4B6CB7;'>📩 Email Sentiment Analysis</h1>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV File", type="csv")

        if uploaded_file:
            with st.spinner("🔄 Loading your data..."):
                time.sleep(1)
                data = load_csv(uploaded_file)
                if data is not None:
                    st.session_state["data"] = data
                    st.success("✅ File uploaded and data loaded successfully!")

        if st.session_state["data"] is not None:
            data = st.session_state["data"]
            sentiment_counts = data['sentiment'].value_counts()
            total = len(data)
            positive = sentiment_counts.get('positive', 0)
            negative = sentiment_counts.get('negative', 0)
            neutral = sentiment_counts.get('neutral', 0)

            st.markdown("### 🧾 Overview")
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Total Emails", total)
            colB.metric("Positive", positive)
            colC.metric("Negative", negative)
            colD.metric("Neutral", neutral)

            with st.expander("📊 View Sentiment Charts", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🎨 Pie Chart")
                    fig1, ax1 = plt.subplots()
                    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                            startangle=90, colors=['#f4f756', '#54d856', '#a3ffe8'])
                    ax1.axis('equal')
                    st.pyplot(fig1)
                with col2:
                    st.markdown("#### 📊 Bar Chart")
                    fig2, ax2 = plt.subplots()
                    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values,
                                palette='Set2', ax=ax2)
                    ax2.set_xlabel("Sentiment")
                    ax2.set_ylabel("Count")
                    st.pyplot(fig2)

            with st.expander("📋 View Full Data Table"):
                st.dataframe(data)

    elif page == "Model Performance":
        st.markdown("<h2 style='text-align: center; color: #4B6CB7;'>📈 Model Performance</h2>", unsafe_allow_html=True)
        if st.session_state["data"] is not None:
            data = st.session_state["data"]
            with st.spinner("⏳ Training models..."):
                accuracies, conf_matrices = train_models(data)

            st.markdown("### 🔍 Accuracy Comparison")
            for model, acc in accuracies.items():
                st.write(f"**{model}**: {acc:.2f}")

            # Display as bar chart
            acc_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
            fig_acc, ax_acc = plt.subplots()
            sns.barplot(data=acc_df, x="Model", y="Accuracy", palette="viridis", ax=ax_acc)
            ax_acc.set_ylim(0, 1)
            ax_acc.set_title("Model Accuracy Comparison")
            ax_acc.set_ylabel("Accuracy Score")
            ax_acc.set_xlabel("")
            st.pyplot(fig_acc)

            st.markdown("### 📊 Confusion Matrices")
            for model, matrix in conf_matrices.items():
                plot_confusion_matrix(matrix, model)
        else:
            st.warning("⚠️ Please upload a CSV file first from the Dashboard page.")

if __name__ == "__main__":
    main()
