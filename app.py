import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Page Configuration
st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="⭐",
    layout="wide"
)

# Load Files
@st.cache_resource
def load_model():
    model = pickle.load(open("models/model.pkl", "rb"))
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
    return model, vectorizer

@st.cache_data
def load_data():
    return pd.read_csv("clean_reviews.csv")

model, vectorizer = load_model()
df = load_data()

# Sidebar Navigation
page = st.sidebar.radio(
    "Go to",
    ["Home", "Prediction", "Dataset Statistics", "Word Cloud","Model Performance" ,"About"]
)

# HOME
if page == "Home":
    st.title("⭐ Amazon Review Sentiment Analysis Dashboard")
    st.write("""
    Predict whether an Amazon customer review is **Positive** or **Negative**
    using Natural Language Processing and Machine Learning.
    """)

    positive = (df["Sentiment"] == "Positive").sum()
    negative = (df["Sentiment"] == "Negative").sum()
    total = len(df)

    positive_percent = round((positive / total) * 100, 2)
    negative_percent = round((negative / total) * 100, 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", total)
    c2.metric("Positive", positive)
    c3.metric("Negative", negative)
    c4.metric("Accuracy", "89%")

    st.divider()

    sentiment_counts = df["Sentiment"].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Review Sentiment Distribution",
        hole=0.55,
        color=sentiment_counts.index,
        color_discrete_map={"Positive":"green","Negative":"red"}
    )
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# PREDICTION
elif page == "Prediction":
    review = st.text_area(
        "Enter Customer Review",
        height=170,
        placeholder="Example: Amazing product. Fast delivery."
    )

    if st.button("Predict Sentiment", use_container_width=True):
        if review.strip() == "":
            st.warning("Please enter a review.")
        else:
            vector = vectorizer.transform([review])
            prediction = model.predict(vector)[0]
            probs = model.predict_proba(vector)[0]
            confidence = max(probs)

            if prediction == "Positive":
                st.success("😊 Positive Review")
            else:
                st.error("😞 Negative Review")

            st.write(f"Confidence : **{confidence*100:.2f}%**")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Negative","Positive"],
                y=probs,
                text=[f"{i:.2%}" for i in probs],
                textposition="outside"
            ))
            fig.update_layout(title="Prediction Probability", template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)

# DATASET STATISTICS
elif page == "Dataset Statistics":
    st.title("📊 Dataset Statistics")

    positive = (df["Sentiment"]=="Positive").sum()
    negative = (df["Sentiment"]=="Negative").sum()

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Positive",positive)
    col2.metric("Negative",negative)
    col3.metric("Total",len(df))
    col4.metric("Average Rating", round(df["Rating_Num"].mean(),2))

    st.divider()

    sentiment_counts = df["Sentiment"].value_counts()
    fig = px.bar(
        sentiment_counts,
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        color=sentiment_counts.index,
        text=sentiment_counts.values,
        title="Sentiment Distribution"
    )
    fig.update_layout(template="plotly_dark", showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(
        df,
        x="Rating_Num",
        nbins=5,
        color="Sentiment",
        title="Rating Distribution"
    )
    fig2.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig2, use_container_width=True)

# WORD CLOUD
elif page == "Word Cloud":
    st.title("☁️ Word Cloud")
    sentiment = st.selectbox("Select Sentiment", ["Positive","Negative"])
    text = " ".join(df[df["Sentiment"]==sentiment]["Clean_Review"])

    wc = WordCloud(width=1000, height=500, background_color="black").generate(text)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

# MODEL PERFORMANCE
elif page == "Model Performance":
    st.title("📈 Model Performance Metrics")

    # Classification report values multiplied by 100
    report_data = {
        "Class": ["Negative", "Positive", "Accuracy", "Macro Avg", "Weighted Avg"],
        "Precision (%)": [94, 89, None, 91, 92],
        "Recall (%)": [96, 82, None, 89, 92],
        "F1-Score (%)": [95, 85, 92, 90, 92],
        "Support": [3066, 1124, 4190, 4190, 4190]
    }

    df_report = pd.DataFrame(report_data)

    # Display as table
    st.dataframe(df_report)

    # Visualize with bar chart
    fig = px.bar(
        df_report.dropna().melt(id_vars="Class", value_vars=["Precision (%)","Recall (%)","F1-Score (%)"]),
        x="Class",
        y="value",
        color="variable",
        barmode="group",
        text="value",
        title="Performance Metrics by Class (%)"
    )
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Summary:**
    - Negative reviews: Precision **94%**, Recall **96%**, F1-score **95%**.
    - Positive reviews: Precision **89%**, Recall **82%**, F1-score **85%**.
    - Overall Accuracy: **92%** across 4190 reviews.
    - Macro average F1-score: **90%**, showing balanced performance.
    """)

# ABOUT
elif page == "About":
    st.title("ℹ️ About")
    st.write("This dashboard demonstrates Amazon Review Sentiment Analysis using NLP and ML.")

# Footer
st.markdown("---")
st.caption("Developed by Mayank Raj | Machine Learning | NLP | Streamlit")
