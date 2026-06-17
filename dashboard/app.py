import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd

import re
import contractions
import sys
import os
import nltk

nltk.download('stopwords')

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.data.preprocess import preprocess_text

from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px

# ======================
# CONFIG
# ======================
MAX_LEN = 200

st.set_page_config(
    page_title="Binary Sentiment Classification Dashboard (HBGRU v2)",
    layout="wide"
)

# ======================
# LOAD MODEL + TOKENIZER
# ======================
@st.cache_resource
def load_model(model):
    return tf.keras.models.load_model(model)

@st.cache_resource
def load_tokenizer(tokenizer):
    with open(tokenizer, "rb") as f:
        return pickle.load(f)


MODEL_PATH = "dashboard/hbgru_glove_v2.keras"
TOKENIZER_PATH = "dashboard/tokenizer.pkl"
model = load_model(MODEL_PATH)
tokenizer = load_tokenizer(TOKENIZER_PATH)

# ======================
# SIDEBAR INFO
# ======================
st.sidebar.title("📊 Model Info")

st.sidebar.write("**Model:** HBGRU v2")
st.sidebar.write("**Embedding:** GloVe 100D")
st.sidebar.write("**Architecture:** Bidirectional GRU + Hybrid layer")

st.sidebar.divider()

st.sidebar.metric("Validation Accuracy", "88.06%")
st.sidebar.metric("Training Accuracy", "91.22%")

# ======================
# TITLE
# ======================
st.title("🧠 Sentiment Analysis Dashboard")
st.caption("HBGRU v2 + GloVe Embeddings")

# ======================
# INPUT SECTION
# ======================
review = st.text_area("Enter a product review OR multiple reviews (one per line):", height=200)

use_batch = st.checkbox("Batch mode (multiple reviews)")
col1, col2 = st.columns([1, 1])

# ======================
# PREDICTION FUNCTION
# ======================
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    
    seq = tokenizer.texts_to_sequences([processed_text])

    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    pred = model.predict(padded, verbose=0)[0][0]

    POSI_THRESHOLD = 0.7
    NEGA_THRESHOLD = 0.4

    if pred >= POSI_THRESHOLD:
        sentiment = "Positive"
        confidence = pred

    elif pred <= NEGA_THRESHOLD:
        sentiment = "Negative"
        confidence = 1 - pred

    else:
        sentiment = "Neutral / Mixed"
        confidence = max(pred, 1 - pred)

    positive_prob = pred
    negative_prob = 1 - pred

    # st.write(f"Positive probability: {positive_prob * 100:.2f}%")
    # st.write(f"Negative probability: {negative_prob * 100:.2f}%")

    return sentiment, confidence, pred

# ======================
# N-GRAM FUNCTION
# ======================
def extract_ngrams(texts, n=2, top_k=10):
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words="english")
    X = vectorizer.fit_transform(texts)

    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    sorted_words = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return sorted_words[:top_k]

# ======================
# SESSION STORAGE
# ======================
if "reviews" not in st.session_state:
    st.session_state.reviews = []

# ======================
# PREDICT BUTTON
# ======================
# if st.button("Analyze Review") and review.strip() != "":
if st.button("Analyze Review(s)") and review.strip() != "":

    reviews_list = [r.strip() for r in review.split("\n") if r.strip()]

    results = []

    for review in reviews_list:

        sentiment, confidence, raw_score = predict_sentiment(review)

        results.append({
            "review": review,
            "sentiment": sentiment,
            "raw_score": raw_score,
            "confidence": confidence
        })

        # st.write(f"Raw Score: {raw_score:.4f}")

        st.session_state.reviews.append({
            "review": review,
            "sentiment": sentiment
        })

    df_results = pd.DataFrame(results)
    st.subheader("📊 Batch Results")
    st.dataframe(df_results)
    st.success("Batch analysis complete!")

    # st.success(f"Prediction: {sentiment}")
    # st.metric("Confidence", f"{confidence * 100:.2f}%")

# ======================
# BATCH ANALYSIS
# ======================
st.divider()
st.subheader("📁 Batch Analysis via CSV Upload")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.dataframe(df_uploaded.head())

    # Let user choose text column
    text_column = st.selectbox(
        "Select the column that contains review text:",
        df_uploaded.columns
    )

    if st.button("Run Batch Sentiment Analysis"):

        results = []

        for text in df_uploaded[text_column].fillna("").astype(str):

            sentiment, confidence, raw_score = predict_sentiment(text)

            results.append({
                "review": text,
                "sentiment": sentiment,
                "raw_score": raw_score,
                "confidence": confidence
            })

            # store in session for analytics
            st.session_state.reviews.append({
                "review": text,
                "sentiment": sentiment
            })

        result_df = pd.DataFrame(results)

        st.subheader("📊 Batch Results")
        st.dataframe(result_df)

        st.success("Batch analysis completed!")

# ======================
# ANALYTICS SECTION
# ======================
if len(st.session_state.reviews) > 0:

    st.divider()
    st.subheader("📈 Session Analytics")

    df = pd.DataFrame(st.session_state.reviews)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Reviews", len(df))
    col2.metric("Positive", len(df[df["sentiment"] == "Positive"]))
    col3.metric("Negative", len(df[df["sentiment"] == "Negative"]))

    st.divider()

    # ======================
    # PIE CHART DATA
    # ======================
    st.subheader("📊 Sentiment Distribution")

    chart_data = df["sentiment"].value_counts().reset_index()
    chart_data.columns = ["sentiment", "count"]

    fig = px.pie(
        chart_data,
        names="sentiment",
        values="count",
        title="Sentiment Distribution (Session Data)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # N-GRAM INSIGHTS
    # ======================
    st.subheader("🔑 Key Insights (N-grams)")

    all_reviews = df["review"].tolist()

    if len(all_reviews) > 1:

        st.write("### Top Bigrams")
        bigrams = extract_ngrams(all_reviews, n=2)
        st.table(pd.DataFrame(bigrams, columns=["Bigram", "Frequency"]))

        st.write("### Top Trigrams")
        trigrams = extract_ngrams(all_reviews, n=3)
        st.table(pd.DataFrame(trigrams, columns=["Trigram", "Frequency"]))

    else:
        st.info("Add more reviews to generate insights.")

# ======================
# RECENT REVIEWS
# ======================
st.divider()
st.subheader("📝 Recent Reviews")

if len(st.session_state.reviews) > 0:
    for r in reversed(st.session_state.reviews[-5:]):
        st.write(f"**{r['sentiment']}** - {r['review']}")
else:
    st.info("No reviews yet.")