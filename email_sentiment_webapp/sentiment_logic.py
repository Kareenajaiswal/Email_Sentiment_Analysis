# sentiment_logic.py

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk import download

# Download stopwords once
download('stopwords')
stop_words = set(stopwords.words('english'))

# Sentiment scoring dictionary
word_weights = {
    "good": 1, "great": 2, "fantastic": 3, "excellent": 3, "well": 1, "love": 2,
    "helpful": 1, "amazing": 3, "wonderful": 2, "nice": 1, "satisfied": 2,
    "bad": -1, "terrible": -3, "problem": -2, "unhelpful": -2, "late": -1,
    "delay": -1, "not happy": -2, "frustrated": -2, "disappointed": -3,
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join(word for word in text.split() if word not in stop_words)

def analyze_sentiment(text):
    cleaned = clean_text(text)
    score = sum(word_weights.get(word, 0) for word in cleaned.split())
    if score > 1:
        return "positive"
    elif score < -1:
        return "negative"
    else:
        return "neutral"
