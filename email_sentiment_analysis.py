# import pandas as pd
# import matplotlib.pyplot as plt
# import re
# from nltk.corpus import stopwords
# from nltk import download

# # Download stopwords from NLTK
# download('stopwords')
# stop_words = set(stopwords.words('english'))

# # Load your dataset (update the path if needed)
# df = pd.read_csv("email_datasets.csv")

# # Weighted sentiment dictionary
# word_weights = {
#     # Positive words
#     "good": 1, "great": 2, "fantastic": 3, "excellent": 3, "well": 1, "love": 2,
#     "helpful": 1, "impressed": 2, "quick": 1, "amazing": 3, "wonderful": 2,
#     "nice": 1, "satisfied": 2, "pleasant": 1, "friendly": 1,

#     # Negative words
#     "bad": -1, "terrible": -3, "disappointed": -2, "poor": -2, "problem": -1,
#     "late": -1, "crash": -2, "unhelpful": -2, "rude": -2, "issue": -1, "damaged": -2,
#     "horrible": -3, "slow": -1, "unacceptable": -3
# }

# # Common negation words
# negation_words = {"not", "never", "no", "didn't", "isn't", "wasn't", "aren't", "don't", "doesn't", "won't", "can't"}

# # Smarter sentiment scoring function
# def smart_sentiment_score(text):
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     words = text.split()
#     words = [w for w in words if w not in stop_words]

#     score = 0
#     negate = False
#     for word in words:
#         if word in negation_words:
#             negate = True
#             continue
#         weight = word_weights.get(word, 0)
#         if negate:
#             weight *= -1
#             negate = False
#         score += weight

#     return score / (len(words) + 1)  # Normalize score

# # Apply sentiment scoring
# df["sentiment_score"] = df["email_body"].apply(smart_sentiment_score)

# # Classify sentiment based on score
# def classify_sentiment(score):
#     if score > 0.3:
#         return "positive" 
#     elif score < -0.3:
#         return "negative"  # -0.3 > Neu < 0.3
#     else:
#         return "neutral"

# df["sentiment"] = df["sentiment_score"].apply(classify_sentiment)

# # Visualization
# sentiment_counts = df["sentiment"].value_counts()

# plt.figure(figsize=(8, 5))
# sentiment_counts.plot(kind='bar', color=['green', 'red', 'orange'])
# plt.title("Smarter Sentiment Distribution (Rule-Based)")
# plt.xlabel("Sentiment")
# plt.ylabel("Number of Emails")
# plt.xticks(rotation=0)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# # Export to CSV (with score + sentiment)
# df[["sender_email", "email_subject", "email_body", "sentiment_score", "sentiment"]].to_csv("email_datasets_with_sentiments.csv", index=False)
# print("✅ Sentiment-tagged data (with scores) saved to email_datasets_with_sentiments.csv")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load and clean CSV
df = pd.read_csv("labeled_email_data.csv")
df.columns = df.columns.str.strip()  # remove leading/trailing spaces

# Show sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Sentiment', data=df, palette='viridis')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Feature extraction
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["Body"])
y = df["Sentiment"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to test
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

accuracies = {}
conf_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    conf = confusion_matrix(y_test, predictions, labels=["positive", "neutral", "negative"])

    accuracies[name] = acc
    conf_matrices[name] = conf

# Plot model accuracies
plt.figure(figsize=(8,5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="coolwarm")
plt.ylim(0,1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot confusion matrices
for name, conf in conf_matrices.items():
    plt.figure(figsize=(6,4))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", xticklabels=["Positive", "Neutral", "Negative"], yticklabels=["Positive", "Neutral", "Negative"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
