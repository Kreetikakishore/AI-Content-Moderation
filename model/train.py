import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

df = pd.read_csv("data/train.csv")

# Combine all toxicity labels into a severity score (0–6)
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df["severity"] = df[label_cols].sum(axis=1)
df["label"] = (df["severity"] > 0).astype(int)  # binary for now

X_train, X_test, y_train, y_test = train_test_split(
    df["comment_text"], df["label"], test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, C=5))
])

pipeline.fit(X_train, y_train)

print(classification_report(y_test, pipeline.predict(X_test)))

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/moderation_model.pkl")
print("✅ Model saved.")