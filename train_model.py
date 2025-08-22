import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load dataset
data = pd.read_csv("resume.csv")

# Basic check
if data.empty or 'text' not in data.columns or 'label' not in data.columns:
    raise Exception("Dataset must have 'text' and 'label' columns and not be empty.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', LogisticRegression(solver='liblinear')),
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=1))

# Save model pipeline (vectorizer + model)
joblib.dump(pipeline, "resume_sentiment_model.pkl")
print("✅ Model trained and saved as resume_sentiment_model.pkl")
