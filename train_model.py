import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load dataset
df = pd.read_csv("IMLP6_TASK_03-products.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Keep required columns
df = df.dropna(subset=["Product Title", "Category Label"])

df["Product Title"] = (
    df["Product Title"]
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df["Category Label"] = df["Category Label"].astype(str).str.strip()

df = df[
    (df["Product Title"] != "") &
    (df["Category Label"] != "")
].drop_duplicates()

# Features and target
X = df["Product Title"]
y = df["Category Label"]

# Convert text into numerical features
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "category_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
