import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample data
data = pd.DataFrame({
    "text": [
        "I love this product",
        "This is awful",
        "Really enjoyed it",
        "Worst experience ever",
        "Absolutely fantastic",
        "Not worth the money"
    ],
    "label": [1, 0, 1, 0, 1, 0]
})

X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.3, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix  = vectorizer.fit_transform(data["text"])
feature_names = vectorizer.get_feature_names_out()

# Optional: Convert to a DataFrame for better visualization
# df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print(classification_report(y_test, preds))
