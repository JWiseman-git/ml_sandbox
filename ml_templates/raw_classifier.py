from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# goal is to predict target label where 0 is malignant and 1 is benign

def load_data_prepare_splits(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into test and training data"""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test

def build_pipeline():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    classifier =  LogisticRegression(random_state=42)

    return Pipeline(
        steps=[
            ('preprocessor', numeric_transformer),
            ('classifier', classifier),
        ]
    )

def train_model(pipeline: Pipeline, X_train, y_train):
    """Train model with cross-validation and return metrics."""
    logger.info("Starting cross-validation with %d folds", 10)

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=10, scoring="accuracy"
    )

    logger.info("CV Accuracy: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std() * 2)

    logger.info("Fitting final model on full training set")
    pipeline.fit(X_train, y_train)

    metrics = {
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "train_samples": len(X_train),
    }

    return pipeline, metrics

def evaluate_model(pipeline: Pipeline, X_test, y_test) -> dict:
    """Evaluate model on held-out test set."""
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Malignant", "Benign"]
    )
    plt.show()

    return {
        "accuracy": report["accuracy"],
        "classification_report": report,
    }

def main():
    data = load_breast_cancer()

    X = data.data
    y = data.target

    x_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)

    X_train, X_test, y_train, y_test = load_data_prepare_splits(x_df, y_df, test_size=0.2, random_state=42)
    pipeline = build_pipeline()
    trained_pipeline, metrics = train_model(
        pipeline, X_train, y_train,)
    test_metrics = evaluate_model(trained_pipeline, X_test, y_test)

if __name__ == "__main__":
    main()