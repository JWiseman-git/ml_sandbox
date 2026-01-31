from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

# goal is to predict target label where 0 is malignant and 1 is benign

data = load_breast_cancer()

X = data.data
y = data.target

x_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)

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
    return Pipeline([
        ('scaler', StandardScaler()),
    ])

