"""
Minimal Classification Pipeline with scikit-learn
Using the Breast Cancer Wisconsin dataset
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. LOAD DATA
# =============================================================================
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {dict(zip(data.target_names, np.bincount(y)))}")

# =============================================================================
# 2. TRAIN-TEST SPLIT (before any preprocessing)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 3. DEFINE PIPELINES
# =============================================================================
pipelines = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ]),
}

# =============================================================================
# 4. CROSS-VALIDATION
# =============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nCross-validation (F1 score):")
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    print(f"  {name}: {scores.mean():.3f} ± {scores.std():.3f}")

# =============================================================================
# 5. HYPERPARAMETER TUNING
# =============================================================================
param_grid = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [None, 10, 20],
}

grid_search = GridSearchCV(
    pipelines["Random Forest"],
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_

# =============================================================================
# 6. FINAL EVALUATION ON TEST SET
# =============================================================================
y_pred = best_model.predict(X_test)

print("\nTest Set Results:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# =============================================================================
# 7. VISUALISATION
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay.from_estimator(
    best_model, X_test, y_test, display_labels=data.target_names, cmap="Blues", ax=axes[0]
)
axes[0].set_title("Confusion Matrix")

RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=axes[1])
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("ROC Curve")

plt.tight_layout()
plt.savefig("classification_results.png", dpi=150)
print("\nSaved: classification_results.png")