"""
Comprehensive Classification Pipeline with scikit-learn
========================================================

This script demonstrates a complete classification workflow using the
Wisconsin Breast Cancer dataset. Each stage is explained in detail to
build intuition for why each step matters.

Dataset: Wisconsin Breast Cancer (569 samples, 30 features)
Task: Binary classification (malignant vs benign tumours)
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

def load_and_explore_data():
    """
    Load the breast cancer dataset and perform initial exploration.

    The Wisconsin Breast Cancer dataset contains measurements from digitised
    images of fine needle aspirates (FNA) of breast masses. Features describe
    characteristics of cell nuclei present in the image.

    Returns
    -------
    X : ndarray of shape (569, 30)
        Feature matrix with 30 numeric features
    y : ndarray of shape (569,)
        Target labels (0 = malignant, 1 = benign)
    feature_names : ndarray
        Names of the 30 features
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    print(f"\nDataset shape: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target classes: {data.target_names}")

    # Class distribution - crucial for choosing metrics and handling imbalance
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for label, count in zip(unique, counts):
        pct = count / len(y) * 100
        print(f"  {data.target_names[label]}: {count} ({pct:.1f}%)")

    # Feature statistics - helps identify scaling needs
    df = pd.DataFrame(X, columns=feature_names)
    print(f"\nFeature statistics (first 5 features):")
    print(df.iloc[:, :5].describe().round(2))

    # Check for missing values
    missing = np.isnan(X).sum()
    print(f"\nMissing values: {missing}")

    # Feature scale differences - this is why we need StandardScaler
    print(f"\nFeature scale ranges (showing need for scaling):")
    print(f"  'mean radius' range: {X[:, 0].min():.1f} - {X[:, 0].max():.1f}")
    print(f"  'mean area' range: {X[:, 3].min():.1f} - {X[:, 3].max():.1f}")
    print(f"  'mean smoothness' range: {X[:, 4].min():.4f} - {X[:, 4].max():.4f}")

    return X, y, feature_names


# =============================================================================
# STAGE 3: TRAIN-TEST SPLIT
# =============================================================================
# We split data BEFORE any preprocessing to prevent data leakage. This is
# critical: if we fit our scaler on all data, information from the test set
# "leaks" into our training process, giving optimistically biased results.
#
# Key parameters:
# - test_size: 20% held out for final evaluation (common choice: 15-30%)
# - stratify: Ensures class proportions are preserved in both splits
# - random_state: Makes results reproducible


def split_data(X, y):
    """
    Split data into training and test sets with stratification.

    Stratification is essential for classification to ensure both splits
    have the same class distribution as the original data. Without it,
    you might accidentally get all of one class in training and none in test.

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target labels

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarrays
        Split datasets
    """
    print("\n" + "=" * 60)
    print("STAGE 3: TRAIN-TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # Preserve class proportions
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Verify stratification worked
    train_pct = y_train.mean() * 100
    test_pct = y_test.mean() * 100
    print(f"\nClass 1 (benign) proportion:")
    print(f"  Training: {train_pct:.1f}%")
    print(f"  Test: {test_pct:.1f}%")
    print("  (These should be nearly identical if stratification worked)")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STAGE 4: PREPROCESSING WITH PIPELINES
# =============================================================================
# Pipelines chain preprocessing and modelling steps together. This is superior
# to manual preprocessing because:
#
# 1. Prevents data leakage: fit_transform only called on training data
# 2. Simplifies cross-validation: the pipeline handles fitting at each fold
# 3. Easier deployment: one object contains the entire workflow
# 4. Reproducibility: all steps are documented in code
#
# StandardScaler transforms features to zero mean and unit variance:
#   z = (x - mean) / std
#
# This is essential for:
# - SVM: Distance-based, sensitive to feature scales
# - KNN: Distance-based, larger features dominate
# - Logistic Regression: Gradient descent converges faster with scaled features
# - Neural Networks: Same reason as logistic regression
#
# NOT needed for tree-based models (Random Forest, Gradient Boosting) as they
# use feature thresholds and are scale-invariant.


def create_pipelines():
    """
    Create preprocessing + model pipelines for different classifiers.

    Each pipeline includes:
    1. StandardScaler: Normalise features to zero mean, unit variance
    2. Classifier: The actual model

    We create multiple pipelines to compare different algorithms.

    Returns
    -------
    pipelines : dict
        Dictionary mapping model names to Pipeline objects
    """
    print("\n" + "=" * 60)
    print("STAGE 4: CREATING PIPELINES")
    print("=" * 60)

    pipelines = {
        # Logistic Regression: Good baseline, interpretable coefficients
        # max_iter increased because default (100) may not converge
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        # Random Forest: Ensemble of decision trees, robust to overfitting
        # n_jobs=-1 uses all CPU cores for parallel training
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),  # Not strictly needed, but harmless
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                ),
            ]
        ),
        # SVM with RBF kernel: Excellent for non-linear boundaries
        # probability=True enables predict_proba for ROC curves
        "SVM (RBF)": Pipeline(
            [
                ("scaler", StandardScaler()),  # Critical for SVM
                ("classifier", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        ),
        # KNN: Instance-based learning, simple but effective
        # n_neighbors=5 is a common default; tune via cross-validation
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),  # Critical for distance-based
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        # Naive Bayes: Fast, works well with limited data
        # Assumes feature independence (often violated but still works)
        "Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB()),
            ]
        ),
        # Gradient Boosting: Sequential ensemble, often best accuracy
        # Slower to train than Random Forest but often more accurate
        "Gradient Boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    GradientBoostingClassifier(n_estimators=100, random_state=42),
                ),
            ]
        ),
    }

    print("\nCreated pipelines for:")
    for name in pipelines:
        print(f"  - {name}")

    print("\nPipeline structure example (Logistic Regression):")
    print(pipelines["Logistic Regression"])

    return pipelines


# =============================================================================
# STAGE 5: CROSS-VALIDATION
# =============================================================================
# Cross-validation gives a more robust estimate of model performance than a
# single train-test split. It works by:
#
# 1. Split training data into K folds (typically 5 or 10)
# 2. For each fold: train on K-1 folds, validate on the remaining fold
# 3. Average the K validation scores
#
# StratifiedKFold ensures each fold has the same class distribution.
#
# Why cross-validation matters:
# - Single splits can be lucky or unlucky
# - CV gives mean AND standard deviation of performance
# - Helps detect overfitting (high variance across folds)
#
# Scoring metrics:
# - accuracy: Overall correctness (misleading for imbalanced data)
# - f1: Harmonic mean of precision and recall (better for imbalanced)
# - roc_auc: Area under ROC curve (measures ranking quality)


def cross_validate_models(pipelines, X_train, y_train):
    """
    Evaluate all pipelines using stratified k-fold cross-validation.

    Cross-validation prevents overfitting to a single train-test split
    and gives confidence intervals on performance estimates.

    Parameters
    ----------
    pipelines : dict
        Dictionary of model pipelines
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels

    Returns
    -------
    results : dict
        Cross-validation scores for each model
    """
    print("\n" + "=" * 60)
    print("STAGE 5: CROSS-VALIDATION")
    print("=" * 60)

    # Use stratified k-fold to preserve class distribution in each fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    print("\nCross-validation results (5-fold, F1 score):")
    print("-" * 50)

    for name, pipeline in pipelines.items():
        # F1 score balances precision and recall - good for slight imbalance
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )

        results[name] = {
            "mean": scores.mean(),
            "std": scores.std(),
            "scores": scores,
        }

        # ± shows standard deviation - lower is more consistent
        print(f"{name:25s}: {scores.mean():.3f} ± {scores.std():.3f}")

    print("-" * 50)
    print("\nInterpretation:")
    print("  - Higher mean = better average performance")
    print("  - Lower std = more consistent across different data splits")

    return results


# =============================================================================
# STAGE 6: HYPERPARAMETER TUNING
# =============================================================================
# Most models have hyperparameters that control their behaviour:
# - Random Forest: n_estimators, max_depth, min_samples_split
# - SVM: C (regularisation), gamma (kernel width)
# - KNN: n_neighbors, weights
#
# GridSearchCV exhaustively searches all combinations. For larger spaces,
# use RandomizedSearchCV which samples random combinations.
#
# Important: GridSearchCV uses cross-validation internally, so we're doing
# "nested cross-validation" - CV for hyperparameter selection inside CV for
# model evaluation. This prevents overfitting to the validation set.


def tune_best_model(X_train, y_train):
    """
    Perform hyperparameter tuning on the most promising model.

    We'll tune Random Forest as it typically performs well and has
    interpretable hyperparameters.

    Parameters
    ----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels

    Returns
    -------
    best_pipeline : Pipeline
        Tuned pipeline with optimal hyperparameters
    """
    print("\n" + "=" * 60)
    print("STAGE 6: HYPERPARAMETER TUNING")
    print("=" * 60)

    # Define the pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )

    # Parameter grid - use classifier__ prefix to target the classifier step
    # This is pipeline syntax: stepname__parametername
    param_grid = {
        "classifier__n_estimators": [50, 100, 200],  # Number of trees
        "classifier__max_depth": [None, 10, 20],  # Tree depth (None = unlimited)
        "classifier__min_samples_split": [2, 5, 10],  # Min samples to split node
    }

    print("\nSearching over parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    print(f"\nTotal combinations: {3 * 3 * 3} = 27")
    print("Each evaluated with 5-fold CV = 135 model fits")

    # GridSearchCV with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        return_train_score=True,  # Helps diagnose over/underfitting
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nBest cross-validation F1 score: {grid_search.best_score_:.3f}")

    # Check for overfitting: large gap between train and test scores is bad
    results_df = pd.DataFrame(grid_search.cv_results_)
    best_idx = grid_search.best_index_
    train_score = results_df.loc[best_idx, "mean_train_score"]
    test_score = results_df.loc[best_idx, "mean_test_score"]

    print(f"\nOverfitting check:")
    print(f"  Mean train score: {train_score:.3f}")
    print(f"  Mean test score:  {test_score:.3f}")
    print(f"  Gap: {train_score - test_score:.3f} (smaller is better)")

    return grid_search.best_estimator_


# =============================================================================
# STAGE 7: FEATURE SELECTION (OPTIONAL)
# =============================================================================
# Feature selection can:
# - Reduce overfitting by removing noisy features
# - Speed up training and inference
# - Improve interpretability
#
# Common approaches:
# - Filter methods: Score features independently (fast but ignores interactions)
# - Wrapper methods: Use model performance to select (slow but thorough)
# - Embedded methods: Feature selection built into model (e.g., L1 regularisation)
#
# SelectKBest with f_classif uses ANOVA F-statistic to score features.
# Higher scores mean the feature better separates the classes.


def demonstrate_feature_selection(X_train, y_train, feature_names):
    """
    Demonstrate feature selection and show most important features.

    Parameters
    ----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    feature_names : ndarray
        Names of features
    """
    print("\n" + "=" * 60)
    print("STAGE 7: FEATURE SELECTION")
    print("=" * 60)

    # Score all features using ANOVA F-statistic
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X_train, y_train)

    # Create DataFrame of feature scores
    feature_scores = pd.DataFrame(
        {
            "feature": feature_names,
            "f_score": selector.scores_,
            "p_value": selector.pvalues_,
        }
    ).sort_values("f_score", ascending=False)

    print("\nTop 10 features by ANOVA F-score:")
    print("(Higher F-score = better class separation)")
    print("-" * 50)
    for _, row in feature_scores.head(10).iterrows():
        print(f"  {row['feature']:25s}: F={row['f_score']:8.1f}, p={row['p_value']:.2e}")

    print("\nBottom 5 features:")
    for _, row in feature_scores.tail(5).iterrows():
        print(f"  {row['feature']:25s}: F={row['f_score']:8.1f}, p={row['p_value']:.2e}")

    # Compare model with all features vs selected features
    print("\nComparing full vs reduced feature sets:")

    pipeline_full = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    pipeline_reduced = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=10)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores_full = cross_val_score(pipeline_full, X_train, y_train, cv=cv, scoring="f1")
    scores_reduced = cross_val_score(
        pipeline_reduced, X_train, y_train, cv=cv, scoring="f1"
    )

    print(f"  All 30 features: {scores_full.mean():.3f} ± {scores_full.std():.3f}")
    print(f"  Top 10 features: {scores_reduced.mean():.3f} ± {scores_reduced.std():.3f}")


# =============================================================================
# STAGE 8: FINAL EVALUATION
# =============================================================================
# After all experimentation, we evaluate our chosen model on the held-out
# test set ONCE. This gives an unbiased estimate of real-world performance.
#
# Key metrics for classification:
#
# Precision: Of predicted positives, how many are correct?
#   precision = TP / (TP + FP)
#   High precision = few false alarms
#
# Recall (Sensitivity): Of actual positives, how many did we catch?
#   recall = TP / (TP + FN)
#   High recall = few missed cases
#
# F1 Score: Harmonic mean of precision and recall
#   f1 = 2 * (precision * recall) / (precision + recall)
#   Balances both concerns
#
# For medical diagnosis (like cancer detection):
# - High recall is crucial: we don't want to miss cancer cases
# - Some false positives are acceptable (further testing can confirm)


def final_evaluation(best_pipeline, X_train, y_train, X_test, y_test):
    """
    Perform final evaluation on the held-out test set.

    This should only be done ONCE after all model selection and tuning
    is complete. Running this multiple times and selecting based on
    test performance would invalidate the results.

    Parameters
    ----------
    best_pipeline : Pipeline
        The tuned model pipeline
    X_train, y_train : ndarrays
        Training data (for fitting)
    X_test, y_test : ndarrays
        Test data (for evaluation)
    """
    print("\n" + "=" * 60)
    print("STAGE 8: FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    # Fit on full training data
    best_pipeline.fit(X_train, y_train)

    # Predict on test set
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

    # ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.3f}")

    print("\nInterpretation for this cancer detection task:")
    print("  - Recall for Malignant: How many cancer cases we correctly identified")
    print("  - Precision for Malignant: Of predicted cancers, how many were real")
    print("  - In medical contexts, high recall is often prioritised")

    return y_pred, y_proba


# =============================================================================
# STAGE 9: VISUALISATION
# =============================================================================
# Visualisations help communicate results and diagnose issues:
#
# Confusion Matrix: Shows the 4 outcomes (TP, TN, FP, FN)
# - Diagonal = correct predictions
# - Off-diagonal = errors
#
# ROC Curve: Trade-off between true positive rate and false positive rate
# - Area Under Curve (AUC) summarises overall ranking quality
# - AUC = 1.0 is perfect, 0.5 is random guessing
#
# Learning Curve: Performance vs training set size
# - Diagnoses whether more data would help
# - Gap between train and validation indicates overfitting


def create_visualisations(best_pipeline, X_train, y_train, X_test, y_test):
    """
    Create diagnostic visualisations.

    Parameters
    ----------
    best_pipeline : Pipeline
        Fitted model pipeline
    X_train, y_train : ndarrays
        Training data
    X_test, y_test : ndarrays
        Test data
    """
    print("\n" + "=" * 60)
    print("STAGE 9: VISUALISATIONS")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(
        best_pipeline,
        X_test,
        y_test,
        display_labels=["Malignant", "Benign"],
        cmap="Blues",
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")

    # 2. ROC Curve
    RocCurveDisplay.from_estimator(
        best_pipeline,
        X_test,
        y_test,
        ax=axes[1],
    )
    axes[1].plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    # 3. Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(
        best_pipeline,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )

    axes[2].plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training")
    axes[2].fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.1,
    )

    axes[2].plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation")
    axes[2].fill_between(
        train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.1,
    )

    axes[2].set_xlabel("Training Set Size")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Learning Curve")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("classification_diagnostics.png", dpi=150, bbox_inches="tight")
    print("\nSaved visualisation to 'classification_diagnostics.png'")

    print("\nLearning curve interpretation:")
    print("  - If curves converge at high score: Good fit, more data won't help much")
    print("  - If gap remains: Model is overfitting, try regularisation")
    print("  - If both curves are low: Model is underfitting, try more complexity")


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """
    Execute the complete classification pipeline.
    """
    print("\n" + "=" * 60)
    print("SCIKIT-LEARN CLASSIFICATION PIPELINE")
    print("Breast Cancer Wisconsin Dataset")
    print("=" * 60)

    # Stage 2: Load and explore data
    X, y, feature_names = load_and_explore_data()

    # Stage 3: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Stage 4: Create pipelines
    pipelines = create_pipelines()

    # Stage 5: Cross-validate all models
    cv_results = cross_validate_models(pipelines, X_train, y_train)

    # Stage 6: Tune the best model
    best_pipeline = tune_best_model(X_train, y_train)

    # Stage 7: Feature selection analysis
    demonstrate_feature_selection(X_train, y_train, feature_names)

    # Stage 8: Final evaluation
    y_pred, y_proba = final_evaluation(
        best_pipeline, X_train, y_train, X_test, y_test
    )

    # Stage 9: Visualisations
    create_visualisations(best_pipeline, X_train, y_train, X_test, y_test)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print("\nKey takeaways:")
    print("  1. Always split data BEFORE preprocessing to prevent leakage")
    print("  2. Use pipelines to chain preprocessing and modelling")
    print("  3. Cross-validation gives robust performance estimates")
    print("  4. Tune hyperparameters systematically with GridSearchCV")
    print("  5. Evaluate on test set only ONCE at the very end")
    print("  6. Choose metrics appropriate to your problem domain")


if __name__ == "__main__":
    main()