import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")

    print("\n=== DATA EXPLORATION ===")
    print("\nFirst 5 rows:")
    print(X.head())

    print("\nDataset shape:")
    print(X.shape)

    print("\nSummary statistics:")
    print(X.describe())

    print("\nClass distribution:")
    print(y.value_counts())

    print("\nDiscussion:")
    print("- The dataset is slightly imbalanced but acceptable for ML.")
    print("- Some features (e.g., proline, color_intensity) show large variance → possible outliers.")

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline_model = LogisticRegression(max_iter=5000)
    baseline_model.fit(X_train, y_train)

    y_pred = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred)

    print("\n=== BASELINE MODEL (Train-Test Split) ===")
    print(f"Baseline Test Accuracy: {baseline_accuracy:.4f}")

    print("\n=== K-FOLD CROSS VALIDATION (k=5) ===")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lr = LogisticRegression(max_iter=5000)

    kfold_scores = cross_val_score(lr, X, y, cv=kf, scoring='accuracy')

    print("Fold scores:", kfold_scores)
    print(f"Mean CV Accuracy: {kfold_scores.mean():.4f}")
    print(f"Std CV Accuracy: {kfold_scores.std():.4f}")

    print("\nInterpretation:")
    print("- CV mean is more reliable because it uses multiple train/test splits.")
    print("- Std shows stability: lower std = more consistent performance.")
   
    print("\n=== STRATIFIED K-FOLD CV (k=5) ===")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestClassifier(random_state=42)

    lr_scores = cross_val_score(lr, X, y, cv=skf, scoring='accuracy')
    rf_scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')

    print("Logistic Regression:")
    print(f"  Mean Accuracy: {lr_scores.mean():.4f}")
    print(f"  Std Accuracy: {lr_scores.std():.4f}")

    print("Random Forest:")
    print(f"  Mean Accuracy: {rf_scores.mean():.4f}")
    print(f"  Std Accuracy: {rf_scores.std():.4f}")

    print("\nAnalysis:")
    print("- Random Forest is usually more accurate.")
    print("- Logistic Regression often has lower variance (more stable).")
    print("- Increased accuracy may justify Random Forest complexity.")

    
    print("\n=== MULTIPLE METRICS (Stratified CV) ===")

    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    print("\nLogistic Regression metrics:")
    for metric in scoring:
        scores = cross_val_score(lr, X, y, cv=skf, scoring=metric)
        print(f"{metric}: mean={scores.mean():.4f}, std={scores.std():.4f}")

    print("\nRandom Forest metrics:")
    for metric in scoring:
        scores = cross_val_score(rf, X, y, cv=skf, scoring=metric)
        print(f"{metric}: mean={scores.mean():.4f}, std={scores.std():.4f}")

    print("\nMetric Analysis:")
    print("- Random Forest usually dominates on recall and F1.")
    print("- Accuracy differences are smaller than recall/F1 differences.")
    print("\n=== GRID SEARCH CV (Random Forest) ===")

    params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 3, 5, 10],
        'min_samples_split': [2, 4, 6]
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=params,
        scoring='accuracy',
        cv=skf,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best CV Accuracy:", grid.best_score_)
    print("Best Parameters:", grid.best_params_)
    print("Best Estimator:", grid.best_estimator_)

    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)

    test_pred = best_model.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, test_pred)

    print("\n=== FINAL COMPARISON ===")
    print(f"Baseline Accuracy (Logistic Regression): {baseline_accuracy:.4f}")
    print(f"Tuned Random Forest Accuracy: {tuned_accuracy:.4f}")

    print("\nConclusion:")
    print("- Cross Validation provides a robust estimate of performance.")
    print("- Hyperparameter tuning improves generalization.")
    print("- Stratified CV is preferred for multiclass problems.")


if __name__ == "__main__":
    main()
