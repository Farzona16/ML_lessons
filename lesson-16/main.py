from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification


RANDOM_STATE = 42
DEFAULT_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_telco_or_simulate(csv_path: str = DEFAULT_CSV) -> tuple[pd.DataFrame, bool]:
    """
    Returns: (df, is_telco_real)
    If CSV exists -> loads real Telco dataset
    Else -> creates a simulated dataset (numeric-only + a few fake categorical columns)
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df, True
    X, y = make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.73, 0.27], 
        random_state=RANDOM_STATE,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    df["Churn"] = np.where(y == 1, "Yes", "No")

    rng = np.random.default_rng(RANDOM_STATE)
    df["Contract"] = rng.choice(["Month-to-month", "One year", "Two year"], size=len(df), p=[0.6, 0.2, 0.2])
    df["InternetService"] = rng.choice(["DSL", "Fiber optic", "No"], size=len(df), p=[0.35, 0.5, 0.15])
    df["SeniorCitizen"] = rng.choice([0, 1], size=len(df), p=[0.84, 0.16])
    return df, False


def print_head_and_schema(df: pd.DataFrame, is_real: bool) -> None:
    print("\n" + "=" * 90)
    print("1) LOAD DATASET")
    print("=" * 90)
    print("Using dataset:", "REAL Telco CSV" if is_real else "SIMULATED fallback dataset")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nShape:", df.shape)
    print("\nColumns and dtypes:")
    print(df.dtypes)

def explore_target_distribution(df: pd.DataFrame, target_col: str = "Churn") -> None:
    print("\n" + "=" * 90)
    print("2) DATA EXPLORATION")
    print("=" * 90)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    counts = df[target_col].astype(str).value_counts(dropna=False)
    print("\nTarget distribution:")
    print(counts)

    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values) #type:ignore
    plt.title("Churn distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def clean_telco_totalcharges(df: pd.DataFrame) -> pd.DataFrame:
    """
    For Telco dataset: TotalCharges is sometimes ' ' (blank) -> convert to numeric NaN.
    """
    if "TotalCharges" in df.columns:
        df = df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def quick_numeric_correlations(df: pd.DataFrame, target_col: str = "Churn") -> None:
    """
    Show correlations among numeric features and optionally with churn (if we can map churn to 0/1).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("\nNo numeric columns found for correlation.")
        return

    print("\nNumeric columns:", numeric_cols)
    corr = df[numeric_cols].corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(corr.values, aspect="auto")
    plt.title("Correlation matrix (numeric features)")
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    if target_col in df.columns:
        churn_map = df[target_col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        if churn_map.notna().all():
            tmp = df[numeric_cols].copy()
            tmp["_churn_bin"] = churn_map.values
            corr_to_churn = tmp.corr(numeric_only=True)["_churn_bin"].drop("_churn_bin").sort_values(key=np.abs, ascending=False)
            print("\nCorrelation to churn (binary) (sorted by absolute correlation):")
            print(corr_to_churn)


def churn_rate_by_category(df: pd.DataFrame, col: str, target_col: str = "Churn") -> None:
    """
    For a categorical column, print churn rates.
    """
    if col not in df.columns or target_col not in df.columns:
        return

    churn_bin = df[target_col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if churn_bin.isna().any():
        return

    rates = df.assign(_churn=churn_bin).groupby(col)["_churn"].mean().sort_values(ascending=False)
    print(f"\nChurn rate by {col}:")
    print(rates)

    # Plot
    plt.figure(figsize=(7, 4))
    plt.bar(rates.index.astype(str), rates.values) #type: ignore
    plt.title(f"Churn rate by {col}")
    plt.xlabel(col)
    plt.ylabel("Churn rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

def prepare_features_and_target(df: pd.DataFrame, target_col: str = "Churn") -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y = df[target_col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        vc = df[target_col].astype(str).value_counts()
        majority = vc.index[0]
        y = (df[target_col].astype(str) != majority).astype(int)

    X = df.drop(columns=[target_col])

    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    print("\nIdentified numeric columns:", numeric_cols)
    print("Identified categorical columns:", categorical_cols)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()), 
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor

def train_and_evaluate_knn(preprocessor: ColumnTransformer, X_train, X_test, y_train, y_test, knn_params: dict, title: str) -> dict:
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("knn", KNeighborsClassifier(**knn_params)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)
    print("Params:", knn_params)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No churn", "Churn"])
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    plt.show()

    return {"model": model, "acc": acc, "y_pred": y_pred, "cm": cm}

def tune_k_with_cv(preprocessor: ColumnTransformer, X_train, y_train, k_values: list[int]) -> tuple[int, list[float]]:
    """
    Uses cross-validation on training set to pick best k.
    """
    cv_scores = []

    for k in k_values:
        pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("knn", KNeighborsClassifier(n_neighbors=k)),
            ]
        )
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        cv_scores.append(scores.mean())

    best_idx = int(np.argmax(cv_scores))
    best_k = k_values[best_idx]
    return best_k, cv_scores


def plot_k_vs_accuracy(k_values: list[int], cv_scores: list[float]) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, cv_scores, marker="o")
    plt.title("K vs Cross-Validation Accuracy")
    plt.xlabel("k (n_neighbors)")
    plt.ylabel("CV Accuracy")
    plt.xticks(k_values)
    plt.tight_layout()
    plt.show()

def compare_distance_metrics(preprocessor: ColumnTransformer, X_train, X_test, y_train, y_test, best_k: int) -> None:
    """
    Compare:
      - Euclidean: metric='minkowski', p=2 (default)
      - Manhattan: metric='minkowski', p=1
      - Minkowski p=3
    """
    print("\n" + "=" * 90)
    print("7) COMPARE DISTANCE METRICS")
    print("=" * 90)

    configs = [
        ("Euclidean (p=2)", {"n_neighbors": best_k, "metric": "minkowski", "p": 2}),
        ("Manhattan (p=1)", {"n_neighbors": best_k, "metric": "minkowski", "p": 1}),
        ("Minkowski (p=3)", {"n_neighbors": best_k, "metric": "minkowski", "p": 3}),
    ]

    summary = []
    for name, params in configs:
        res = train_and_evaluate_knn(preprocessor, X_train, X_test, y_train, y_test, params, f"KNN — {name}")
        summary.append((name, res["acc"]))

    print("\nMetric comparison (accuracy):")
    for name, acc in sorted(summary, key=lambda x: x[1], reverse=True):
        print(f"  {name:<18} -> {acc:.4f}")

def pca_2d_visualization(preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series) -> None:
    """
    Reduce transformed feature space to 2D and plot.
    Note: We fit preprocessor on all X only for visualization.
    """
    print("\n" + "=" * 90)
    print("BONUS) PCA 2D VISUALIZATION")
    print("=" * 90)

    X_trans = preprocessor.fit_transform(X)
    if hasattr(X_trans, "toarray"):
        X_trans_dense = X_trans.toarray()
    else:
        X_trans_dense = X_trans

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_trans_dense)

    plt.figure(figsize=(7, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y.values, s=10, alpha=0.7) #type:ignore
    plt.title("PCA (2D) of customer features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    cbar = plt.colorbar()
    cbar.set_label("Churn (0=No, 1=Yes)")
    plt.tight_layout()
    plt.show()


def main():
    df, is_real = load_telco_or_simulate(DEFAULT_CSV)
    df = clean_telco_totalcharges(df)
    print_head_and_schema(df, is_real)
    explore_target_distribution(df, target_col="Churn")
    quick_numeric_correlations(df, target_col="Churn")
    for cat in ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen"]:
        churn_rate_by_category(df, cat, target_col="Churn")

    print("\n" + "=" * 90)
    print("3–4) CLEANING + ENCODING + SPLITTING")
    print("=" * 90)

    X, y = prepare_features_and_target(df, target_col="Churn")
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,   
    )

    _ = train_and_evaluate_knn(
        preprocessor, X_train, X_test, y_train, y_test,
        knn_params={}, 
        title="KNN — Default (k=5, Euclidean)"
    )

    
    print("\n" + "=" * 90)
    print("6) OPTIMIZE k (n_neighbors)")
    print("=" * 90)

    k_values = [1, 3, 5, 7, 10, 15, 20, 25, 30]
    best_k, cv_scores = tune_k_with_cv(preprocessor, X_train, y_train, k_values)
    print("k values:", k_values)
    print("CV accuracies:", [round(s, 4) for s in cv_scores])
    print(f"Best k by CV accuracy: {best_k}")

    plot_k_vs_accuracy(k_values, cv_scores)

    _ = train_and_evaluate_knn(
        preprocessor, X_train, X_test, y_train, y_test,
        knn_params={"n_neighbors": best_k},
        title=f"KNN — Tuned k={best_k} (Euclidean)"
    )

    compare_distance_metrics(preprocessor, X_train, X_test, y_train, y_test, best_k=best_k)

    pca_2d_visualization(preprocessor, X, y)

    print("\n" + "=" * 90)
    print("INSIGHTS (what you should write in your brief report)")
    print("=" * 90)
    print("- Scaling matters a lot for KNN because distance depends on feature magnitudes.")
    print("- k controls bias/variance: small k can overfit; larger k smooths boundaries.")
    print("- Different distance metrics can change which neighbors are 'closest'.")
    print("- KNN can be slow on large datasets because prediction searches neighbors among many points.")
    print("- One-hot encoding can create many dimensions; this can make distances less informative (curse of dimensionality).")


if __name__ == "__main__":
    main()
