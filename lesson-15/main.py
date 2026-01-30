from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.decomposition import PCA


RANDOM_STATE = 42


def show_basic_info(digits) -> None:
    X = digits.data
    y = digits.target
    images = digits.images

    print("\n" + "=" * 80)
    print("1) LOAD DATASET")
    print("=" * 80)
    print(f"X shape (samples, features): {X.shape}")
    print(f"images shape (samples, 8, 8): {images.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {np.unique(y)}")

    print("\nShowing 10 sample digits...")
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"Label: {y[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def data_exploration(digits) -> None:
    X = digits.data
    y = digits.target

    print("\n" + "=" * 80)
    print("2) DATA EXPLORATION")
    print("=" * 80)
    n_samples, n_features = X.shape
    print(f"Number of samples: {n_samples}")
    print(f"Number of features: {n_features} (8x8 = 64 pixels)")

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution (digit -> count):")
    for k, v in zip(unique, counts):
        print(f"  {k} -> {v}")

    plt.figure(figsize=(8, 4))
    plt.bar(unique, counts)
    plt.xticks(unique)
    plt.title("Target class distribution (Digits 0–9)")
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    print("\nPlotting some random examples...")
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y), size=12, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    axes = axes.ravel()
    for ax, i in zip(axes, idx):
        ax.imshow(digits.images[i], cmap="gray")
        ax.set_title(f"y={y[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def split_and_scale(digits):
    X = digits.data
    y = digits.target

    print("\n" + "=" * 80)
    print("3) TRAIN/TEST SPLIT + SCALING")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train size: {X_train.shape[0]} samples")
    print(f"Test size:  {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def evaluate_model(name: str, model: Pipeline, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "-" * 80)
    print(f"{name}")
    print("-" * 80)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return {"name": name, "acc": acc, "macro_f1": macro_f1, "cm": cm, "y_pred": y_pred}


def train_linear_poly_rbf(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 80)
    print("4–5) SVM: LINEAR vs POLY vs RBF")
    print("=" * 80)

    results = []

    linear = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="linear", C=1.0)),
        ]
    )
    linear.fit(X_train, y_train)
    results.append(evaluate_model("SVM (Linear kernel)", linear, X_test, y_test))

    poly = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="poly", C=1.0, degree=3, gamma="scale")),
        ]
    )
    poly.fit(X_train, y_train)
    results.append(evaluate_model("SVM (Polynomial kernel)", poly, X_test, y_test))

    rbf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ]
    )
    rbf.fit(X_train, y_train)
    results.append(evaluate_model("SVM (RBF kernel)", rbf, X_test, y_test))

    print("\n" + "=" * 80)
    print("KERNEL COMPARISON (quick)")
    print("=" * 80)
    for r in results:
        print(f"{r['name']:<28}  acc={r['acc']:.4f}  macro_f1={r['macro_f1']:.4f}")

    return results


def grid_search_best_svm(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 80)
    print("6) HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 80)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC()),
        ]
    )

    param_grid = [
        {
            "svc__kernel": ["linear"],
            "svc__C": [0.1, 1, 10],
        },
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1, 10, 50],
            "svc__gamma": [0.001, 0.01, 0.1, "scale"],
        },
        {
            "svc__kernel": ["poly"],
            "svc__C": [0.1, 1, 10],
            "svc__degree": [2, 3, 4],
            "svc__gamma": [0.001, 0.01, 0.1, "scale"],
        },
    ]

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",   
        cv=5,
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train, y_train)

    print("\nBest CV score (macro F1):", f"{grid.best_score_:.4f}")
    print("Best params:", grid.best_params_)

    best_model = grid.best_estimator_
    best_res = evaluate_model("BEST MODEL (from GridSearchCV)", best_model, X_test, y_test)

    return best_model, best_res


def plot_best_confusion_matrix(best_model: Pipeline, X_test, y_test) -> None:
    print("\n" + "=" * 80)
    print("7) VISUALIZATION — CONFUSION MATRIX (Best Model)")
    print("=" * 80)

    y_pred = best_model.predict(X_test)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=np.unique(y_test), cmap="Blues", values_format="d"
    )
    plt.title("Confusion Matrix (Best SVM)")
    plt.tight_layout()
    plt.show()


def pca_visualization(digits, best_model: Pipeline) -> None:
    """
    PCA to 2D just for visualization.
    We'll show points colored by true label.
    """
    print("\n" + "=" * 80)
    print("7) VISUALIZATION — PCA (2D)")
    print("=" * 80)

    X = digits.data
    y = digits.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance (2D): {pca.explained_variance_ratio_.sum():.4f}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=18, alpha=0.8)
    plt.title("Digits dataset visualized in 2D with PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Digit label")
    plt.tight_layout()
    plt.show()


def support_vectors_info(best_model: Pipeline) -> None:
    """
    Bonus-ish: support vectors role.
    """
    print("\n" + "=" * 80)
    print("BONUS: SUPPORT VECTORS INFO (Best Model)")
    print("=" * 80)

    svc: SVC = best_model.named_steps["svc"]
    n_sv = svc.n_support_.sum()
    print(f"Support vectors per class: {svc.n_support_}")
    print(f"Total support vectors: {n_sv}")
    print("Idea: support vectors are the 'critical' training points closest to the margin;")
    print("they define the decision boundary. More SVs can mean a more complex boundary.")


def main():
    digits = load_digits()

  
    show_basic_info(digits)

    data_exploration(digits)


    X_train, X_test, y_train, y_test = split_and_scale(digits)

    _ = train_linear_poly_rbf(X_train, y_train, X_test, y_test)

    best_model, best_res = grid_search_best_svm(X_train, y_train, X_test, y_test)

    plot_best_confusion_matrix(best_model, X_test, y_test)


    pca_visualization(digits, best_model)

  
    support_vectors_info(best_model)


if __name__ == "__main__":
    main()
