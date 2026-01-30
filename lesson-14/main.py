from __future__ import annotations

import sys
from dataclasses import dataclass

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd


@dataclass
class SplitConfig:
    test_size: float = 0.30
    random_state: int = 42


def gaussian_nb_iris(cfg: SplitConfig) -> None:
    print("\n" + "=" * 70)
    print("TASK 1 — Gaussian Naive Bayes on Iris (real-valued data)")
    print("=" * 70)

    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def load_sms_spam_dataset(url: str) -> pd.DataFrame:
    """
    Expected format: tab-separated with columns: label, message
    Example dataset: https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
    """
    try:
        df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SMS dataset from URL.\nURL: {url}\nError: {e}"
        ) from e

    if "label" not in df.columns or "message" not in df.columns:
        raise ValueError("Dataset does not contain required columns: label, message")

    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df[df["label"].isin(["spam", "ham"])].copy()
    if df.empty:
        raise ValueError("After filtering, dataset has no rows with label in {spam, ham}.")

    df["target"] = (df["label"] == "spam").astype(int)
    return df


def multinomial_nb_sms(cfg: SplitConfig) -> None:
    print("\n" + "=" * 70)
    print("TASK 2 — Multinomial Naive Bayes on SMS Spam (word frequencies)")
    print("=" * 70)

    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = load_sms_spam_dataset(url)

    X_text = df["message"].values
    y = df["target"].values
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y # type: ignore
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Dataset size: {len(df)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report (0=ham, 1=spam):")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))


def main() -> None:
    cfg = SplitConfig(test_size=0.30, random_state=42)

    gaussian_nb_iris(cfg)

    multinomial_nb_sms(cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
