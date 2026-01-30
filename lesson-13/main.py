import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


def main() -> None:
    df = pd.read_csv(DATA_URL, sep="\t")
    if "label" not in df.columns or "message" not in df.columns:
        raise ValueError("Expected columns 'label' and 'message' in the dataset.")

   
    df["label_bin"] = df["label"].map({"ham": 0, "spam": 1})
    if df["label_bin"].isna().any():
        bad = df.loc[df["label_bin"].isna(), "label"].unique()
        raise ValueError(f"Unexpected label values found: {bad}")

    X_text = df["message"].astype(str)
    y = df["label_bin"].astype(int)

   
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    vectorizer = CountVectorizer(binary=True)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = BernoulliNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred) 
    print("=== Bernoulli Naive Bayes (Binary CountVectorizer) ===")
    print(f"Accuracy: {acc:.4f}\n")

    print("Confusion Matrix (rows=true, cols=pred):")
    print("         Pred 0    Pred 1")
    print(f"True 0   {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"True 1   {cm[1,0]:6d}  {cm[1,1]:6d}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham(0)", "spam(1)"]))


if __name__ == "__main__":
    main()
