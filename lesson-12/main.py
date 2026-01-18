from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def main():
    data = load_breast_cancer()
    X = data.data
    y = data.target  

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   
    X_test_scaled = scaler.transform(X_test)        

    baseline_clf = LogisticRegression(max_iter=10000, random_state=42)
    baseline_clf.fit(X_train_scaled, y_train)

    y_pred_baseline = baseline_clf.predict(X_test_scaled)
    baseline_acc = accuracy_score(y_test, y_pred_baseline)

   
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=10000, random_state=42))
    ])

    pipe.fit(X_train, y_train)
    y_pred_pipe = pipe.predict(X_test)
    pipe_acc = accuracy_score(y_test, y_pred_pipe)

    print("=== Breast Cancer Wisconsin: Logistic Regression ===")
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print()
    print(f"Baseline accuracy (manual scaling): {baseline_acc:.4f}")
    print(f"Pipeline accuracy:                {pipe_acc:.4f}")
    print()

    if abs(baseline_acc - pipe_acc) < 1e-12:
        diff_msg = "No, they are the same (for this split/settings)."
    else:
        diff_msg = f"Yes, they differ by {abs(baseline_acc - pipe_acc):.6f}."

    print("Comparison:")
    print(f"- Is the accuracy different? {diff_msg}")
    print("- Why prefer the pipeline even if accuracy is similar?")
    print("  1) Prevents data leakage: scaler is always fit only on training folds.")
    print("  2) Cleaner and safer workflow: one object handles preprocessing + model.")
    print("  3) Easier cross-validation / GridSearchCV: tuning works correctly end-to-end.")
    print("  4) More reproducible deployment: same steps applied consistently to new data.")


if __name__ == "__main__":
    main()
