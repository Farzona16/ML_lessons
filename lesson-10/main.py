import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

X = digits.data      
y = digits.target   
print("Dataset shape:", X.shape)
print("Target shape:", y.shape)
print("Number of classes:", len(np.unique(y)))
print("Unique labels:", np.unique(y))
plt.matshow(digits.images[0], cmap="gray")
plt.title(f"Label: {y[0]}")
plt.axis("off")
plt.show()
print("Min pixel:", np.min(X))
print("Max pixel:", np.max(X))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    multi_class="ovr",
    max_iter=5000,
    random_state=42
)

log_reg.fit(X_train_scaled, y_train)

from sklearn.svm import SVC

svm = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

svm.fit(X_train_scaled, y_train)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_svm = svm.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
plot_confusion(y_test, y_pred_lr, "Logistic Regression Confusion Matrix")
plot_confusion(y_test, y_pred_svm, "SVM Confusion Matrix")
plot_confusion(y_test, y_pred_rf, "Random Forest Confusion Matrix")
print("Logistic Regression Report\n", classification_report(y_test, y_pred_lr))
print("SVM Report\n", classification_report(y_test, y_pred_svm))
print("Random Forest Report\n", classification_report(y_test, y_pred_rf))
from sklearn.metrics import precision_score, recall_score, f1_score

def multiclass_metrics(y_true, y_pred):
    return {
        "Macro Precision": precision_score(y_true, y_pred, average="macro"),
        "Macro Recall": recall_score(y_true, y_pred, average="macro"),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted Precision": precision_score(y_true, y_pred, average="weighted"),
        "Weighted Recall": recall_score(y_true, y_pred, average="weighted"),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted")
    }

print("Logistic Regression:", multiclass_metrics(y_test, y_pred_lr))
print("SVM:", multiclass_metrics(y_test, y_pred_svm))
print("Random Forest:", multiclass_metrics(y_test, y_pred_rf))
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

def plot_roc(model, X_test, title):
    y_score = model.predict_proba(X_test)
    plt.figure(figsize=(8,6))

    for i in range(10):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i]) # type:ignore
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Digit {i} (AUC={roc_auc:.2f})")

    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()
plot_roc(log_reg, X_test_scaled, "Logistic Regression ROC Curves")
plot_roc(svm, X_test_scaled, "SVM ROC Curves")
import random

indices = random.sample(range(len(X_test)), 10)

plt.figure(figsize=(12,4))

for i, idx in enumerate(indices):
    plt.subplot(2,5,i+1)
    img = X_test[idx].reshape(8,8)
    true = y_test[idx]
    pred = y_pred_svm[idx]

    plt.imshow(img, cmap="gray")
    plt.title(f"T:{true} P:{pred}")
    color = "green" if true == pred else "red"
    for spine in plt.gca().spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
    plt.axis("off")

plt.show()


