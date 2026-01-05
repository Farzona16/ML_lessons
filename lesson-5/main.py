import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

df = pd.concat([X, y], axis=1)
df.head()
df.shape
df.columns.tolist()
df.describe()
from sklearn.preprocessing import PolynomialFeatures

poly_features = ["bmi", "bp", "s5"]

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly_part = poly.fit_transform(X[poly_features])
poly_feature_names = poly.get_feature_names_out(poly_features)

print("Number of polynomial features:", len(poly_feature_names))
print(poly_feature_names)
X_remaining = X.drop(columns=poly_features)

X_poly_full = pd.concat(
    [
        X_remaining.reset_index(drop=True),
        pd.DataFrame(X_poly_part, columns=poly_feature_names)
    ],
    axis=1
)
from sklearn.model_selection import train_test_split

X_train_base, X_test_base, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_poly, X_test_poly, _, _ = train_test_split(
    X_poly_full, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LinearRegression

lr_base = LinearRegression()
lr_base.fit(X_train_base, y_train)


lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)

coef_df = pd.DataFrame({
    "feature": X_poly_full.columns,
    "coefficient": lr_poly.coef_
})

coef_df["abs_coef"] = coef_df["coefficient"].abs()

coef_df.sort_values("abs_coef", ascending=False).head(10)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }
baseline_metrics = evaluate(lr_base, X_test_base, y_test)
poly_metrics = evaluate(lr_poly, X_test_poly, y_test)

print(baseline_metrics, poly_metrics)

import matplotlib.pyplot as plt

y_pred_base = lr_base.predict(X_test_base)
y_pred_poly = lr_poly.predict(X_test_poly)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_base, alpha=0.6, label="Baseline")
plt.scatter(y_test, y_pred_poly, alpha=0.6, label="Polynomial")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'k--')

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
plt.title("Predictions vs Actual")
plt.show()
metrics_df = pd.DataFrame({
    "Baseline": baseline_metrics,
    "Polynomial": poly_metrics
})

metrics_df.loc[["MAE", "RMSE"]].T.plot(kind="bar", figsize=(6,4))
plt.title("Error Comparison")
plt.ylabel("Error")
plt.show()
X_bmi = X["bmi"].values.reshape(-1,1)
y_target = y.values

poly_bmi = PolynomialFeatures(degree=2, include_bias=False)
X_bmi_poly = poly_bmi.fit_transform(X_bmi)

model_bmi = LinearRegression()
model_bmi.fit(X_bmi_poly, y_target) #type:ignore

sorted_idx = np.argsort(X_bmi.flatten())
X_sorted = X_bmi[sorted_idx]
y_pred_curve = model_bmi.predict(
    poly_bmi.transform(X_sorted)
)

plt.scatter(X_bmi, y_target, alpha=0.3)  #type:ignore
plt.plot(X_sorted, y_pred_curve, color="red")
plt.xlabel("BMI")
plt.ylabel("Target")
plt.title("Polynomial Relationship: BMI vs Target")
plt.show()
input_data = pd.DataFrame([{
    "age": 0.05,
    "sex": 0.02,
    "bmi": 0.04,
    "bp": 0.03,
    "s1": -0.02,
    "s2": -0.01,
    "s3": 0.00,
    "s4": 0.02,
    "s5": 0.03,
    "s6": 0.01
}])

poly_part = poly.transform(input_data[poly_features])

input_poly = pd.concat(
    [
        input_data.drop(columns=poly_features),
        pd.DataFrame(poly_part, columns=poly_feature_names)  #type:ignore
    ],
    axis=1
)
linear_pred = lr_base.predict(input_data)[0]
poly_pred = lr_poly.predict(input_poly)[0]

print(linear_pred, poly_pred)

