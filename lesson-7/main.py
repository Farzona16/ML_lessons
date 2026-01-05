import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("data/california_housing.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression MSE:", mse_lr)
print("Linear Regression R2:", r2_lr)

alphas = np.logspace(-3, 3, 50)

ridge = Ridge()
ridge_cv = GridSearchCV(
    ridge,
    {"alpha": alphas},
    scoring="neg_mean_squared_error",
    cv=5
)

ridge_cv.fit(X_train_scaled, y_train)
ridge_best = ridge_cv.best_estimator_

y_pred_ridge = ridge_best.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("Ridge Best Alpha:", ridge_cv.best_params_)
print("Ridge MSE:", mse_ridge)
print("Ridge R2:", r2_ridge)

ridge_coefs = []
for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_train_scaled, y_train)
    ridge_coefs.append(model.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_coefs)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficients")
plt.title("Ridge Coefficient Shrinkage")
plt.show()


lasso = Lasso(max_iter=5000)

lasso_cv = GridSearchCV(
    lasso,
    {"alpha": alphas},
    scoring="neg_mean_squared_error",
    cv=5
)

lasso_cv.fit(X_train_scaled, y_train)
lasso_best = lasso_cv.best_estimator_

y_pred_lasso = lasso_best.predict(X_test_scaled)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("Lasso Best Alpha:", lasso_cv.best_params_)
print("Lasso MSE:", mse_lasso)
print("Lasso R2:", r2_lasso)

lasso_coefs = []
for a in alphas:
    model = Lasso(alpha=a, max_iter=5000)
    model.fit(X_train_scaled, y_train)
    lasso_coefs.append(model.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alphas, lasso_coefs)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficients")
plt.title("Lasso Coefficient Shrinkage")
plt.show()

lasso_feature_importance = pd.Series(
    lasso_best.coef_, index=X.columns
)

print("Lasso Selected Features:")
print(lasso_feature_importance[lasso_feature_importance != 0].sort_values())

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression"],
    "MSE": [mse_lr, mse_ridge, mse_lasso],
    "R2": [r2_lr, r2_ridge, r2_lasso],
    "Non-zero Coefficients": [
        len(lr.coef_),
        len(ridge_best.coef_),
        (lasso_best.coef_ != 0).sum()
    ]
})

print("\nModel Comparison:")
print(comparison)
