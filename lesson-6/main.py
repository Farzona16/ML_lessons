import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/car_price.csv")

df.head()

df.info()
df.describe()

df.isnull().sum()

df = df.dropna()

X = df.drop(columns=["price"])
y = df["price"].values.reshape(-1, 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
m = X_scaled.shape[0]
X_b = np.c_[np.ones((m, 1)), X_scaled]
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost
def gradient_descent(X, y, alpha=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []

    for i in range(iterations):
        gradients = (1 / m) * X.T @ (X @ theta - y)
        theta -= alpha * gradients
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_b, y, test_size=0.2, random_state=42
)
learning_rates = [0.001, 0.01, 0.1]
histories = {}

for lr in learning_rates:
    theta, cost_history = gradient_descent(
        X_train, y_train, alpha=lr, iterations=1000
    )
    histories[lr] = cost_history
plt.figure(figsize=(8, 5))
for lr, cost in histories.items():
    plt.plot(cost, label=f"alpha={lr}")

plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.legend()
plt.title("Gradient Descent Convergence")
plt.show()
theta_final, _ = gradient_descent(X_train, y_train, alpha=0.01, iterations=1000)

y_train_pred = X_train @ theta_final
y_test_pred = X_test @ theta_final
from sklearn.metrics import mean_squared_error, r2_score

print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))

print("Train R²:", r2_score(y_train, y_train_pred))
print("Test R²:", r2_score(y_test, y_test_pred))
feature_name = "horsepower"
feature_index = X.columns.get_loc(feature_name)

plt.figure(figsize=(6, 4))
plt.scatter(X[feature_name], y, alpha=0.4, label="Actual")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Horsepower vs Car Price")
plt.show()
def mini_batch_gradient_descent(X, y, alpha=0.01, iterations=1000, batch_size=32):
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []

    for _ in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            gradients = (1 / len(y_batch)) * X_batch.T @ (X_batch @ theta - y_batch)
            theta -= alpha * gradients

        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train[:, 1:], y_train)

y_pred_sklearn = lr_model.predict(X_test[:, 1:])

print("Sklearn MSE:", mean_squared_error(y_test, y_pred_sklearn))
print("Sklearn R²:", r2_score(y_test, y_pred_sklearn))
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

X_poly_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
