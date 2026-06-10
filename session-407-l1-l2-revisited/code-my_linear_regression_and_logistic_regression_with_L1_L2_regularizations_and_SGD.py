import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap



class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.0001, n_iters=30000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            y_predicted = np.dot(X, self.weights) + self.bias

            # compute gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted


class LogisticRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)



def run_my_own_logistic_regression():
    # =========================
    # Load dataset
    # =========================

    data = pd.read_csv("Social_Network_Ads.csv")

    X_raw = data[["Age", "EstimatedSalary"]].values
    y = data["Purchased"].values

    # =========================
    # Feature Scaling
    # =========================

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # =========================
    # Train/Test Split
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=0
    )

    X_train_raw, X_test_raw, _, _ = train_test_split(
        X_raw, y, test_size=0.25, random_state=0
    )

    # =========================
    # Train Model
    # =========================

    model = LogisticRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    # =========================
    # Visualization
    # =========================

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    cmap = ListedColormap(("red", "green"))
    colors = ["red", "green"]

    # =========================================================
    # LEFT SUBPLOT : SCALED FEATURES
    # =========================================================

    ax = axes[0]

    X_set, y_set = X_test, y_test

    x_min, x_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
    y_min, y_max = X_set[:, 1].min() - 1, X_set[:, 1].max() + 1

    X1, X2 = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    grid = np.c_[X1.ravel(), X2.ravel()]
    Z = model.predict(grid).reshape(X1.shape)

    ax.contourf(X1, X2, Z, alpha=0.75, cmap=cmap)

    for i, label in enumerate(np.unique(y_set)):
        ax.scatter(
            X_set[y_set == label, 0],
            X_set[y_set == label, 1],
            color=colors[i],
            label=label,
        )

    ax.set_title("Scaled Features")
    ax.set_xlabel("Age (Standardized)")
    ax.set_ylabel("Estimated Salary (Standardized)")
    ax.legend()

    # =========================================================
    # RIGHT SUBPLOT : ORIGINAL FEATURES
    # =========================================================

    ax = axes[1]

    X_set, y_set = X_test_raw, y_test

    x_min, x_max = X_set[:, 0].min() - 5, X_set[:, 0].max() + 5
    y_min, y_max = X_set[:, 1].min() - 10000, X_set[:, 1].max() + 10000

    X1, X2 = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    grid_raw = np.c_[X1.ravel(), X2.ravel()]

    # transform before prediction
    grid_scaled = scaler.transform(grid_raw)

    Z = model.predict(grid_scaled).reshape(X1.shape)

    ax.contourf(X1, X2, Z, alpha=0.75, cmap=cmap)

    for i, label in enumerate(np.unique(y_set)):
        ax.scatter(
            X_set[y_set == label, 0],
            X_set[y_set == label, 1],
            color=colors[i],
            label=label,
        )

    ax.set_title("Original Features")
    ax.set_xlabel("Age")
    ax.set_ylabel("Estimated Salary")
    ax.legend()

    plt.tight_layout()
    plt.show()


def run_my_own_linear_regression():
    dataset = pd.read_csv("Salary_Data.csv")

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # =========================
    # Train/Test split
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=0
    )

    # =========================
    # Train model
    # =========================

    model = MyOwnLinearRegression()
    model.fit(X_train, y_train)

    # =========================
    # Visualization
    # =========================

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ====================================
    # LEFT : Training set
    # ====================================

    ax = axes[0]

    ax.scatter(X_train, y_train, color="red")

    # regression line
    X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)

    ax.plot(X_line, y_line, color="blue")

    ax.set_title("Salary vs Experience (Training set)")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")

    # ====================================
    # RIGHT : Test set
    # ====================================

    ax = axes[1]

    ax.scatter(X_test, y_test, color="red")

    # same regression line
    ax.plot(X_line, y_line, color="blue")

    ax.set_title("Salary vs Experience (Test set)")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_my_own_linear_regression()
    run_my_own_logistic_regression()