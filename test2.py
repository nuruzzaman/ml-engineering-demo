import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset
X, y = load_diabetes(return_X_y=True)

# Select the age feature
X = X[:, np.newaxis, 2]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class SimpleLinearRegression:
    def __init__(self):
        self.b = None
        self.w = None

    def fit(self, X, y, learning_rate=0.1, regularization=0.0, n_iters=10000):
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features)
        self.b = 0

        for _ in range(n_iters):
            # Select a random sample
            idx = np.random.randint(n_samples)
            x = X[idx]
            y_pred = self.predict(x)
            error = y[idx] - y_pred
            self.w = self.w + learning_rate * (error * x + regularization * self.w)
            self.b = self.b + learning_rate * error

    def predict(self, X):
        return np.dot(X, self.w) + self.b


# Initialize the model
model = SimpleLinearRegression()

# Fit the model to the data
model.fit(X_train, y_train, learning_rate=0.1, regularization=0.1)

# Print the model's coefficients
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Mean squared error: {mse:.2f}")

r2 = r2_score(y, y_pred)
print(f"Coefficient of determination: {r2:.2f}")
