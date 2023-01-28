import numpy as np
import pickle
from simple_linear_regr_utils import generate_data, evaluate


class SimpleLinearRegression:

    def __init__(self, iterations=15000, lr=0.1):
        self.iterations = iterations
        self.lr = lr
        self.losses = []
        self.W, self.b = None, None

    def _loss(self, y, y_hat):
        diff = y_hat - y
        differences_squared = diff ** 2
        loss = differences_squared.mean()
        self.losses.append(loss)
        return loss

    def _init_weights(self, X):
        weights = np.random.normal(size=X.shape[1] + 1)
        self.W = weights[:X.shape[1]].reshape(-1, X.shape[1])
        self.b = weights[-1]

    def _sgd(self, X, y, y_hat):
        dW = (2 / X.shape[0]) * np.dot(X.T, (y_hat - y))
        db = (2 / X.shape[0]) * np.sum(y_hat - y)
        # Updates the weights using W and b using the learning rate and the values for dW and db
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db

    def fit(self, X, y):
        self._init_weights(X)

        for i in range(self.iterations + 1):
            y_hat = self.predict(X)
            loss = self._loss(y, y_hat)

            self._sgd(X, y, y_hat)
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")

        print(f"Initial Loss: {self.losses[0]}, Final Loss: {self.losses[-1]}")

    def predict(self, X):
        y_hat = np.dot(X, self.W) + self.b
        return y_hat

    def predict_proba(self, X):
        return self.predict(X)


if __name__ == "__main__":
    model = SimpleLinearRegression()
    X_train, y_train, X_test, y_test = generate_data()

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    r2 = evaluate(model, X_test, y_test, predicted)
    if r2 >= 0.4:
        with open("Diabetes.pkl", "wb") as file:
            pickle.dump(model, file)

        # Load existing model
        loaded_model = pickle.load(open("Diabetes.pkl", "rb"))

        # Make prediction on unseen data
        X = np.array([[1], [2], [3]])
        y = loaded_model.predict(X)
        print('Predicted value: ', y)

    y_pred = model.predict_proba(np.array([[1]]))
    print('y_pred: ', y_pred)
