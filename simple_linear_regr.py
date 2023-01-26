import numpy as np
import joblib
import pickle
from numpy import ravel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedKFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from simple_linear_regr_utils import generate_data, evaluate


class SimpleLinearRegression:
    """
    This is a simple implementation of linear regression
    using stochastic gradient descent (SGD) to fit the model.
    """

    def __init__(self, iterations=15000, lr=0.1):
        self.iterations = iterations # number of iterations the fit method will be called
        self.lr = lr # The learning rate
        self.losses = [] # A list to hold the history of the calculated losses
        self.W, self.b = None, None # the slope and the intercept of the model

    def _loss(self, y, y_hat):
        """
        :param y: the actual output on the training set
        :param y_hat: the predicted output on the training set
        :return:
            loss: the sum of squared error
        """
        # Calculate the sum of squared errors (L2 loss) between the actual
        # and predicted outputs on the training set.
        diff = y_hat - y
        differences_squared = diff ** 2
        loss = differences_squared.mean()
        self.losses.append(loss)
        return loss

    def _init_weights(self, X):
        """
        :param X: The training set
        """
        # Initializes the weights (slope and intercept) with random values.
        weights = np.random.normal(size=X.shape[1] + 1)
        self.W = weights[:X.shape[1]].reshape(-1, X.shape[1])
        self.b = weights[-1]

    def _sgd(self, X, y, y_hat):
        """
        :param X: The training set
        :param y: The actual output on the training set
        :param y_hat: The predicted output on the training set
        :return:
            sets updated W and b to the instance Object (self)
        """
        # Calculated the gradients of the loss function using dW & db.
        dW = (2 / X.shape[0]) * np.dot(X.T, (y_hat - y))
        db = (2 / X.shape[0]) * np.sum(y_hat - y)
        # Updates the weights using W and b using the learning rate and the values for dW and db
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db

    def fit(self, X, y):
        """
        :param X: The training set
        :param y: The true output of the training set
        :return:
        """
        self.input_validation(X)

        self._init_weights(X)
        y_hat = self.predict(X)
        loss = self._loss(y, y_hat)
        print(f"Initial Loss: {loss}")
        for i in range(self.iterations + 1):
            self._sgd(X, y, y_hat)
            y_hat = self.predict(X)
            loss = self._loss(y, y_hat)
            if not i % 100:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, X):
        """
        :param X: The training dataset
        :return:
            y_hat: the predicted output
        """
        self.input_validation(X)

        # Calculate the predicted output y_hat.
        # remember the function of a line is defined as y = WX + b
        y_hat = np.dot(X, self.W.T) + self.b
        return y_hat

    def input_validation(self, X):
        """
        Validate the input data before training or predicting with the model.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X and y must be of type numpy.ndarray")
        if X.shape[1] <= 0:
            raise ValueError("X must have at least one feature")


if __name__ == "__main__":
    model = SimpleLinearRegression()
    X_train, y_train, X_test, y_test = generate_data()

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    evaluate(model, X_test, y_test, predicted)

    # Save trained model to file
    pickle.dump(model, open("web/model/Diabetes.pkl", "wb"))
    # Load model
    loaded_model = pickle.load(open("web/model/Diabetes.pkl", "rb"))

    # Make prediction on unseen data
    X = np.array([[1], [2], [3]])
    y = loaded_model.predict(X)
    print('Predicted value: ', y)
