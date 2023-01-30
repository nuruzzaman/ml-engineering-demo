import numpy as np
import pickle
import time
import sys
from numpy import ravel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from simple_linear_regr_utils import generate_data, evaluate, feature_scaling


class SimpleLinearRegression:
    """
    This is a simple implementation of linear regression
    using stochastic gradient descent (SGD) to fit the model.
    """

    def __init__(self, iterations=15000, learning_rate=0.1, l2=0.01):
        self.iterations = iterations  # number of iterations the fit method will be called
        self.learning_rate = learning_rate  # The learning rate
        self.losses = []  # A list to hold the history of the calculated losses
        self.W, self.b = None, None  # the slope and the intercept of the model
        self.l2 = l2  # prevent overfitting by adding a penalty term to the loss function
        self.time_complexity = None

    def _init_weights(self, X):
        """
        :param X: The training set
        """
        # Initializes the weights (slope and intercept) with random values.
        weights = np.random.normal(size=X.shape[1] + 1)
        self.W = weights[:X.shape[1]].reshape(-1, X.shape[1])
        self.b = weights[-1]
        return self

    def _loss(self, y, y_hat):
        """
        :param y: the actual output on the training set
        :param y_hat: the predicted output on the training set
        :return:
            loss: the sum of squared error (MSE)
        """
        # Calculate the sum of squared errors (L2 loss) between the actual
        # and predicted outputs on the training set.
        loss = (1 / y.shape[0]) * np.sum((y - y_hat) ** 2)

        # Add the L2 regularization term to the loss
        # np.sum(self.W ** 2) is the L2 regularization term
        loss += (self.l2 / (2 * y.shape[0])) * np.sum(self.W ** 2)
        self.losses.append(loss)
        return loss

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
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def fit(self, X, y):
        """
        :param X: The training set
        :param y: The true output of the training set
        :return:
        """
        start_time = time.time()
        self.data_validation(X)
        self._init_weights(X)

        # gradient descent learning to update weights
        for i in range(self.iterations+1):
            y_hat = self.predict(X)
            self._sgd(X, y, y_hat)
            loss = self._loss(y, y_hat)

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")

        end_time = time.time()
        self.time_complexity = end_time - start_time

        print(f"\nInitial Loss: {self.losses[0]}, Final Loss: {self.losses[-1]}")
        return self

    def predict(self, X):
        """
        :param X: The training dataset
        :return:
            y_hat: the predicted output
        """
        self.data_validation(X)

        # Calculate the predicted output y_hat.
        # remember the function of a line is defined as y = WX + b
        y_hat = np.dot(X, self.W.T) + self.b
        return y_hat

    def data_validation(self, X):
        """
        Data validation to check the input data is in correct format before fitting the model.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X and y must be of type numpy.ndarray")
        if X.shape[1] <= 0:
            raise ValueError("X must have at least one feature")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

    def memory_complexity(self):
        """  memory complexity of the model's parameters.
            calculates the amount of memory used by an algorithm during its execution
        params:
            coef_: holds the values of the coefficients
            W.T : the transpose weight matrix
        :return:
            bytes
        """
        return sys.getsizeof(self.W.T)


def main():
    # Get and split dataset
    X_train, y_train, X_test, y_test = generate_data()

    model = SimpleLinearRegression()
    # Training
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    # Evaluation
    r2 = evaluate(model, X_test, y_test, predicted, 'plot_output.png')
    if r2 >= 0.4:
        # Save model if r2_score>0.4
        with open("Diabetes.pkl", "wb") as file:
            pickle.dump(model, file)
        print('\nModel saved!')

        # Load existing model
        loaded_model = pickle.load(open("Diabetes.pkl", "rb"))
        print('Model reloaded for inference!')

        # Make prediction on unseen data
        ##################################
        new_data = np.array([[1], [2], [3]])
        y_new = loaded_model.predict(new_data)
        print(f'Predicted values from new data:\n {y_new}')

    print(f'Memory complexity: {model.memory_complexity()} bytes')
    print(f"Total time taken: {model.time_complexity:.2f} second\n")


if __name__ == "__main__":
    main()
