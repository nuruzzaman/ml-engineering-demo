import numpy as np
import pickle
from numpy import ravel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from simple_linear_regr_utils import generate_data, evaluate


class Test1:
    """
    This is a simple implementation of linear regression
    using stochastic gradient descent (SGD) to fit the model.
    """

    def __init__(self, iterations=15000, lr=0.1, l2=0.01, reg_coef=0.01):
        self.iterations = iterations # number of iterations the fit method will be called
        self.lr = lr # The learning rate
        self.losses = [] # A list to hold the history of the calculated losses
        self.W, self.b = None, None # the slope and the intercept of the model
        self.reg_coef = reg_coef
        self.l2 = l2
        self.coef_ = None
        self.intercept_ = None

    def _loss(self, y, y_hat):
        """
        :param y: the actual output on the training set
        :param y_hat: the predicted output on the training set
        :return:
            loss: the sum of squared error
        """
        # Calculate the sum of squared errors (L2 loss) between the actual
        # and predicted outputs on the training set.
        loss = (1 / y.shape[0]) * np.sum((y - y_hat) ** 2)
        reg_loss = self.reg_coef * (self.coef_ ** 2).sum()
        self.losses.append(loss)
        return loss + reg_loss

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
        return dW, db

    def fit(self, X, y):
        """
        :param X: The training set
        :param y: The true output of the training set
        :return:
        """
        self.input_validation(X)
        self._init_weights(X)

        # Initialize the parameters
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        # Compute the loss and gradient
        for i in range(self.iterations + 1):
            y_hat = self.predict(X)
            loss = self._loss(y, y_hat)
            dW, db = self._sgd(X, y, y_hat)

            # Update the parameters
            # self.coef_ -= self.lr * grad[:-1]
            # self.intercept_ -= self.lr * grad[-1]

            self.W -= self.lr * dW
            self.b -= self.lr * db

            # Regularization
            self.coef_ = self.coef_ - (self.lr * self.l2 * self.coef_)

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")
        print(f"Initial Loss: {self.losses[0]}, Final Loss: {self.losses[-1]}")
        return self

    def predict(self, X):
        """
        :param X: The training dataset
        :return:
            y_hat: the predicted output
        """
        self.input_validation(X)

        # Calculate the predicted output y_hat.
        # remember the function of a line is defined as y = WX + b
        y_hat = np.dot(X, self.W) + self.b
        return y_hat

    def input_validation(self, X):
        """
        Validate the input data before training or predicting with the model.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X and y must be of type numpy.ndarray")
        if X.shape[1] <= 0:
            raise ValueError("X must have at least one feature")

    def params_optimization(self, X_train, y_train, X_test, y_test):
        # Define the parameter grid
        param_grid = {'learning_rate': [0.1, 0.01, 0.001],
                      'n_estimators': [500, 1000],
                      'max_depth': [4, 6, 8, 10],
                      'subsample': [0.9, 0.5, 0.2, 0.1],
                      'min_samples_split': [2, 4, 6],
                      'min_samples_leaf': [1, 2, 4]
                      }

        # Fit the grid search and find best params
        GBR = GradientBoostingRegressor()
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=GBR, param_grid=param_grid, cv=cv,
                                   verbose=1, n_jobs=-1, scoring='r2', return_train_score=True)
        grid_search.fit(X_train, ravel(y_train))

        print(f"The best estimator across ALL searched params: {grid_search.best_estimator_}")
        print(f"Best parameters: {grid_search.best_params_}")

        # Make predictions on the test set
        y_pred = grid_search.predict(X_test)
        print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
        print(f"R^2 score: {r2_score(y_test, y_pred)}")
        return grid_search.best_params_


if __name__ == "__main__":
    model = Test1()
    X_train, y_train, X_test, y_test = generate_data()

    # Hyperparam optimization
    # model.params_optimization(X_train, y_train, X_test, y_test)

    # Training
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    r2 = evaluate(model, X_test, y_test, predicted, 'plot_output.png')
    if r2 >= 0.4:
        # Save model if r2_score>0.4
        with open("Diabetes.pkl", "wb") as file:
            pickle.dump(model, file)

        # Load existing model
        loaded_model = pickle.load(open("Diabetes.pkl", "rb"))

        # Make prediction on unseen data
        X = np.array([[1], [2], [3]])
        y = loaded_model.predict(X)
        print('Predicted value: ', y)
