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


class SimpleLinearRegression2:
    """
    This is a simple implementation of linear regression
    using stochastic gradient descent (SGD) to fit the model.
    """

    def __init__(self, iterations=50, learning_rate=0.1, decay_rate=0, batch_size=1, tolerance=1e-06):
        self.n_iter = iterations
        self.learn_rate = learning_rate
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.tolerance = tolerance
        self.time_complexity = None
        self.model = None

    def _init_weights(self, n_observation: int, random_state: int):
        """
        :param X: The training set
        """
        # ***********************************************
        #            Initialization
        # ***********************************************
        # [2.1] Initializing the random number generator
        seed = None if random_state is None else int(random_state)
        self.rng = np.random.default_rng(seed=seed)

        # [2.2] Initializing the values of the variables
        self.vector = np.array((0, 0), dtype="float64")

        # [2.3] Initializing and set the learning rate
        learn_rate = np.array(self.learn_rate, dtype="float64")
        if np.any(learn_rate <= 0):
            raise ValueError("'learn_rate' must be greater than zero")

        # [2.4] Initializing and set the decay rate
        decay_rate = np.array(self.decay_rate, dtype="float64")
        if np.any(decay_rate < 0) or np.any(decay_rate > 1):
            raise ValueError("'decay_rate' must be between zero and one")

        # [2.5] Initializing and set the size of mini-batches
        batch_size = int(self.batch_size)
        if not 0 < batch_size <= n_observation:
            raise ValueError(
                "'batch_size' must be greater than zero and less than "
                "or equal to the number of observations")

        # [2.6] Initializing and set the maximal number of iterations
        n_iter = int(self.n_iter)
        if n_iter <= 0:
            raise ValueError("'n_iter' must be greater than zero")

        # [2.7] Initializing and set the tolerance
        tolerance = np.array(self.tolerance, dtype="float64")
        if np.any(tolerance <= 0):
            raise ValueError("'tolerance' must be greater than zero")
        return self

    def _loss(self, x, y, b):
        """
        :param y: the actual output on the training set
        :param y_hat: the predicted output on the training set
        :return:
            loss: the sum of squared error (MSE)
        """
        # sum of squared residuals SSR or MSE
        residual = b[0] + b[1] * x - y
        return residual.mean(), (residual * x).mean()

    def _sgd(self, X_train, y_train, learn_rate=0.1, decay_rate=0.0, batch_size=1, n_iter=50,
             tolerance=1e-06, random_state=1234):

        # [1.1] Setting up the data type for NumPy arrays
        dtype_ = np.dtype("float64")

        # [1.2] Converting x and y to NumPy arrays
        x, y = np.array(X_train, dtype=dtype_), np.array(y_train, dtype=dtype_)
        n_observation = x.shape[0]
        if n_observation != y.shape[0]:
            raise ValueError("'X' and 'y' lengths do not match")

        # [1.3] use .reshape() to make sure that both x and y become 2D arrays
        # y has exactly one column and
        # concatenates the columns of x and y into a single array xy.
        xy = np.c_[x.reshape(n_observation, -1), y.reshape(n_observation, 1)]


        # [2] Weights Initialization
        self._init_weights(n_observation=n_observation, random_state=random_state)


        diff = 0
        # [3] Performing the stochastic gradient descent loop
        for _ in range(n_iter):
            # [3.1] Shuffle x and y to choose mini-batches randomly
            self.rng.shuffle(xy)

            # [3.2] Performing mini-batch moves
            for start in range(0, n_observation, batch_size):
                stop = start + batch_size
                x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

                # [3.3] Recalculating the difference in mini-batch
                grad = np.array(self._loss(x_batch, y_batch, self.vector), dtype_)

                # [3.4] momentum and the impact of the current gradient
                diff = decay_rate * diff - learn_rate * grad

                # [3.5] Checking if the absolute difference is small enough
                if np.all(np.abs(diff) <= tolerance):
                    break

                # [3.6] Updating the values of the variables
                self.vector += diff

            print(f"Initial Loss: {diff[0]}, Final Loss: {diff[-1]}")
        return self.vector if self.vector.shape else self.vector.item()

    def fit(self, X_train, y_train):
        start_time = time.time()

        self.data_validation(X_train)

        self.model = self._sgd(X_train, y_train,
                               learn_rate=self.learn_rate,
                               decay_rate=self.decay_rate,
                               batch_size=self.batch_size,
                               n_iter=self.n_iter,
                               tolerance=self.tolerance,
                               random_state=1234)

        end_time = time.time()
        self.time_complexity = end_time - start_time
        return self.model

    def predict(self, X):
        """
        :param X: The training dataset
        :return:
            y_hat: the predicted output
        """
        self.data_validation(X)

        # Calculate the predicted output y_hat.
        # remember the function of a line is defined as y = WX + b
        y_hat = self.model[0] + self.model[1] * X
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

        feature_importance = grid_search.best_estimator_.feature_importances_
        print(f'feature importance: {feature_importance}')

        # Make predictions on the test set
        y_pred = grid_search.best_estimator_.predict(X_test)
        print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
        print(f"Coefficient of determination: {r2_score(y_test, y_pred)}")
        return grid_search


def main():
    # Get and split dataset
    X_train, y_train, X_test, y_test = generate_data()

    model = SimpleLinearRegression2()
    # Hyperparam optimization
    # model.params_optimization(X_train, y_train, X_test, y_test)

    # Features Scaling
    # X_train_scaled, X_test_scaled = feature_scaling(X_train, X_test)

    # Training
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print('predicted: ', predicted)

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
        # scaler = StandardScaler()
        # X_new = scaler.fit_transform(new_data)
        y_new = loaded_model.predict(new_data)
        print(f'Predicted values from new data:\n {y_new}')

    print(f"Total time taken: {model.time_complexity:.2f} second\n")


if __name__ == "__main__":
    main()

