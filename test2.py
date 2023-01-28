from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np


class CustomPerceptron(object):

    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate

    '''
    Stochastic Gradient Descent
    1. Weights are updated based on each training examples.
    2. Learning of weights can continue for multiple iterations
    3. Learning rate needs to be defined
    '''
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        for _ in range(self.n_iterations):
            for xi, expected_value in zip(X, y):
                predicted_value = self.predict(xi)
                self.coef_[1:] += self.learning_rate * (expected_value - predicted_value) * xi
                self.coef_[0] += self.learning_rate * (expected_value - predicted_value) * 1
    '''
    Activation function calculates the value of weighted sum of input value
    '''
    def activation(self, X):
        return np.dot(X, self.coef_[1:]) + self.coef_[0]

    '''
    Prediction is made on the basis of unit step function
    '''
    def predict(self, X):
        output = self.activation(X)
        return np.where(output >= 0.0, 1, 0)

    '''
    Model score is calculated based on comparison of
    expected value and predicted value
    '''
    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
            if (target != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count) / total_data_count
        return self.score_

#
# Load the data set
#
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target
#
# Create training and test split
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
#
# Instantiate CustomPerceptron
#
prcptrn = CustomPerceptron()
#
# Fit the model
#
prcptrn.fit(X_train, y_train)
#
# Score the model
#
score = prcptrn.score(X_test, y_test), prcptrn.score(X_train, y_train)
print('self.score_', score)


