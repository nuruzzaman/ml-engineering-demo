import unittest
import numpy as np
from simple_linear_regr_utils import generate_data, evaluate
from simple_linear_regr_demo1 import SimpleLinearRegression


class TestSimpleLinearRegression(unittest.TestCase):
    """
    This test suite includes six test cases for end-to-end functionality testing of the model:
        1. test_init: tests for params initialization
        2. test_loss: tests that the loss is a float.
        3. test_init_weights: tests that the weights are of the correct type (ndarray and float).
        4. test_sgd: tests that the updated weights are of the correct type (ndarray and float).
        5. test_fit: tests for fit the model
        6. test_predict: the shape of the predicted output is the same as the shape of the true output on the test set.
        7. test_evaluate: evaluates the model using the test data and the predicted output.
        8. test_validate_input: validate training dataset
    """
    def setUp(self):
        self.X_train, self.y_train, self.X_test, self.y_test = generate_data()
        self.model = SimpleLinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def test_init(self):
        self.assertIsInstance(self.model, SimpleLinearRegression)
        self.assertEqual(self.model.iterations, 15000)
        self.assertEqual(self.model.learning_rate, 0.1)

    def test_loss(self):
        # y = np.array([1, 2, 3])
        # y_predicted = np.array([1.1, 1.9, 3.2])
        y_predicted = self.model.predict(self.X_train)
        loss = self.model._loss(self.y_train, y_predicted)
        self.assertIsInstance(loss, float)

    def test_init_weights(self):
        # X = np.array([[1], [2], [3]])
        self.model._init_weights(self.X_train)
        self.assertIsInstance(self.model.W, np.ndarray)
        self.assertEqual(self.model.W.shape, (1, 1))
        self.assertIsInstance(self.model.b, float)

    def test_sgd(self):
        # X = np.array([[1], [2], [3]])
        # y = np.array([1, 2, 3])
        y_predicted = self.model.predict(self.X_train)
        self.model._sgd(self.X_train, self.y_train, y_predicted)
        self.assertIsInstance(self.model.W, np.ndarray)
        self.assertIsInstance(self.model.b, float)

    def test_fit(self):
        self.model.fit(self.X_train, self.y_train)
        self.assertIsInstance(self.model.W, np.ndarray)
        self.assertEqual(self.model.W.shape, (1, 1))
        self.assertIsInstance(self.model.b, float)
        self.assertGreater(self.model.losses[-1], 0)

    def test_predict(self):
        # X = np.array([[1], [2], [3]])
        y_predicted = self.model.predict(self.X_test)
        self.assertIsInstance(y_predicted, np.ndarray)
        self.assertTrue(np.all(y_predicted.shape == self.y_test.shape))

    def test_evaluate(self):
        self.model.fit(self.X_train, self.y_train)
        y_predicted = self.model.predict(self.X_test)
        evaluate(self.model, self.X_test, self.y_test, y_predicted)


if __name__ == '__main__':
    unittest.main()
