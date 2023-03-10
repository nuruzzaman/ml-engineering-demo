import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def generate_data():
    """
    Generates a random dataset from a normal distribution.

    Returns:
        diabetes_X_train: the training dataset
        diabetes_y_train: The output corresponding to the training set
        diabetes_X_test: the test dataset
        diabetes_y_test: The output corresponding to the test set
    """
    # Load the diabetes dataset
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data (X) into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets (y) into training/testing sets
    diabetes_y_train = diabetes_y[:-20].reshape(-1, 1)
    diabetes_y_test = diabetes_y[-20:].reshape(-1, 1)

    print(f"# Training Samples: {len(diabetes_X_train)}; # Test samples: {len(diabetes_X_test)};")
    return diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test


def feature_scaling(X_train, X_test):
    """ Features Scaling using StandardScaler and Imputation
    Only one feature is give which is numeric. so we dont need `OneHotEncoder`
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    X_train_scaled = num_pipeline.fit_transform(X_train)
    X_test_scaled = num_pipeline.transform(X_test)
    return X_train_scaled, X_test_scaled


def evaluate(model, X, y, y_predicted, plot_filename=None):
    """ Calculates and prints evaluation metrics. """
    # The mean squared error
    mse = mean_squared_error(y, y_predicted)
    print(f"Mean squared error: {mse:.2f}")

    # The mean absolute error
    mae = mean_absolute_error(y, y_predicted)
    print(f"Mean absolute error: {mae:.2f}")

    # The coefficient of determination: 1 is perfect prediction (how much variance in the target variable)
    r2 = r2_score(y, y_predicted)
    print(f"Coefficient of determination: {r2:.2f}")

    # Plot outputs
    plt.scatter(X, y, color="black")
    plt.plot(X, y_predicted, color="blue", linewidth=3)
    plt.xticks(())
    plt.yticks(())

    if plot_filename:
        plt.savefig(plot_filename)
    plt.show()

    if r2 >= 0.4:
        print("****** Success ******")
    else:
        print("****** Failed ******")
    return r2
