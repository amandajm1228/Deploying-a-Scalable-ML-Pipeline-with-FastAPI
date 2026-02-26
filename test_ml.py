import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.model import compute_model_metrics, inference, train_model

# TODO: implement the first test. Change the function name and input as needed
def test_train_model_returns_random_forest():
    """
    Verify that train_model returns a fitted RandomForestClassifier
    instance when presented with valid training data.
    """
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_inference_returns_array_with_expected_length():
    """
    Verify that inference returns a NumPy array and the number of predictions
    matches the number of input rows.
    """
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y_train = np.array([0, 1, 0, 1])
    X_test = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_returns_bounded_values():
    """
    Verify that compute_model_metrics returns float values for Precision, Recall, and F1
    and that the values are bounded between 0 and 1.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
