import numpy as np
import pytest
from dbc import DiscreteBayesianClassifier
from sklearn.utils.validation import NotFittedError


def test_init():
    classifier = DiscreteBayesianClassifier(discretization_method="kmeans", random_state=42)
    assert classifier.discretization_method == "kmeans"
    assert classifier.random_state == 42
    assert classifier.label_encoder is None
    assert classifier.prior is None


def test_fit():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    classifier = DiscreteBayesianClassifier(discretization_method="kmeans", random_state=42)
    classifier.fit(X, y)
    assert classifier.label_encoder is not None
    assert classifier.prior is not None
    assert hasattr(classifier, "discretization_model")
    assert hasattr(classifier, "cluster_centers_")
    assert hasattr(classifier, "p_hat")


def test_predict_before_fit():
    X = np.array([[1, 2], [3, 4]])
    classifier = DiscreteBayesianClassifier(discretization_method="kmeans", random_state=42)
    with pytest.raises(NotFittedError):
        classifier.predict(X, prior=np.array([0.5, 0.5]), loss_function=np.array([[0, 1], [1, 0]]))


def test_predict():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y_train = np.array([0, 1, 0, 1, 0])
    X_test = np.array([[2, 3], [8, 9]])
    classifier = DiscreteBayesianClassifier(discretization_method="kmeans", random_state=42)
    classifier.fit(X_train, y_train)

    # Mock prior and loss_function for testing
    prior = np.array([0.5, 0.5])
    loss_function = np.array([[0, 1], [1, 0]])

    predictions = classifier.predict(X_test, prior=prior, loss_function=loss_function)
    assert len(predictions) == len(X_test)
    assert set(predictions).issubset(set(y_train))


def test_invalid_loss_function_in_predict():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y_train = np.array([0, 1, 0, 1, 0])
    X_test = np.array([[2, 3], [8, 9]])
    classifier = DiscreteBayesianClassifier(discretization_method="kmeans", random_state=42)
    classifier.fit(X_train, y_train)

    prior = np.array([0.5, 0.5])

    with pytest.raises(ValueError):
        classifier.predict(X_test, prior=prior, loss_function=None)
