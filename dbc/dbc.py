from numbers import Real, Integral

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._param_validation import Interval, StrOptions, RealNotInt
from sklearn.utils.validation import check_is_fitted, check_array


class DiscreteBayesianClassifier(BaseEstimator, ClassifierMixin):
    _parameter_constraints = {
        "discretization_method": [StrOptions({"kmeans", "dt", "fcm", "kmeans-fcm"})],
        "discretization_params": [
            {
                "kmeans": {
                    "n_clusters": [Interval(Integral, 1, None, closed="left")],
                    "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
                    "n_init": [
                        StrOptions({"auto"}),
                        Interval(Integral, 1, None, closed="left"),
                    ],
                    "max_iter": [Interval(Integral, 1, None, closed="left")],
                    "tol": [Interval(Real, 0, None, closed="left")],
                    "verbose": ["verbose"],
                    "random_state": ["random_state"],
                    "copy_x": ["boolean"],
                    "algorithm": [StrOptions({"lloyd", "elkan"})],
                },
                "dt": {
                    "splitter": [StrOptions({"best", "random"})],
                    "max_depth": [Interval(Integral, 1, None, closed="left"), None],
                    "min_samples_split": [
                        Interval(Integral, 2, None, closed="left"),
                        Interval(RealNotInt, 0.0, 1.0, closed="right"),
                    ],
                    "min_samples_leaf": [
                        Interval(Integral, 1, None, closed="left"),
                        Interval(RealNotInt, 0.0, 1.0, closed="neither"),
                    ],
                    "min_weight_fraction_leaf": [Interval(Real, 0.0, 0.5, closed="both")],
                    "max_features": [
                        Interval(Integral, 1, None, closed="left"),
                        Interval(RealNotInt, 0.0, 1.0, closed="right"),
                        StrOptions({"sqrt", "log2"}),
                        None,
                    ],
                    "random_state": ["random_state"],
                    "max_leaf_nodes": [Interval(Integral, 2, None, closed="left"), None],
                    "min_impurity_decrease": [Interval(Real, 0.0, None, closed="left")],
                    "ccp_alpha": [Interval(Real, 0.0, None, closed="left")],
                    "monotonic_cst": ["array-like", None],
                    "criterion": [StrOptions({"gini", "entropy", "log_loss"})],
                    "class_weight": [dict, list, StrOptions({"balanced"}), None],
                },
                "fcm": {
                    "n_clusters": Interval(Integral, 1, None, closed="left"),
                    "fuzzifier": Interval(Real, 1, None, closed="neither"),
                    "tol": Interval(Real, 0, None, closed="neither"),
                    "max_iter": Interval(Integral, 1, None, closed="left")
                }
            }
        ]
    }

    def __init__(
            self,
            *,
            discretization_method="kmeans",
            discretization_params=None,
            random_state=None
    ):
        self.label_encoder = None
        self.prior = None
        self.random_state = random_state
        self.discretization_method = discretization_method

        # 这里设置每种方法的默认值
        default_discretization_params = {
            "kmeans": {"n_clusters": 5, "algorithm": "lloyd"},
            "dt": {"criterion": "gini", "min_samples_split": 2},
            "fcm": {"n_clusters": 3, "fuzzifier": 1.5, "tol": 1e-4, "max_iter": 300},
        }
        self.discretization_params = {
            **default_discretization_params.get(discretization_method, {}),
            **(discretization_params or {}),
        }

    def fit(self, X, y):
        n_classes = len(set(y))
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.prior = compute_prior(y_encoded, n_classes)
        self._fit_discretization(X, y_encoded, n_classes)

    def _fit_discretization(self, X, y, n_classes):
        if self.discretization_method == "kmeans":
            self.discretization_model = KMeans(
                random_state=self.random_state,
                **self.discretization_params  # 字典解包
            )
            self.discretization_model.fit(X)
            self.cluster_centers_ = self.discretization_model.cluster_centers_
            self.p_hat = compute_p_hat(self.discretization_model.labels_, y, n_classes,
                                       self.discretization_model.n_clusters)

    def predict(self, X, prior=None, loss_function=None):
        check_is_fitted(self, ['p_hat', 'prior'])
        X = check_array(X)  # 检查 X 是否为合法的数组或类似数组的结构
        if prior is None:
            prior = self.prior
        if loss_function is None:
            raise ValueError("The parameter 'L' should not be None")

        if self.discretization_method == "kmeans":
            discrete_profiles = self.discretization_model.predict(X)
            return self.label_encoder.inverse_transform(
                predict_profile_label(prior, self.p_hat, loss_function)[discrete_profiles]
            )


def compute_p_hat(profile_labels: np.ndarray, y: np.ndarray, n_classes: int, n_clusters: int):
    """
    Compute the probability estimates for each class and cluster. The function calculates the relative frequency of
    each cluster label within each class and returns these probabilities as a matrix.

    :param profile_labels: Array containing cluster labels for each sample.
    :type profile_labels: np.ndarray
    :param y: Array containing class labels for each sample.
    :type y: np.ndarray
    :param n_classes: Number of distinct classes.
    :type n_classes: int
    :param n_clusters: Number of distinct clusters.
    :type n_clusters: int
    :return: A matrix where each row represents a class and each column represents a cluster. Each entry contains the
             probability estimate of the cluster for the respective class.
    :rtype: np.ndarray
    """
    p_hat = np.zeros((n_classes, n_clusters))

    for k in range(n_classes):
        indices_of_class_k = np.where(y == k)[0]
        nk = len(indices_of_class_k)
        p_hat[k] = np.bincount(profile_labels[indices_of_class_k], minlength=n_clusters) / nk
        # Count number of occurrences of each value in array of non-negative ints.
    return p_hat


def compute_prior(y: np.ndarray, n_classes: int):
    """
    Compute the prior probabilities for each class.

    This function calculates the prior probabilities for each class k in the
    range [0, n_classes-1]. It returns a numpy array where each element
    represents the proportion of occurrences of the class in the input array y.

    :param y: A numpy array of class labels.
    :type y: np.ndarray
    :param n_classes: Total number of unique classes.
    :type n_classes: int
    :return: A numpy array of prior probabilities.
    :rtype: np.ndarray
    """
    pi = np.zeros(n_classes)
    total_count = len(y)

    for k in range(n_classes):
        pi[k] = np.sum(y == k) / total_count
    return pi


def predict_profile_label(prior, p_hat, loss_function):
    """
    Predict the profile label based on prior probabilities, the predicted
    probabilities, and a loss function.

    This function calculates the class risk using the provided prior
    probabilities, predicted probabilities, and the loss function, and then
    returns the label with the minimum risk.

    :param prior: Array of prior probabilities for each class.
    :type prior: np.ndarray
    :param p_hat: Matrix of predicted probabilities for each class and instance.
    :type p_hat: np.ndarray
    :param loss_function: Matrix of loss values for each class combination.
    :type loss_function: np.ndarray
    :return: Array of predicted labels for each instance.
    :rtype: np.ndarray
    """
    class_risk = (prior.reshape(-1, 1) * loss_function).T @ p_hat
    l_predict = np.argmin(class_risk, axis=0)
    return l_predict


def compute_conditional_risk(y_true: np.ndarray, y_pred: np.ndarray, loss_function: np.ndarray):
    """
    Computes the conditional risk and the normalized confusion matrix for given true labels,
    predicted labels, and a loss function.

    This function uses label encoding to transform string labels into integer codes. It then
    calculates the confusion matrix and normalizes it. Finally, it computes the conditional
    risk by multiplying the loss function with the normalized confusion matrix and summing the
    resulting products.

    :param y_true:
        The true labels of the data as a NumPy array.
    :param y_pred:
        The predicted labels as a NumPy array.
    :param loss_function:
        A loss function represented as a NumPy array.
    :return:
        A tuple containing the conditional risk and the normalized confusion matrix.
    """

    # 使用LabelEncoder将字符串标签转换为整数编码
    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)

    # 计算混淆矩阵
    confusion_matrix_normalized = confusion_matrix(y_true_encoded, y_pred_encoded, normalize='true')

    # 计算条件风险
    conditional_risk = np.sum(np.multiply(loss_function, confusion_matrix_normalized), axis=1)

    return conditional_risk, confusion_matrix_normalized
