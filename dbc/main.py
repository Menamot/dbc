from numbers import Real, Integral

import numpy as np
import skfuzzy as fuzz
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import TensorDataset, DataLoader

from dbc.utils import (
    compute_piStar,
    compute_prior,
    compute_p_hat,
    predict_profile_label,
    discretize_features,
    compute_p_hat_with_degree,
    compute_prob,
    compute_b_risk,
    compute_SPDBC_pi_star,
    # compute_p_hat_with_soft_labels
)


class BaseDiscreteBayesianClassifier(BaseEstimator):
    """
    Base class for discrete Bayesian classifiers(DBC) and discrete minimax classifiers(DMC).

    This class provides a framework for implementing DBC and DMC . It defines core methods for fitting
    the model to data, making predictions, and calculating probabilities.
    Subclasses should implement the abstract methods for specific model
    behaviors such as discretization of features and prediction logic.

    Attributes
    ----------
    prior: ndarray of shape (n_classes, )
        Estimated prior probabilities for each class.

    loss_function: ndarray of shape (n_classes, n_classes)
        A matrix of loss function values for each class.

    p_hat: ndarray of shape (n_classes, n_clusters)
        Estimated probabilities for each class label in different profiles.

    prior_attribute: {'prior', 'prior_star'} str
        The attribute used to store prior probabilities. In Discrete Bayesian
        classifier it will be 'prior' and in Discrete Minimax classifier it will
        be 'prior_star'.

    """

    def __init__(self):
        self.prior_attribute = None
        self.prior = None
        self.p_hat = None
        self.label_encoder = LabelEncoder()
        self.loss_function = None

    def fit(self, X, y, loss_function="01"):
        """
        Fits the model according to the given training data. The method initializes
        class-specific attributes, encodes the target labels, and computes the
        prior probabilities. It also prepares the discretization of feature
        values based on the encoded labels and specified loss function.
        Currently, only the "01" loss function is supported, which penalizes
        misclassifications uniformly across classes.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.

        y: {array-like, sparse matrix} of shape (n_samples, )
            Labels for training instances.

        loss_function: {'01'}, callable or array-like of shape (n_clusters, n_features), default='01'
            Loss function for calculating class risk. Currently, only the "01" loss function is supported.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(set(y_encoded))
        if loss_function == "01":
            self.loss_function = np.ones((n_classes, n_classes)) - np.eye(n_classes)
        else:
            raise ValueError("Invalid loss_function")
        self.prior = compute_prior(y_encoded, n_classes)
        self._fit_discretization(X, y_encoded, n_classes)

    def predict(self, X, prior_pred=None):
        """
        Predicts the label for the given input data using either the
        prior prediction provided or a previously stored attribute.

        Parameters
        ----------
        X : array-like
            Input data for which predictions are to be made.

        prior_pred : ndarray of shape (n_classes,) or None, default=None
            Prior predictions to be used in conjunction with `X`. If
            None, the method uses the model's `prior_attribute`.

        Returns
        -------
        labels: ndarray of shape (n_samples,)
            Predicted labels corresponding to the input data `X`.
        """
        check_is_fitted(self, ["p_hat", self.prior_attribute])
        if prior_pred is None:
            prior_pred = getattr(self, self.prior_attribute)
        labels = self.label_encoder.inverse_transform(
            self._predict_profiles(X, prior_pred)
        )
        return labels

    def predict_prob(self, X, prior_pred=None):
        """
        Predict the probabilities for a given set of samples, `X`, using either the
        prior prediction provided or a previously stored attribute.

        Parameters
        ----------
        X : array-like
            Input data for which predictions are to be made.

        prior_pred : ndarray of shape (n_classes,) or None, default=None
            Prior predictions to be used in conjunction with `X`. If
            None, the method utilizes the model's `prior_attribute`.

        Returns
        -------
        probability: ndarray of shape (n_samples, n_classes)
            Predicted probabilities corresponding to the input data `X`.
        """
        check_is_fitted(self, ["p_hat", self.prior_attribute])
        if prior_pred is None:
            prior_pred = getattr(self, self.prior_attribute)
        probability = self._predict_probabilities(X, prior_pred)
        return probability

    def _fit_discretization(self, X, y, n_classes):
        raise NotImplementedError

    def _transform_to_discrete_profiles(self, X):
        raise NotImplementedError

    def _predict_profiles(self, X, prior):
        raise NotImplementedError

    def _predict_probabilities(self, X, prior):
        raise NotImplementedError


class ClasswiseKMeans:
    def __init__(self, n_clusters_per_class=3, random_state=None):
        """
        n_clusters_per_class: int 或 dict
            若为 int，则所有类别使用相同的簇数；
            若为 dict，则每个类别单独指定簇数，如 {0:3, 1:5}
        """
        self.n_clusters_per_class = n_clusters_per_class
        self.random_state = random_state
        self.centers_ = None
        self.center_labels_ = None
        self.labels_ = None
        self.n_clusters = None

    def fit(self, X, y):
        classes = np.unique(y)
        centers = []
        center_labels = []

        for c in classes:
            Xc = X[y == c]
            # 若 n_clusters_per_class 是 dict，则取对应的簇数
            if isinstance(self.n_clusters_per_class, dict):
                k = self.n_clusters_per_class.get(c, 3)
            else:
                k = self.n_clusters_per_class

            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(Xc)

            centers.append(kmeans.cluster_centers_)
            center_labels.extend([c] * k)

        # 合并所有类别的簇中心
        self.centers_ = np.vstack(centers)
        self.center_labels_ = np.array(center_labels)
        self.n_clusters = len(self.centers_)

        # 🔁 重新计算所有样本的最近簇编号（全局一致）
        dists = np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=2)
        self.labels_ = np.argmin(dists, axis=1)

        return self

    def predict(self, X):
        # 计算每个样本到所有簇中心的欧氏距离
        dists = np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=2)
        # 最近中心的全局簇编号
        nearest_idx = np.argmin(dists, axis=1)
        return nearest_idx


class _KmeansDiscretization(BaseDiscreteBayesianClassifier, KMeans):
    """
    Handles the discretization of continuous features for DBC and DMC
    using KMeans partitioning and provides functionality to fit, transform, and predict
    using the discretized data profiles.

    This class integrates KMeans partitioning to transform continuous variables into
    discrete profiles, which are then utilized for DBC and DMC tasks.
    It extends both BaseDiscreteBayesianClassifier and KMeans classes to inherit
    their functionalities and combines them to generate a discrete form suitable
    for classification models.
    """

    def __init__(
        self,
        n_clusters,
        *,
        init,
        n_init,
        max_iter,
        tol,
        verbose,
        random_state,
        copy_x,
        algorithm,
        box,
        classes_wise,
    ):
        self.classes_wise = classes_wise
        if classes_wise is False:
            KMeans.__init__(
                self,
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                random_state=random_state,
                copy_x=copy_x,
                algorithm=algorithm,
            )
            self.box = box
            BaseDiscreteBayesianClassifier.__init__(self)
        else:
            ClasswiseKMeans.__init__(
                self, n_clusters_per_class=n_clusters, random_state=random_state
            )
            self.box = box

            BaseDiscreteBayesianClassifier.__init__(self)

    def _fit_discretization(self, X, y, n_classes):
        """
        Fits a discretization model to the input data.

        This method initializes and fits a `KMeans` model to the provided feature
        matrix `X` and computes p_hat, the estimated probabilities for each class
        in different profiles. If prior_attribute is 'prior_star', it also computes
        prior_star, the best prior probability that minimizes the maximum class risk.
        """

        if self.classes_wise is False:
            self.discretization_model = KMeans(
                n_clusters=self.n_clusters,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                random_state=self.random_state,
                copy_x=self.copy_x,
                algorithm=self.algorithm,
            )
            self.discretization_model.fit(X)
            self.cluster_centers = self.discretization_model.cluster_centers_
            self.p_hat = compute_p_hat(
                self.discretization_model.labels_,
                y,
                n_classes,
                self.discretization_model.n_clusters,
            )
        else:
            self.discretization_model = ClasswiseKMeans(
                n_clusters_per_class=self.n_clusters_per_class,
                random_state=self.random_state,
            )
            self.discretization_model.fit(X, y)
            self.cluster_centers = self.discretization_model.centers_
            self.p_hat = compute_p_hat(
                self.discretization_model.labels_,
                y,
                n_classes,
                self.discretization_model.n_clusters,
            )
        if self.prior_attribute == "prior_star":
            self.prior_star = compute_piStar(
                self.p_hat, y, n_classes, self.loss_function, 1000, self.box
            )[0]

    def _transform_to_discrete_profiles(self, X):
        """
        Convert continuous data into discrete profiles using a discretization model.
        """
        return self.discretization_model.predict(X)

    def _predict_profiles(self, X, prior):
        """
        Predict the class labels for a given set of profiles using the prior distribution.

        This method first transforms the profiles into a discrete representation, then utilizes the label encoder to
        provide the predicted class labels.
        """
        discrete_profiles = self._transform_to_discrete_profiles(X)
        return predict_profile_label(prior, self.p_hat, self.loss_function)[
            discrete_profiles
        ]

    def _predict_probabilities(self, X, prior):
        """
        Compute probabilities for each class based on input features and prior
        """
        class_risk = (prior.reshape(-1, 1) * self.loss_function).T @ self.p_hat
        prob = 1 / (class_risk + 1e-6)
        prob /= np.sum(prob, axis=0, keepdims=True)
        return prob[:, self._transform_to_discrete_profiles(X)].T


class KmeansDiscreteBayesianClassifier(_KmeansDiscretization):
    """
    A discrete Bayesian classifier base on K-Means partitioning, all parameters are inherited from the KMeans.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.

        For an example of how to choose an optimal value for `n_clusters` refer to
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_silhouette_analysis.py`.

    init : {'k-means++', 'random'}, callable or array-like of shape
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        * 'k-means++' : selects initial cluster centroids using sampling
            based on an empirical probability distribution of the points'
            contribution to the overall inertia. This technique speeds up
            convergence. The algorithm implemented is "greedy k-means++". It
            differs from the vanilla k-means++ by making several trials at
            each sampling step and choosing the best centroid among them.

        * 'random': choose `n_clusters` observations (rows) at random from
        data for the initial centroids.

        * If an array is passed, it should be of shape (n_clusters, n_features)
        and give the initial centers.

        * If a callable is passed, it should take arguments X, n_clusters, and a
        random state and return an initialization.

        For an example of how to use the different `init` strategy, see the example
        entitled :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_digits.py`.

    n_init : 'auto' or int, default='auto'
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regard to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance, or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True,
        When pre-computing distances, it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters by using the triangle inequality. However, it's
        more memory-intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.


    Attributes
    ----------
    prior: ndarray of shape (n_classes,)
        Estimated prior probabilities for each class.

    loss_function: ndarray of shape (n_classes, n_classes)
        A matrix of loss function values for each class.

    p_hat: ndarray of shape (n_classes, n_clusters)
        Estimated probabilities for each class label in different profiles.

    prior_attribute: {'prior', 'prior_star'} str
        The attribute used to store prior probabilities. In Discrete Bayesian
        classifier it will be 'prior' and in Discrete Minimax classifier it will
        be 'prior_star'.

    cluster_centers : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    box: ndarray or None
        box constrain for calculating prior_star. Will be equal to None in
        discrete Bayesian classifier.

    prior_star: ndarray of shape (n_classes,) or None
        Best prior probability that minimize the maximum class risk. Will be equal
        to None in discrete Bayesian classifier. More details in "C. Gilet, S. Barbosa,
        and L. Fillatre, “Discrete Box-Constrained Minimax Classifier for Uncertain
        and Imbalanced Class Proportions,” IEEE Trans. Pattern Anal. Mach. Intell.,
        vol. 44, no. 6, pp. 2923–2937, Jun. 2022, doi: 10.1109/TPAMI.2020.3046439".

    discretization_model: KMeans object
        The Kmeans model used for discretization.
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
        classes_wise=False,
    ):
        _KmeansDiscretization.__init__(
            self,
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
            box=None,
            classes_wise=classes_wise,
        )
        self.prior_attribute = "prior"


class KmeansDiscreteMinmaxClassifier(_KmeansDiscretization):
    """
    A discrete Minimax classifier base on K-Means partitioning, all parameters are inherited from the KMeans.

    This classifier base on KmeansBayesianClassifier and calculates prior_star to minimize the maximum class risk.
    Which means it will try to balance risk of different classes.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

        For an example of how to choose an optimal value for `n_clusters` refer to
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_silhouette_analysis.py`.

    init : {'k-means++', 'random'}, callable or array-like of shape
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        * 'k-means++' : selects initial cluster centroids using sampling
            based on an empirical probability distribution of the points'
            contribution to the overall inertia. This technique speeds up
            convergence. The algorithm implemented is "greedy k-means++". It
            differs from the vanilla k-means++ by making several trials at
            each sampling step and choosing the best centroid among them.

        * 'random': choose `n_clusters` observations (rows) at random from
        data for the initial centroids.

        * If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        * If a callable is passed, it should take arguments X, n_clusters, and a
        random state and return an initialization.

        For an example of how to use the different `init` strategy, see the example
        entitled :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_digits.py`.

    n_init : 'auto' or int, default='auto'
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance in regard to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance, or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However, it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

    Attributes
    ----------
    prior: ndarray of shape (n_classes,)
        Estimated prior probabilities for each class.

    loss_function: ndarray of shape (n_classes, n_classes)
        A matrix of loss function values for each class.

    p_hat: ndarray of shape (n_classes, n_clusters)
        Estimated probabilities for each class label in different profiles.

    prior_attribute: {'prior', 'prior_star'} str
        The attribute used to store prior probabilities. In Discrete Bayesian
        classifier it will be 'prior' and in Discrete Minimax classifier it will
        be 'prior_star'.

    cluster_centers : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    box: ndarray or None
        box constrain for calculating prior_star. Will be equal to None in
        discrete Bayesian classifier.

    prior_star: ndarray of shape (n_classes,) or None
        Best prior probability that minimize the maximum class risk. Will be equal
        to None in discrete Bayesian classifier. More details in "C. Gilet, S. Barbosa,
        and L. Fillatre, “Discrete Box-Constrained Minimax Classifier for Uncertain
        and Imbalanced Class Proportions,” IEEE Trans. Pattern Anal. Mach. Intell.,
        vol. 44, no. 6, pp. 2923–2937, Jun. 2022, doi: 10.1109/TPAMI.2020.3046439".

    discretization_model: KMeans object
        The Kmeans model used for discretization.
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
        box=None,
        classes_wise=False,
    ):
        _KmeansDiscretization.__init__(
            self,
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
            box=box,
            classes_wise=classes_wise,
        )
        self.prior_attribute = "prior_star"


class _DecisionTreeDiscretization(
    BaseDiscreteBayesianClassifier, DecisionTreeClassifier
):
    """
    Handles the discretization of continuous features for DBC and DMC
    using Decision Tree partitioning and provides functionality to fit, transform, and predict
    using the discretized data profiles.

    This class integrates Decision Tree partitioning to transform continuous variables into
    discrete profiles, which are then utilized for DBC and DMC tasks.
    It extends both BaseDiscreteBayesianClassifier and DecisionTreeClassifier classes to inherit
    their functionalities and combines them to generate a discrete form suitable
    for classification models.
    """

    def __init__(
        self,
        *,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        random_state,
        max_leaf_nodes,
        min_impurity_decrease,
        class_weight,
        ccp_alpha,
        monotonic_cst,
        box,
    ):
        DecisionTreeClassifier.__init__(
            self,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            monotonic_cst=monotonic_cst,
            ccp_alpha=ccp_alpha,
        )
        self.box = box
        BaseDiscreteBayesianClassifier.__init__(self)

    def _fit_discretization(self, X, y, n_classes):
        """
        Fits a discretization model to the input data.

        This method initializes and fits a `DecisionTree` model to the provided feature
        matrix `X` and computes p_hat, the estimated probabilities for each class
        in different profiles. If prior_attribute is 'prior_star', it also computes
        prior_star, the best prior probability that minimize the maximum class risk.
        """
        self.discretization_model = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            class_weight=self.class_weight,
            random_state=self.random_state,
            min_impurity_decrease=self.min_impurity_decrease,
            monotonic_cst=self.monotonic_cst,
            ccp_alpha=self.ccp_alpha,
        )
        self.discretization_model.fit(X, y)
        self.p_hat = compute_p_hat(
            discretize_features(X, self.discretization_model),
            y,
            n_classes,
            self.discretization_model.get_n_leaves(),
        )
        if self.prior_attribute == "prior_star":
            self.prior_star = compute_piStar(
                self.p_hat, y, n_classes, self.loss_function, 1000, self.box
            )[0]

    def _transform_to_discrete_profiles(self, X):
        """
        Convert continuous data into discrete profiles using a discretization model.
        """
        return discretize_features(X, self.discretization_model)

    def _predict_profiles(self, X, prior):
        """
        Predict the class labels for a given set of profiles using the prior
        distribution.

        This method first transforms the profiles into a discrete
        representation, then utilizes the label encoder to provide the predicted
        class labels.
        """
        discrete_profiles = self._transform_to_discrete_profiles(X)
        return predict_profile_label(prior, self.p_hat, self.loss_function)[
            discrete_profiles
        ]

    def _predict_probabilities(self, X, prior):
        """
        Compute probabilities for each class based on input features and prior
        """
        class_risk = (prior.reshape(-1, 1) * self.loss_function).T @ self.p_hat
        prob = 1 - (class_risk / np.sum(class_risk, axis=0))
        return prob[:, self._transform_to_discrete_profiles(X)].T


class DecisionTreeDiscreteBayesianClassifier(_DecisionTreeDiscretization):
    """
    A discrete Bayesian classifier base on Decision Tree partitioning, all parameters
    are inherited from the DecisionTreeClassifier.

    This classifier base on DecisionTreeBayesianClassifier and calculates prior_star
    to minimize the maximum class risk.Which means it will try to balance risk of
    different classes.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `max(1, int(max_features * n_features_in_))` features are considered at
              each split.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance, or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for:
          - multiclass classifications (i.e. when `n_classes > 2`),
          - multioutput classifications (i.e. when `n_outputs_ > 1`),
          - classifications trained on data with missing values.

        The constraints hold over the probability of the positive class.

    Attributes
    ----------
    prior: ndarray of shape (n_classes,)
        Estimated prior probabilities for each class.

    loss_function: ndarray of shape (n_classes, n_classes)
        A matrix of loss function values for each class.

    p_hat: ndarray of shape (n_classes, n_clusters)
        Estimated probabilities for each class label in different profiles.

    prior_attribute: {'prior', 'prior_star'} str
        The attribute used to store prior probabilities. In Discrete Bayesian
        classifier it will be 'prior' and in Discrete Minimax classifier it will
        be 'prior_star'.

    box: ndarray or None
        box constrain for calculating prior_star. Will be equal to None in
        discrete Bayesian classifier.

    prior_star: ndarray of shape (n_classes,) or None
        Best prior probability that minimize the maximum class risk. Will be equal
        to None in discrete Bayesian classifier. More details in "C. Gilet, S. Barbosa,
        and L. Fillatre, “Discrete Box-Constrained Minimax Classifier for Uncertain
        and Imbalanced Class Proportions,” IEEE Trans. Pattern Anal. Mach. Intell.,
        vol. 44, no. 6, pp. 2923–2937, Jun. 2022, doi: 10.1109/TPAMI.2020.3046439".

    discretization_model: DecisionTreeClassifier object
        The DecisionTree model used for discretization.
    """

    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
    ):
        _DecisionTreeDiscretization.__init__(
            self,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            monotonic_cst=monotonic_cst,
            ccp_alpha=ccp_alpha,
            box=None,
        )
        self.prior_attribute = "prior"


class DecisionTreeDiscreteMinimaxClassifier(_DecisionTreeDiscretization):
    """
    A discrete Bayesian classifier base on Decision Tree partitioning, all parameters
    are inherited from the DecisionTreeClassifier.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `max(1, int(max_features * n_features_in_))` features are considered at
              each split.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance, or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for:
          - multiclass classifications (i.e. when `n_classes > 2`),
          - multioutput classifications (i.e. when `n_outputs_ > 1`),
          - classifications trained on data with missing values.

        The constraints hold over the probability of the positive class.

    Attributes
    ----------
    prior: ndarray of shape (n_classes,)
        Estimated prior probabilities for each class.

    loss_function: ndarray of shape (n_classes, n_classes)
        A matrix of loss function values for each class.

    p_hat: ndarray of shape (n_classes, n_clusters)
        Estimated probabilities for each class label in different profiles.

    prior_attribute: {'prior', 'prior_star'} str
        The attribute used to store prior probabilities. In Discrete Bayesian
        classifier it will be 'prior' and in Discrete Minimax classifier it will
        be 'prior_star'.

    box: ndarray or None
        box constrain for calculating prior_star. Will be equal to None in
        discrete Bayesian classifier.

    prior_star: ndarray of shape (n_classes,) or None
        Best prior probability that minimize the maximum class risk. Will be equal
        to None in discrete Bayesian classifier. More details in "C. Gilet, S. Barbosa,
        and L. Fillatre, “Discrete Box-Constrained Minimax Classifier for Uncertain
        and Imbalanced Class Proportions,” IEEE Trans. Pattern Anal. Mach. Intell.,
        vol. 44, no. 6, pp. 2923–2937, Jun. 2022, doi: 10.1109/TPAMI.2020.3046439".

    discretization_model: DecisionTreeClassifier object
        The DecisionTree model used for discretization.

    """

    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
        box=None,
    ):
        _DecisionTreeDiscretization.__init__(
            self,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            monotonic_cst=monotonic_cst,
            ccp_alpha=ccp_alpha,
            box=box,
        )
        self.prior_attribute = "prior_star"


class _CmeansDiscretization(BaseDiscreteBayesianClassifier):
    _parameter_constraints = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "fuzzifier": [Interval(Real, 1, None, closed="neither")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "init": [callable, "array-like", None],
        "metric": [StrOptions({"euclidean", "cityblock", "minkowski"}), callable],
    }

    def __init__(
        self,
        n_clusters,
        fuzzifier,
        *,
        tol,
        max_iter,
        init,
        cluster_centers,
        metric,
        random_state,
        use_kmeans=False,
    ):
        BaseDiscreteBayesianClassifier.__init__(self)
        self.n_clusters = n_clusters
        self.fuzzifier = fuzzifier
        self.init = init
        self.cluster_centers = cluster_centers
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric
        self.random_state = random_state
        self.use_kmeans = use_kmeans
        self._validate_params()

    def _fit_discretization(self, X, y, n_classes):
        if self.use_kmeans:
            self.discretization_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            self.discretization_model.fit(X)
            self.cluster_centers = self.discretization_model.cluster_centers_
            self.membership_degree, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                X.T,
                cntr_trained=self.cluster_centers,
                m=self.fuzzifier,
                error=self.tol,
                maxiter=self.max_iter,
                metric=self.metric,
                init=self.init,
                seed=self.random_state,
            )
        elif self.cluster_centers is None:
            self.cluster_centers, self.membership_degree, _, _, _, _, _ = (
                fuzz.cluster.cmeans(
                    X.T,
                    c=self.n_clusters,
                    m=self.fuzzifier,
                    error=self.tol,
                    maxiter=self.max_iter,
                    metric=self.metric,
                    init=self.init,
                    seed=self.random_state,
                )
            )
        else:
            self.membership_degree, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                X.T,
                cntr_trained=self.cluster_centers,
                m=self.fuzzifier,
                error=self.tol,
                maxiter=self.max_iter,
                metric=self.metric,
                init=self.init,
                seed=self.random_state,
            )
        self.p_hat = compute_p_hat_with_degree(self.membership_degree, y, n_classes)
        if self.prior_attribute == "prior_star":
            self.prior_star = compute_SPDBC_pi_star(
                X,
                y,
                self.loss_function,
                self.p_hat,
                self.membership_degree,
                self.prior,
                alpha=2,
                n_iter=300,
                eps=1e-6,
            )

    def _predict_probabilities(self, X, prior):

        membership_degree_pred, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            X.T,
            cntr_trained=self.cluster_centers,
            m=self.fuzzifier,
            error=self.tol,
            maxiter=self.max_iter,
        )
        # prob = compute_posterior(membership_degree_pred, self.p_hat, prior, self.loss_function)
        prob = compute_prob(
            self.membership_degree, membership_degree_pred, self.p_hat, prior
        )
        return prob

    # def _predict_profiles(self, X, prior):
    #     prob = self._predict_probabilities(X, prior)
    #     return np.argmax(prob, axis=1)
    def _predict_profiles(self, X, prior):
        membership_degree_pred, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            X.T,
            cntr_trained=self.cluster_centers,
            m=self.fuzzifier,
            error=self.tol,
            maxiter=self.max_iter,
        )
        risk = compute_b_risk(
            membership_degree_pred, self.p_hat, prior, self.loss_function
        )
        return np.argmin(risk, axis=1)

    # def _predict_profiles(self, X, prior):
    #     prob = self._predict_probabilities(X, prior)
    #     # 对每一行样本，根据概率分布随机选择一个类别
    #     predictions = np.array([
    #         np.random.choice(len(p), p=p)
    #         for p in prob
    #     ])
    #     return predictions


class CmeansDiscreteBayesianClassifier(_CmeansDiscretization):
    """
    A discrete Bayesian classifier base on Fuzzy C-Means partitioning, all parameters are
    inherited from the fuzz.cluster.cmeans() function.

    Parameters
    ----------
    n_clusters : int
        Desired number of clusters or classes.

    fuzzifier : float
        The degree of fuzziness of membership determines the influence of the cluster
        center on the membership of the data point. Should be greater than 1.

    tol : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error
        .
    max_iter : int
        Maximum number of iterations allowed.

    metric: string
        By default is set to Euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.

    init : ndarray of shape (n_clusters, n_samples)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.

    random_state : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.

    Attributes
    ----------
    prior: ndarray of shape (n_classes,)
        Estimated prior probabilities for each class.

    loss_function: ndarray of shape (n_classes, n_classes)
        A matrix of loss function values for each class.

    p_hat: ndarray of shape (n_classes, n_clusters)
        Estimated probabilities for each class label in different profiles.

    prior_attribute: {'prior', 'prior_star'} str
        The attribute used to store prior probabilities. In Discrete Bayesian
        classifier it will be 'prior' and in Discrete Minimax classifier it will
        be 'prior_star'.

    cluster_centers : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    """

    def __init__(
        self,
        n_clusters=8,
        fuzzifier=1.5,
        *,
        tol=1e-4,
        max_iter=300,
        init=None,
        cluster_centers=None,
        metric="euclidean",
        random_state=None,
        use_kmeans=False,
    ):
        _CmeansDiscretization.__init__(
            self,
            n_clusters=n_clusters,
            fuzzifier=fuzzifier,
            tol=tol,
            max_iter=max_iter,
            init=init,
            cluster_centers=cluster_centers,
            metric=metric,
            random_state=random_state,
            use_kmeans=use_kmeans,
        )
        self.prior_attribute = "prior"

    # def em(self, X_source, X_target, y_source):
    #     n_source = len(y_source)
    #     n_target = X_target.shape[0]
    #     alpha = n_source / (n_source + n_target)
    #     membership_degree_target, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
    #         X_target.T,
    #         cntr_trained=self.cluster_centers,
    #         m=self.fuzzifier,
    #         error=self.tol,
    #         maxiter=self.max_iter,
    #     )
    #     posterior_iter_source = compute_prob(
    #         self.membership_degree, self.membership_degree, self.p_hat, self.prior
    #     )
    #     posterior_iter_target = compute_prob(
    #         self.membership_degree, membership_degree_target, self.p_hat, self.prior
    #     )
    #     for i in range(100):
    #         prior_iter = (
    #             alpha
    #             * np.sum(posterior_iter_source, axis=0)
    #             / posterior_iter_source.shape[0]
    #         ) + (
    #             (1 - alpha)
    #             * np.sum(posterior_iter_target, axis=0)
    #             / posterior_iter_target.shape[0]
    #         )
    #         p_hat_iter = (
    #             alpha
    #             * (
    #                 np.sum(posterior_iter_source, axis=0)
    #                 / posterior_iter_source.shape[0]
    #             )[:, np.newaxis]
    #             * compute_p_hat_with_soft_labels(
    #                 self.membership_degree, posterior_iter_source
    #             )
    #         ) + (
    #             (1 - alpha)
    #             * (
    #                 np.sum(posterior_iter_target, axis=0)
    #                 / posterior_iter_target.shape[0]
    #             )[:, np.newaxis]
    #             * compute_p_hat_with_soft_labels(
    #                 membership_degree_target, posterior_iter_target
    #             )
    #         )
    #         posterior_iter_source = compute_prob(
    #             self.membership_degree, self.membership_degree, p_hat_iter, prior_iter
    #         )
    #         posterior_iter_target = compute_prob(
    #             self.membership_degree, membership_degree_target, p_hat_iter, prior_iter
    #         )
    #     return posterior_iter_target, p_hat_iter, prior_iter


class CmeansDiscreteMinmaxClassifier(_CmeansDiscretization):
    """
    A discrete Bayesian classifier base on Fuzzy C-Means partitioning, all parameters are
    inherited from the fuzz.cluster.cmeans() function.

    Parameters
    ----------
    n_clusters : int
        Desired number of clusters or classes.

    fuzzifier : float
        The degree of fuzziness of membership determines the influence of the cluster
        center on the membership of the data point. Should be greater than 1.

    tol : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error
        .
    max_iter : int
        Maximum number of iterations allowed.

    metric: string
        By default is set to Euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.

    init : ndarray of shape (n_clusters, n_samples)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.

    random_state : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.

    Attributes
    ----------
    prior: ndarray of shape (n_classes,)
        Estimated prior probabilities for each class.

    loss_function: ndarray of shape (n_classes, n_classes)
        A matrix of loss function values for each class.

    p_hat: ndarray of shape (n_classes, n_clusters)
        Estimated probabilities for each class label in different profiles.

    prior_attribute: {'prior', 'prior_star'} str
        The attribute used to store prior probabilities. In Discrete Bayesian
        classifier it will be 'prior' and in Discrete Minimax classifier it will
        be 'prior_star'.

    cluster_centers : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    """

    def __init__(
        self,
        n_clusters=8,
        fuzzifier=1.5,
        *,
        tol=1e-4,
        max_iter=300,
        init=None,
        cluster_centers=None,
        metric="euclidean",
        random_state=None,
        use_kmeans=False,
    ):
        _CmeansDiscretization.__init__(
            self,
            n_clusters=n_clusters,
            fuzzifier=fuzzifier,
            tol=tol,
            max_iter=max_iter,
            init=init,
            cluster_centers=cluster_centers,
            metric=metric,
            random_state=random_state,
            use_kmeans=use_kmeans,
        )
        self.prior_attribute = "prior_star"


import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminativeNN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_profiles,
        num_classes,
        p_y,
    ):
        super().__init__()

        self.num_profiles = num_profiles
        self.num_classes = num_classes
        self.p_y = p_y
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_profiles),
        )

        # 使用 register_buffer 保持状态但不被训练
        self.register_buffer(
            "count_z_given_y", torch.zeros(num_profiles, num_classes)
        )  # [T, K]

    def update_profile_counts(self, q_z_given_x, y_true):
        """
        使用 soft assignment 更新 profile 频数表：
        q_z_given_x: [B, T], y_true: [B]
        """
        with torch.no_grad():
            # B, T = q_z_given_x.shape
            K = self.num_classes
            one_hot_y = F.one_hot(y_true, K).float()  # [B, K]
            # outer product: for each sample, update count[z, y]
            # Resulting shape: [B, T, K]
            contrib = torch.einsum("bt,bk->btk", q_z_given_x, one_hot_y)
            self.count_z_given_y += contrib.sum(dim=0)  # sum over batch

    def compute_p_z_given_y(self):
        # 避免除以 0：加入平滑
        smoothed = self.count_z_given_y + 1e-3
        return smoothed / smoothed.sum(dim=0, keepdim=True)  # [T, K]

    def forward(self, x):
        # q(Z|X)
        profile_logits = self.encoder(x)
        q_z_given_x = F.softmax(profile_logits, dim=1)  # [B, T]

        # Profile 后验
        p_z_given_y = self.compute_p_z_given_y()  # [T, K]

        # 归一化（每行归一）
        p_z = (p_z_given_y * self.p_y.unsqueeze(0)).sum(dim=1)  # shape [T]

        # 构造完整贝叶斯项：P(Z|Y)*P(Y) / P(Z)
        bayes_matrix = (p_z_given_y * self.p_y.unsqueeze(0)) / (
            p_z.unsqueeze(1) + 1e-9
        )  # [T, K]

        # 推理公式：P(Y|X) = sum_z P(Y|Z=z) * P(Z=z|X)
        p_y_given_x = q_z_given_x @ bayes_matrix  # shape [B, K]
        return p_y_given_x, q_z_given_x, p_z_given_y

    def predict(self, x):
        p_y_given_x, _, _ = self.forward(x)
        return p_y_given_x


def profile_entropy_regularization(q_z_given_x):
    avg_profile = q_z_given_x.mean(dim=0) + 1e-9
    return -torch.sum(avg_profile * avg_profile.log())


def profile_kl_regularization(q_z_given_x):
    avg_profile = q_z_given_x.mean(dim=0) + 1e-9
    log_avg = avg_profile.log()
    log_uniform = torch.full_like(
        log_avg, fill_value=torch.log(torch.tensor(1.0 / len(avg_profile)))
    )
    return torch.sum(avg_profile * (log_avg - log_uniform))


class _DiscriminativeNNDiscretization(BaseDiscreteBayesianClassifier):

    def __init__(
        self,
        n_clusters,
        *,
        tol,
        max_iter,
        random_state,
        n_epochs,
        batch_size,
        kl_regularization,
    ):
        BaseDiscreteBayesianClassifier.__init__(self)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.kl_regularization = kl_regularization
        self.random_state = random_state
        # self._validate_params()

    def _fit_discretization(self, X, y, n_classes):
        input_dim = X.shape[1]
        hidden_dim = 64
        p_y = torch.tensor(compute_prior(y, n_classes), dtype=torch.float32)
        self.discretization_model = DiscriminativeNN(
            input_dim,
            hidden_dim,  # hidden_dim
            self.n_clusters,  # num_profiles
            n_classes,  # num_classes
            p_y,  # p_y
        )
        optimizer = torch.optim.Adam(self.discretization_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.n_epochs):
            # 训练阶段
            self.discretization_model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                p_y_given_x, q_z_given_x, _ = self.discretization_model(X_batch)
                self.discretization_model.update_profile_counts(q_z_given_x, y_batch)
                loss = criterion(
                    p_y_given_x, y_batch
                ) + self.kl_regularization * profile_kl_regularization(q_z_given_x)
                loss.backward()
                optimizer.step()

        self.discretization_model.eval()
        with torch.no_grad():
            self.membership_degree = self.discretization_model(X_tensor)[1].numpy().T
        self.p_hat = compute_p_hat_with_degree(self.membership_degree, y, n_classes)
        if self.prior_attribute == "prior_star":
            self.prior_star = compute_SPDBC_pi_star(
                X,
                y,
                self.loss_function,
                self.p_hat,
                self.membership_degree,
                self.prior,
                alpha=2,
                n_iter=300,
                eps=1e-6,
            )

    def _predict_probabilities(self, X, prior):
        device = next(self.discretization_model.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            membership_degree_pred = self.discretization_model(X_tensor)[1].numpy().T
        prob = compute_prob(
            self.membership_degree, membership_degree_pred, self.p_hat, prior
        )
        return prob

    def _predict_profiles(self, X, prior):
        device = next(self.discretization_model.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            membership_degree_pred = (
                self.discretization_model(X_tensor)[1].cpu().numpy().T
            )
        risk = compute_b_risk(
            membership_degree_pred, self.p_hat, prior, self.loss_function
        )
        return np.argmin(risk, axis=1)


class DiscriminativeDiscreteBayesianClassifier(_DiscriminativeNNDiscretization):

    def __init__(
        self,
        n_clusters=8,
        *,
        tol=1e-4,
        max_iter=300,
        n_epochs=100,
        batch_size=128,
        kl_regularization=0.1,
        random_state=None,
    ):
        _DiscriminativeNNDiscretization.__init__(
            self,
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_epochs=n_epochs,
            batch_size=batch_size,
            kl_regularization=kl_regularization,
        )
        self.prior_attribute = "prior"


class DiscriminativeMinmaxClassifier(_DiscriminativeNNDiscretization):

    def __init__(
        self,
        n_clusters=8,
        *,
        tol=1e-4,
        max_iter=300,
        n_epochs=100,
        batch_size=128,
        kl_regularization=0.1,
        random_state=None,
    ):
        _DiscriminativeNNDiscretization.__init__(
            self,
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_epochs=n_epochs,
            batch_size=batch_size,
            kl_regularization=kl_regularization,
        )
        self.prior_attribute = "prior_star"


class _GenerativeDiscreteBayesianClassifier(BaseDiscreteBayesianClassifier):
    def __init__(
        self,
        n_clusters,
        *,
        tol,
        max_iter,
        random_state,
        n_epochs,
        batch_size,
    ):
        BaseDiscreteBayesianClassifier.__init__(self)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        # self._validate_params()

    def _fit_discretization(self, X, y, n_classes):
        input_dim = X.shape[1]
