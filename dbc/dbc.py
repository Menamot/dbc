from numbers import Real, Integral

import skfuzzy as fuzz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from dbc.utils import *

class BaseDiscreteBayesianClassifier(BaseEstimator):
    def __init__(self):
        self.prior_attribute = None
        self.prior = None
        self.p_hat = None
        self.label_encoder = None
        self.loss_function = None

    def fit(self, X, y, loss_function="01"):
        n_classes = len(set(y))
        if loss_function == "01":
            self.loss_function = np.ones((n_classes, n_classes)) - np.eye(n_classes)
        else:
            raise ValueError("Invalid loss_function")
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.prior = compute_prior(y_encoded, n_classes)
        self._fit_discretization(X, y_encoded, n_classes)

    def predict(self, X, prior_pred=None):
        check_is_fitted(self, ['p_hat', self.prior_attribute])
        if prior_pred is None:
            prior_pred = getattr(self, self.prior_attribute)
        return self._predict_profiles(X, prior_pred)

    def predict_prob(self, X, prior_pred=None):
        check_is_fitted(self, ['p_hat', self.prior_attribute])
        if prior_pred is None:
            prior_pred = getattr(self, self.prior_attribute)
        return self._predict_probabilities(X, prior_pred)

    def _fit_discretization(self, X, y, n_classes):
        raise NotImplementedError

    def _transform_to_discrete_profiles(self, X):
        raise NotImplementedError

    def _predict_profiles(self, X, prior):
        raise NotImplementedError

    def _predict_probabilities(self, X, prior):
        raise NotImplementedError


class _KmeansDiscretization(BaseDiscreteBayesianClassifier, KMeans):
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
            box
    ):
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
            algorithm=algorithm
        )
        self.box = box
        BaseDiscreteBayesianClassifier.__init__(self)

    def _fit_discretization(self, X, y, n_classes):
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
        self.p_hat = compute_p_hat(self.discretization_model.labels_, y, n_classes,
                                   self.discretization_model.n_clusters)
        if self.prior_attribute == 'prior_star':
            self.prior_star = compute_piStar(self.p_hat, y, n_classes, self.loss_function, 1000, self.box)[0]

    def _transform_to_discrete_profiles(self, X):
        return self.discretization_model.predict(X)

    def _predict_profiles(self, X, prior):
        discrete_profiles = self._transform_to_discrete_profiles(X)
        return self.label_encoder.inverse_transform(
            predict_profile_label(prior, self.p_hat, self.loss_function)[discrete_profiles]
        )

    def _predict_probabilities(self, X, prior):
        class_risk = (prior.reshape(-1, 1) * self.loss_function).T @ self.p_hat
        prob = 1 - (class_risk / np.sum(class_risk, axis=0))
        return prob[:, self._transform_to_discrete_profiles(X)].T


class KmeansDiscreteBayesianClassifier(_KmeansDiscretization):
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
        )
        self.prior_attribute = "prior"

class KmeansDiscreteMinimaxClassifier(_KmeansDiscretization):
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
        )
        self.prior_attribute = "prior_star"



class _DecisionTreeDiscretization(BaseDiscreteBayesianClassifier, DecisionTreeClassifier):
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
            box
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
        self.p_hat = compute_p_hat(discretize_features(X, self.discretization_model), y, n_classes,
                                   self.discretization_model.get_n_leaves())
        if self.prior_attribute == 'prior_star':
            self.prior_star = compute_piStar(self.p_hat, y, n_classes, self.loss_function, 1000, self.box)[0]
    def _transform_to_discrete_profiles(self, X):
        return discretize_features(X, self.discretization_model)

    def _predict_profiles(self, X, prior):
        discrete_profiles = self._transform_to_discrete_profiles(X)
        return self.label_encoder.inverse_transform(
            predict_profile_label(prior, self.p_hat, self.loss_function)[discrete_profiles]
        )

    def _predict_probabilities(self, X, prior):
        class_risk = (prior.reshape(-1, 1) * self.loss_function).T @ self.p_hat
        prob = 1 - (class_risk / np.sum(class_risk, axis=0))
        return prob[:, self._transform_to_discrete_profiles(X)].T


class DecisionTreeDiscreteBayesianClassifier(_DecisionTreeDiscretization):
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
        "init": [callable, "array-like",None],
        "metric": [StrOptions({"euclidean", "cityblock", "minkowski"}), callable]
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
            random_state
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
        self._validate_params()

    def _fit_discretization(self, X, y, n_classes):
        if self.cluster_centers is None:
            self.cluster_centers, membership_degree, _, _, _, _, _ = fuzz.cluster.cmeans(
                X.T,
                c=self.n_clusters,
                m=self.fuzzifier,
                error=self.tol,
                maxiter=self.max_iter,
                metric=self.metric,
                init=self.init,
                seed=self.random_state
            )
        else:
            membership_degree, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                X.T,
                cntr_trained=self.cluster_centers,
                m=self.fuzzifier,
                error=self.tol,
                maxiter=self.max_iter,
                metric=self.metric,
                init=self.init,
                seed=self.random_state
            )
        self.p_hat = compute_p_hat_with_degree(membership_degree, y, n_classes)

    def _predict_probabilities(self, X, prior):

        membership_degree_pred, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            X.T,
            cntr_trained=self.cluster_centers,
            m=self.fuzzifier,
            error=self.tol,
            maxiter=self.max_iter
        )
        prob = compute_posterior(membership_degree_pred, self.p_hat, prior, self.loss_function)
        return prob

    def _predict_profiles(self, X, prior):
        prob = self._predict_probabilities(X, prior)
        return self.label_encoder.inverse_transform(np.argmax(prob, axis=1))


class CmeansDiscreteBayesianClassifier(_CmeansDiscretization):
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
            random_state=None
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
            random_state=random_state
        )
        self.prior_attribute = "prior"