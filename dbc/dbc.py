from numbers import Real, Integral

import skfuzzy as fuzz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted
from dbc.utils import *

class DiscreteBayesianClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.prior = None
        self.p_hat = None
        self.label_encoder = None
        self.cluster_centers = None
        self.loss_function = None

    def fit(self, X, y, loss_function="01"):
        n_classes = len(set(y))
        if loss_function == "01":
            self.loss_function=np.ones((n_classes, n_classes)) - np.eye(n_classes)
        else:
            raise ValueError("The parameter 'loss_function' should be in options: '01'")
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.prior = compute_prior(y_encoded, n_classes)
        self._fit_discretization(X, y_encoded, n_classes)

    def predict(self, X, prior_pred=None):
        check_is_fitted(self, ['p_hat', 'prior'])
        if prior_pred is None:
            prior_pred = self.prior

        return self._predict_profiles(X, prior_pred)

    def predict_prob(self, X, prior_pred=None):
        check_is_fitted(self, ['p_hat', 'prior'])
        if prior_pred is None:
            prior_pred = self.prior

        return self._predict_probabilities(X, prior_pred)

    def _fit_discretization(self, X, y, n_classes):
        raise NotImplementedError

    def _predict_profiles(self, X, prior):
        raise NotImplementedError

    def _predict_probabilities(self, X, prior):
        raise NotImplementedError


class KmeansDiscreteBayesianClassifier(DiscreteBayesianClassifier):
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
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm


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

    def _predict_profiles(self, X, prior):
        discrete_profiles = self.discretization_model.predict(X)
        return self.label_encoder.inverse_transform(
            predict_profile_label(prior, self.p_hat, self.loss_function)[discrete_profiles]
        )

    def _predict_probabilities(self, X, prior):
        class_risk = (prior.reshape(-1, 1) * self.loss_function).T @ self.p_hat
        prob = 1 - (class_risk / np.sum(class_risk, axis=0))
        return prob[:, self.discretization_model.predict(X)].T


class CmeansDiscreteBayesianClassifier(DiscreteBayesianClassifier):
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
        super().__init__()
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


class KmeansDiscreteMinimaxClassifier(KmeansDiscreteBayesianClassifier):
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
            box=None
    ):
        super().__init__(n_clusters,init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        self.prior_star=None
        self.box=box


    def _fit_discretization(self, X, y, n_classes):
        super()._fit_discretization(X, y, n_classes)
        self.prior_star = compute_piStar(self.p_hat, y, n_classes, self.loss_function, 1000, self.box)[0]


    def predict(self, X, prior_pred=None):
        check_is_fitted(self, ['p_hat', 'prior'])
        if prior_pred is None:
            prior_pred = self.prior_star

        return self._predict_profiles(X, prior_pred)

    def predict_prob(self, X, prior_pred=None):
        check_is_fitted(self, ['p_hat', 'prior'])
        if prior_pred is None:
            prior_pred = self.prior_star

        return self._predict_probabilities(X, prior_pred)
