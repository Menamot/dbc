from sklearn.base import BaseEstimator
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from dbc.discretizers.cmeans import CMeansDiscretizer
from dbc.discretizers.cmeans_class_wise import CMeansClasswiseDiscretizer
from dbc.utils import (
    compute_prior,
    compute_pz_given_y,
    compute_prior_best,
    compute_posterior,
)
from dbc.cyprien_code import compute_piStar


class DiscreteBayesianClassifier(BaseEstimator):
    """组合式 DBC / DMC 基类：负责概率与决策，不负责离散化。"""

    def __init__(
        self, discretizer, loss_function="01", box=None, minmax=False, minmax_eps=1e-2
    ):
        # 1️⃣ Core definition / identity
        self.minmax = minmax  # True -> DMC, False -> DBC
        self.minmax_eps = minmax_eps
        self.box = box  # Model-specific parameter region

        # 2️⃣ External components (composition)
        self.discretizer = discretizer  # External discretizer object
        self.loss_function = loss_function

        # 3️⃣ Learned model parameters (after training)
        self.membership_train = None  # P(Z|X_train)
        self.p_hat = None  # P(Z|Y)
        self.prior = None  # P(Y)
        self.prior_best = None  # Optional: best prior for tuning

        # 4️⃣ Auxiliary components
        self.label_encoder = LabelEncoder()

    # ---------- 拟合 ----------
    def fit(self, X_train, y_train):
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        n_classes = len(np.unique(y_train_encoded))

        # 01 损失函数
        if isinstance(self.loss_function, str) and self.loss_function == "01":
            self.loss_function = np.ones((n_classes, n_classes)) - np.eye(n_classes)
        elif isinstance(self.loss_function, np.ndarray):
            # Check if it is a valid square loss matrix
            if (
                self.loss_function.ndim == 2
                and self.loss_function.shape[0] == n_classes
                and self.loss_function.shape[1] == n_classes
                and np.issubdtype(self.loss_function.dtype, np.number)
            ):
                # Valid custom loss matrix
                pass  # You can keep it as is
            else:
                raise ValueError(
                    f"Invalid loss_function matrix: expected shape ({n_classes}, {n_classes}), "
                    f"got {self.loss_function.shape}"
                )
        else:
            raise ValueError("Unsupported loss_function")

        # 先验概率
        self.prior = compute_prior(y_train_encoded)

        # 拟合离散化器
        self.discretizer.fit(X_train, y_train_encoded)
        self.membership_train = self.discretizer.transform(X_train)

        # 估计条件概率矩阵
        self.p_hat = compute_pz_given_y(self.membership_train, y_train_encoded)

        # 若为 minimax 则求最优先验
        if self.minmax:
            # 记得这里由于硬软离散方式不同会导致计算最优先验的方法不同
            if isinstance(
                self.discretizer, (CMeansDiscretizer, CMeansClasswiseDiscretizer)
            ):
                # Soft version of prior_star (SPDBC)
                self.prior_best, self.history = compute_prior_best(
                    y=y_train_encoded,
                    loss_function=self.loss_function,
                    p_hat=self.p_hat,
                    membership_degree=self.membership_train,
                    prior_train=self.prior,
                    eps=self.minmax_eps,
                    return_history=True,
                )
            else:
                # Hard version (DBC/DMC)
                self.prior_best = compute_piStar(
                    self.p_hat,
                    y_train_encoded,
                    n_classes,
                    self.loss_function,
                    N=1000,
                    Box=self.box,
                )[0]
        return self

    def _compute_sample_conditional_risk(self, X, prior_pred=None):
        """Compute posterior and sample conditional risk for given X."""
        check_is_fitted(self, ["p_hat"])

        # Determine which prior to use
        if prior_pred is None:
            prior_pred = self.prior_best if self.minmax else self.prior

        # Compute posterior probabilities
        posterior = compute_posterior(
            membership_pred=self.discretizer.transform(X),
            p_hat=self.p_hat,
            prior_train=self.prior,
            prior_pred=prior_pred,
        )

        # Compute sample conditional risk: R(Y|X) = P(Y|X) * loss_function
        sample_conditional_risk = posterior @ self.loss_function
        return sample_conditional_risk

    # ---------- hard prediction ----------
    def predict(self, X, prior_pred=None):
        """Predict class labels by minimizing conditional risk."""
        sample_conditional_risk = self._compute_sample_conditional_risk(X, prior_pred)
        y_pred = np.argmin(sample_conditional_risk, axis=1)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        return y_pred_decoded

    # ---------- probabilistic prediction ----------
    def predict_prob(self, X, prior_pred=None):
        """Predict class probabilities based on normalized inverse risks."""
        sample_conditional_risk = self._compute_sample_conditional_risk(X, prior_pred)
        # 然后使用差和比例来计算最终的概率
        temp = (
            np.sum(sample_conditional_risk, axis=1, keepdims=True)
            - sample_conditional_risk
        )
        prob = np.divide(temp, np.sum(temp, axis=1, keepdims=True))
        return prob
