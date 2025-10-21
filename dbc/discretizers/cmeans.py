import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from dbc.discretizers.base import BaseDiscretizer


class CMeansDiscretizer(BaseDiscretizer):
    """
    基于 Fuzzy C-Means 的离散化器。
    支持三种模式：
      1️⃣ use_kmeans=True：先用 KMeans 作为初始中心；
      2️⃣ centers 指定：使用给定中心直接计算隶属度；
      3️⃣ 默认：随机初始化中心并执行完整 FCM。

    transform() 返回样本对各簇的隶属度矩阵（degree），
    形状为 (n_clusters, n_samples)，以便 compute_p_hat_with_degree() 使用。
    """

    def __init__(
        self,
        n_clusters=8,
        fuzzifier=1.5,
        tol=1e-4,
        max_iter=300,
        random_state=None,
        centers=None,  # 若指定则不再迭代
        use_kmeans=False,  # 是否使用 KMeans 初始化
    ):
        self.n_clusters = n_clusters
        self.fuzzifier = fuzzifier
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers = centers
        self.use_kmeans = use_kmeans

    def fit(self, X, y=None):
        """
        拟合 FCM：
          - 若指定 centers，则使用该中心直接计算隶属度；
          - 若 use_kmeans=True，则用 KMeans 结果初始化；
          - 否则执行标准 FCM 随机初始化。
        """
        if self.centers is not None:
            # ⚡ 固定中心模式：直接计算隶属度，不再迭代
            cntr = np.asarray(self.centers)
            u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                X.T,
                cntr_trained=cntr,
                m=self.fuzzifier,
                error=self.tol,
                maxiter=1,
            )
            self.centers_ = cntr
            self.membership_degree_ = u

        else:
            if self.use_kmeans:
                # 🔹 先用 KMeans 初始化中心
                kmeans = KMeans(
                    n_clusters=self.n_clusters, random_state=self.random_state
                )
                kmeans.fit(X)
                cntr = kmeans.cluster_centers_
                # 🔁 用该中心执行 FCM
                u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                    X.T,
                    cntr_trained=cntr,
                    m=self.fuzzifier,
                    error=self.tol,
                    maxiter=1,
                )
            else:
                # 🔁 标准 FCM 模式（随机初始化）
                cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                    X.T,
                    c=self.n_clusters,
                    m=self.fuzzifier,
                    error=self.tol,
                    maxiter=self.max_iter,
                    seed=self.random_state,
                )

            self.centers_ = cntr
            self.membership_degree_ = u

        # 保存硬标签（方便兼容性使用）
        # self.labels_ = np.argmax(self.membership_degree_, axis=0)
        return self

    def transform(self, X):
        """
        返回样本对每个簇的隶属度矩阵 (degree)
        形状为 (n_clusters, n_samples)
        """
        if not hasattr(self, "centers_"):
            raise RuntimeError("CMeansDiscretizer 未拟合，请先调用 fit()。")

        u_pred, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            X.T,
            cntr_trained=self.centers_,
            m=self.fuzzifier,
            error=self.tol,
            maxiter=1,  # 只需一次隶属度计算
        )
        return u_pred
