import numpy as np
from sklearn.cluster import KMeans
from dbc.discretizers.base import BaseDiscretizer


class KMeansDiscretizer(BaseDiscretizer):
    """
    使用全局 KMeans 聚类实现离散化（硬 one-hot 隶属度版）。

    transform() 返回样本对各簇的 one-hot 隶属度矩阵 (n_clusters, n_samples)，
    即每个样本属于最近簇的隶属度为 1，其余为 0。

    这样既保留 KMeans 的硬聚类特性，
    又在接口上兼容 compute_p_hat_with_degree() 的输入格式。
    """

    def __init__(self, n_clusters=8, centers=None, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.centers_ = centers

    def fit(self, X, y=None):
        """对全部数据执行 KMeans 聚类"""
        if self.centers_ is None:
            self.model = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state
            )
            self.model.fit(X)
            self.centers_ = self.model.cluster_centers_
        return self

    def transform(self, X):
        """
        返回样本对每个簇的 one-hot 隶属度矩阵，形状为 (n_clusters, n_samples)。
        """
        if not hasattr(self, "centers_"):
            raise RuntimeError("KMeansDiscretizer 尚未 fit()。")

        dists = np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=2)
        cluster_ids = np.argmin(dists, axis=1)  # (n_samples,)

        n_samples = len(cluster_ids)
        n_clusters = self.n_clusters
        degree = np.zeros((n_clusters, n_samples))
        degree[cluster_ids, np.arange(n_samples)] = 1.0

        return degree
