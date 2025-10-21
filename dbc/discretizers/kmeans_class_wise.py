import numpy as np
from sklearn.cluster import KMeans
from dbc.discretizers.base import BaseDiscretizer


class KMeansClasswiseDiscretizer(BaseDiscretizer):
    """
    按类别分别执行 KMeans 聚类（硬 one-hot 隶属度版）。
    每个类别独立聚类，最后合并为全局 profile 索引。
    transform() 输出每个样本对所有簇的 one-hot 隶属度矩阵 (n_clusters_total, n_samples)。

    优点：
    -------
    - 各类别的局部结构被分别建模；
    - 输出矩阵格式与 `compute_p_hat_with_degree` 完全兼容；
    - 与全局 KMeansDiscretizer 接口保持一致。
    """

    def __init__(self, n_clusters_per_class=3, centers=None, random_state=None):
        """
        参数
        ----------
        n_clusters_per_class : int 或 dict
            若为 int，则所有类别使用相同的簇数；
            若为 dict，则可为每个类别单独指定簇数，例如 {0:3, 1:5}。
        random_state : int 或 None
            随机种子。
        """
        self.n_clusters_per_class = n_clusters_per_class
        self.random_state = random_state
        self.centers_ = centers
        self.labels_ = None
        self.center_labels_ = None
        self.n_clusters_total_ = None

    def fit(self, X, y):
        """为每个类别独立训练 KMeans，并合并结果"""
        if self.centers_ is None:
            classes = np.unique(y)
            centers, center_labels = [], []
            # labels = np.zeros(len(X), dtype=int)
            offset = 0

            for c in classes:
                Xc = X[y == c]
                # 若为 dict，则取每类特定簇数
                if isinstance(self.n_clusters_per_class, dict):
                    k = self.n_clusters_per_class.get(int(c), 3)
                else:
                    k = self.n_clusters_per_class

                kmeans = KMeans(n_clusters=k, random_state=self.random_state)
                kmeans.fit(Xc)

                centers.append(kmeans.cluster_centers_)
                center_labels.extend([c] * k)

                # 保存每个样本在全局簇空间的索引
                # labels[y == c] = kmeans.predict(Xc) + offset

                offset += k

            self.centers_ = np.vstack(centers)
            self.center_labels_ = np.array(center_labels)
            # self.labels_ = labels
        self.n_clusters_total_ = len(self.centers_)
        return self

    def transform(self, X):
        """
        将样本映射到最近的全局聚类中心，
        返回 one-hot 隶属度矩阵 (n_clusters_total, n_samples)。
        """
        if not hasattr(self, "centers_"):
            raise RuntimeError("KMeansClasswiseDiscretizer 尚未 fit()。")

        dists = np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=2)
        cluster_ids = np.argmin(dists, axis=1)  # (n_samples,)

        n_samples = len(cluster_ids)
        n_clusters = self.n_clusters_total_
        degree = np.zeros((n_clusters, n_samples))
        degree[cluster_ids, np.arange(n_samples)] = 1.0

        return degree
