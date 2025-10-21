import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from dbc.discretizers.base import BaseDiscretizer


class CMeansClasswiseDiscretizer(BaseDiscretizer):
    """
    按类别分别执行 Fuzzy C-Means 聚类的离散化器。
    每个类别独立执行 FCM，最后合并为全局 profile 空间。

    支持三种模式：
      1️⃣ use_kmeans=True：先用 KMeans 为每个类初始化中心；
      2️⃣ centers 指定：直接使用给定中心；
      3️⃣ 默认：随机初始化并运行完整 FCM。

    transform() 返回样本对所有簇的隶属度矩阵 (n_clusters_total, n_samples)，
    可直接传入 compute_p_hat_with_degree()。
    """

    def __init__(
        self,
        n_clusters_per_class=3,
        fuzzifier=1.5,
        tol=1e-4,
        max_iter=300,
        random_state=None,
        centers=None,  # 若指定，则不再迭代
        use_kmeans=False,  # 是否用KMeans初始化
    ):
        self.n_clusters_per_class = n_clusters_per_class
        self.fuzzifier = fuzzifier
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers = centers
        self.use_kmeans = use_kmeans

        # 拟合后属性
        self.centers_ = None
        self.center_labels_ = None
        self.n_clusters_total_ = None
        self.membership_degree_train_ = None
        self.labels_ = None

    def fit(self, X, y):
        """
        为每个类别独立执行 FCM 聚类并合并结果，
        然后用全局中心重新计算所有样本的 fuzzy membership。
        """
        classes = np.unique(y)
        centers_all, center_labels = [], []
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
            for c in classes:
                Xc = X[y == c]

                # 每个类别的簇数
                if isinstance(self.n_clusters_per_class, dict):
                    n_clusters = self.n_clusters_per_class.get(int(c), 3)
                else:
                    n_clusters = self.n_clusters_per_class

                # ======== 聚类方式选择 ========

                if self.use_kmeans:
                    # 使用 KMeans 初始化（不再迭代 FCM）
                    kmeans = KMeans(
                        n_clusters=n_clusters, random_state=self.random_state
                    )
                    kmeans.fit(Xc)
                    cntr = kmeans.cluster_centers_

                else:
                    # 标准 FCM
                    cntr, _, _, _, _, _, _ = fuzz.cluster.cmeans(
                        Xc.T,
                        c=n_clusters,
                        m=self.fuzzifier,
                        error=self.tol,
                        maxiter=self.max_iter,
                        seed=self.random_state,
                    )

                centers_all.append(cntr)
                center_labels.extend([c] * n_clusters)
                # ======== 全局中心整合 ========
                self.centers_ = np.vstack(centers_all)

            # ======== ⚡ 关键步骤：重新全局预测 ========
            u_pred, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                X.T,
                cntr_trained=self.centers_,
                m=self.fuzzifier,
                error=self.tol,
                maxiter=1,
            )

            self.membership_degree_train_ = u_pred
        self.labels_ = np.argmax(self.membership_degree_train_, axis=0)

        return self

    def transform(self, X):
        """
        使用拟合得到的全局聚类中心，重新计算输入 X 的隶属度。
        返回形状为 (n_clusters_total, n_samples) 的矩阵。
        """
        if not hasattr(self, "centers_"):
            raise RuntimeError("CMeansDiscretizer 未拟合，请先调用 fit()。")

        u_pred, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            X.T,
            cntr_trained=self.centers_,
            m=self.fuzzifier,
            error=self.tol,
            maxiter=1,
        )
        return u_pred
