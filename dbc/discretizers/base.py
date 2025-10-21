from abc import ABC, abstractmethod


class BaseDiscretizer(ABC):
    """所有离散化器都应实现 fit() 与 transform() 接口。"""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X):
        pass
