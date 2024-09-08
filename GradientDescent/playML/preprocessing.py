import numpy as np

class StandardScalar:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据训练数据集X获得数据的均值和方差"""
        assert X.ndim == 2, "the dimension of X must be 2. "

        self.mean_ = np.array(np.mean(X[:, i]) for i in range(X.shape[1]))
        self.scale_ = np.array(np.std(X[:, i]) for i in range(X.shape[1]))

        return self 

    def transform(self, X):
        """将X根据当前的StandardScalar进行均值方差归一化处理"""
        assert X.ndim == 2, "the dimension of X must be 2."
        assert self.mean_ is not None and self.scale_ is not None, "must fit before transform!"
        assert X.shape[1] == len(self.mean_), "the feature number of X must be equal to mean_ and std_"

        resX = np.empty(shape=X.shape, dtype=float)
        for col in X.shape[1]:
            resX[:, col] = (X[:, col] - self.mean[col]) / self.scale_[col]

        return resX


class MinMaxScalar:
    
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        """根据训练数据集X获得数据的最小值和最大值"""
        assert X.ndim == 2, "the dimension of X must be 2. "

        self.min_ = np.array(np.min(X[:, i]) for i in range(X.shape[1]))
        self.max_ = np.array(np.max(X[:, i]) for i in range(X.shape[1]))

        return self

    def transform(self, X):
        """将X根据当前的MinMaxScalar进行最值归一化处理"""
        assert X.ndim == 2, "the dimension of X must be 2."
        assert self.min_ is not None and self.max_ is not None, "must fit before transform!"
        assert X.shape[1] == len(self.min_), "the feature number of X must be equal to min_ and max_"

        resX = np.empty(shape=X.shape, dtype=float)
        for col in X.shape[1]:
            resX[:, col] = (X[:, col] - self.min_[col]) / (self.max_[col] - slef.min_[col])

        return resX
    




