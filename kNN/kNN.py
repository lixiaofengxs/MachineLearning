import numpy as np

from math import sqrt
from collections import Counter

class kNNClassifier:

    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be valid."
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集 X_train 和 y_train 训练 kNN 分类器"""
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train."

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测数据集 X_predict, 返回表示 X_predict 的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict."
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train."

        y_predict = [self._predict(x) for x in X_predict]

        return np.array(y_predict) 

    def _predict(self, x):
        """给定单个待预测数据 x, 返回 x 的预测结果值 """
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train."

        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
         
        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "kNN(k=%d)" % self.k


    