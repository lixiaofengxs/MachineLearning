import numpy as np

from .metrics import r2_score

class LinearRegression:

    def __init(self):
        """初始化 LinearRegression 模型"""
        self.coefficient_ = None #系数
        self.interception_ = None #截距
        self._theta = None


    def fit_normal(self, X_train, y_train): #正规化方程，直接求解系数和截距
        """根据训练数据集 X_train, y_train 训练 Linear Regression 模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train."

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta =  np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coefficient_ = self._theta[1:]

        return self

    
    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4): #使用梯度下降法训练系数和截距
        """根据训练数据集 X_train, y_train 使用梯度下降法训练 Linear Regression 模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train."

        def J(theta, X_b, y):
            try:
                return np.sum((X_b.dot(theta) - y) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y)

            for i in range(1, len(theta)):
                res[i] = np.sum((X_b.dot(theta) - y).dot(X_b[:, i]))

            return res * 2 / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - gradient * eta

                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train, 1))), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        eta = 0.01

        self._theta = gradient(X_b, y, initial_theta, eta)
        self.interception_ = self._theta[0]
        self.coefficient_ = self._theta[1:]

        return self  
                
    
    def predict(self, X_predict):
        """给定待预测数据集 X_predict，返回表示 X_predict 的结果向量"""
        assert self.coefficient_ is not None and self.interception_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coefficient_), \
            "the feature number of X_predict must be equal to X_train."

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)



    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)


    def __repr__(self):
        return "LinearRegression()"