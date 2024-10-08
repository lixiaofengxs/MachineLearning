import numpy as np

# metrics: 度量，指标

def accuracy_score(y_true, y_predict):
    '''计算 y_true 和 y_predict 之间的准确率'''
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict."
    
    return sum(y_true == y_predict) / len(y_true)