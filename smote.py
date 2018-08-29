#-*- coding:utf-8 -*-
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np

class Smote:
    """
    SMOTE过采样算法.


    Parameters:
    -----------
    k: int
        选取的近邻数目.
    sampling_rate: int
        采样倍数, attention sampling_rate < k.
    newindex: int
        生成的新样本(合成样本)的索引号.
    """
    def __init__(self, sampling_rate=5, k=5):
        self.sampling_rate = sampling_rate
        self.k = k
        self.newindex = 0

    def fit(self, X, y=None):
        if y is not None:
            negative_X = X[y==0]
            X = X[y==1]

        n_samples, n_features = X.shape
        # 初始化一个矩阵, 用来存储合成样本
        self.synthetic = np.zeros((n_samples * self.sampling_rate, n_features))

        # 找出正样本集(数据集X)中的每一个样本在数据集X中的k个近邻
        knn = NearestNeighbors(n_neighbors=self.k).fit(X)
        for i in range(len(X)):
            k_neighbors = knn.kneighbors(X[i].reshape(1,-1), 
                                         return_distance=False)[0]
            # 对正样本集(minority class samples)中每个样本, 分别根据其k个近邻生成
            # sampling_rate个新的样本
            self.synthetic_samples(X, i, k_neighbors)

        if y is not None:
            return ( np.concatenate((self.synthetic, X, negative_X), axis=0), 
                     np.concatenate(([1]*(len(self.synthetic)+len(X)), y[y==0]), axis=0) )

        return np.concatenate((self.synthetic, X), axis=0)


    # 对正样本集(minority class samples)中每个样本, 分别根据其k个近邻生成sampling_rate个新的样本
    def synthetic_samples(self, X, i, k_neighbors):
        for j in range(self.sampling_rate):
            # 从k个近邻里面随机选择一个近邻
            neighbor = np.random.choice(k_neighbors)
            # 计算样本X[i]与刚刚选择的近邻的差
            diff = X[neighbor] - X[i]
            # 生成新的数据
            #self.synthetic[self.newindex] = X[i] + random.random() * diff
            self.synthetic[self.newindex] =X[neighbor]
            self.newindex += 1