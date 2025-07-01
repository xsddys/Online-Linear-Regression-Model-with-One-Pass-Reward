import numpy as np

class LinearRegression:
    """
    线性回归模型：y = Xw
    """
    def __init__(self, input_dim):
        """
        初始化线性回归模型
        
        参数:
            input_dim: 输入特征维度
        """
        self.input_dim = input_dim
        self.weights = np.zeros(input_dim)
    
    def predict(self, X):
        """
        使用当前模型参数进行预测
        
        参数:
            X: 形状为 (n_samples, input_dim) 的特征矩阵
            
        返回:
            预测结果，形状为 (n_samples,)
        """
        return np.dot(X, self.weights)
    
    def set_weights(self, weights):
        """
        设置模型权重
        
        参数:
            weights: 形状为 (input_dim,) 的权重向量
        """
        assert len(weights) == self.input_dim
        self.weights = weights.copy()
    
    def get_weights(self):
        """
        获取当前模型权重
        
        返回:
            当前权重向量
        """
        return self.weights.copy()
    
    def loss(self, X, y):
        """
        计算均方误差损失
        
        参数:
            X: 形状为 (n_samples, input_dim) 的特征矩阵
            y: 形状为 (n_samples,) 的标签向量
            
        返回:
            均方误差损失
        """
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2) 