import numpy as np
from model import LinearRegression

class OnlineGradientDescent:
    """
    在线梯度下降(OGM)方法
    每接收一个新样本点后，重新拟合整个历史数据集
    """
    def __init__(self, input_dim, learning_rate=0.01, max_iter=100, tol=1e-9):
        """
        初始化OGM方法
        
        参数:
            input_dim: 输入特征维度
            learning_rate: 学习率
            max_iter: 每轮最大迭代次数
            tol: 收敛容差
        """
        self.model = LinearRegression(input_dim)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.history = {
            'train_losses': [],
            'test_losses': [],
            'weights': []
        }
    
    def _compute_gradient(self, X, y):
        """
        计算均方误差损失的梯度
        
        参数:
            X: 形状为 (n_samples, input_dim) 的特征矩阵
            y: 形状为 (n_samples,) 的标签向量
            
        返回:
            形状为 (input_dim,) 的梯度向量
            y_hat=w^T*x
            f=(y-y_hat)^2
            f对w的偏导数为-2(y-y_hat)x
            -2(y-y_hat)x=0
        """
        predictions = self.model.predict(X)
        errors = predictions - y
        gradient = 2 * np.dot(X.T, errors) / len(y)
        return gradient
    
    def _gradient_descent(self, X, y):
        """
        执行梯度下降优化
        
        参数:
            X: 形状为 (n_samples, input_dim) 的特征矩阵
            y: 形状为 (n_samples,) 的标签向量
            
        返回:
            优化后的权重向量
        """
        weights = self.model.get_weights()
        
        for i in range(self.max_iter):
            # 计算梯度
            gradient = self._compute_gradient(X, y)
            
            # 更新权重
            new_weights = weights - self.learning_rate * gradient
            
            # 检查收敛性
            if np.linalg.norm(new_weights - weights) < self.tol:
                weights = new_weights
                break
                
            weights = new_weights
        
        return weights
    
    def update(self, X_t, y_t, X_test=None, y_test=None):
        """
        在线更新：接收新的数据集后重新训练模型
        
        参数:
            X_t: 当前时间步的完整历史特征矩阵
            y_t: 当前时间步的完整历史标签向量
            X_test: 测试集特征矩阵
            y_test: 测试集标签向量
            
        返回:
            当前步的训练损失
        """
        # 使用梯度下降重新拟合整个历史数据集
        weights = self._gradient_descent(X_t, y_t)
        
        # 更新模型参数
        self.model.set_weights(weights)
        
        # 计算训练损失
        train_loss = self.model.loss(X_t, y_t)
        self.history['train_losses'].append(train_loss)
        self.history['weights'].append(weights.copy())
        
        # 如果提供了测试集，计算测试损失
        if X_test is not None and y_test is not None:
            test_loss = self.model.loss(X_test, y_test)
            self.history['test_losses'].append(test_loss)
        
        return train_loss
    
    def get_history(self):
        """获取训练历史"""
        return self.history 