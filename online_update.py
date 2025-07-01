import numpy as np
from model import LinearRegression

class OnlineGradientDescent:
    """
    在线梯度下降(OGD)方法
    每接收一个新样本点后，重新拟合整个历史数据集
    """
    def __init__(self, input_dim, learning_rate=0.01, max_iter=100, tol=1e-3):
        """
        初始化OGD方法
        
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


class ImplicitOMD:
    """
    隐式在线镜像下降(Implicit OMD)方法
    使用Hessian矩阵代替历史数据点信息，避免存储完整的数据集
    实现公式4: θ_{t+1} = argmin_θ { ℓ_t(θ) + (1/2η)||θ-θ_t||^2_H_t }
    """
    def __init__(self, input_dim, learning_rate=0.01, reg_param=1e-6, implicit_iters=20, tol=1e-5):
        """
        初始化Implicit OMD方法
        
        参数:
            input_dim: 输入特征维度
            learning_rate: 学习率η
            reg_param: 正则化参数，用于确保Hessian矩阵可逆
            implicit_iters: 隐式更新迭代次数
            tol: 收敛容差
        """
        self.model = LinearRegression(input_dim)
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.implicit_iters = implicit_iters
        self.tol = tol
        
        # 初始化Hessian矩阵为单位矩阵乘以正则化参数
        self.H_t = np.eye(input_dim) * self.reg_param
        
        # 累积数据计数
        self.t = 0
        
        self.history = {
            'train_losses': [],
            'test_losses': [],
            'weights': []
        }
    
    def _compute_loss(self, x, y, theta):
        """
        计算单个样本的损失值
        
        参数:
            x: 单个样本的特征向量，形状为 (input_dim,)
            y: 单个样本的标签
            theta: 模型参数，形状为 (input_dim,)
            
        返回:
            损失值
        """
        pred = np.dot(x, theta)
        return 0.5 * (pred - y) ** 2
    
    def _compute_loss_gradient(self, x, y, theta):
        """
        计算单个样本在指定参数theta处的损失函数梯度
        
        参数:
            x: 单个样本的特征向量，形状为 (input_dim,)
            y: 单个样本的标签
            theta: 模型参数，形状为 (input_dim,)
            
        返回:
            形状为 (input_dim,) 的梯度向量
        """
        # 对于均方误差损失，梯度为 (θ^T x - y) * x
        pred = np.dot(x, theta)
        error = pred - y
        return error * x
    
    def _compute_hessian(self, x):
        """
        计算单个样本的Hessian矩阵
        
        参数:
            x: 单个样本的特征向量，形状为 (input_dim,)
            
        返回:
            形状为 (input_dim, input_dim) 的Hessian矩阵
        """
        # 对于均方误差损失，Hessian为 x*x^T
        return np.outer(x, x)
    
    def _solve_implicit_update(self, x_new, y_new, theta_t):
        """
        求解隐式更新方程
        
        参数:
            x_new: 新样本的特征向量
            y_new: 新样本的标签
            theta_t: 当前参数
            
        返回:
            更新后的参数
        """
        # 创建一个临时模型用于迭代
        theta = theta_t.copy()
        
        # 使用迭代方法求解隐式更新方程
        for i in range(self.implicit_iters):
            # 计算当前参数下的梯度
            grad = self._compute_loss_gradient(x_new, y_new, theta)
            
            # 使用当前Hessian矩阵的逆计算更新方向
            H_inv = np.linalg.inv(self.H_t)
            
            # 计算隐式更新方程的解
            # θ_{t+1} = θ_t - η * H_t^(-1) * grad
            theta_new = theta_t - self.learning_rate * np.dot(H_inv, grad)
            
            # 检查收敛性
            if np.linalg.norm(theta_new - theta) < self.tol:
                theta = theta_new
                break
            
            theta = theta_new
        
        return theta
    
    def update(self, X_t, y_t, X_test=None, y_test=None):
        """
        在线更新：接收新的数据后更新模型
        
        参数:
            X_t: 当前时间步的完整历史特征矩阵
            y_t: 当前时间步的完整历史标签向量
            X_test: 测试集特征矩阵
            y_test: 测试集标签向量
            
        返回:
            当前步的训练损失
        """
        # 只使用最新的样本点更新模型
        if self.t < len(X_t):
            x_new = X_t[self.t]
            y_new = y_t[self.t]
            
            # 获取当前权重
            theta_t = self.model.get_weights()
            
            # 计算当前样本的Hessian矩阵
            hessian = self._compute_hessian(x_new)
            
            # 更新累积Hessian矩阵
            self.H_t += hessian
            
            # 求解隐式更新方程
            theta_new = self._solve_implicit_update(x_new, y_new, theta_t)
            
            # 更新模型参数
            self.model.set_weights(theta_new)
            
            # 增加时间步
            self.t += 1
        
        # 计算训练损失
        train_loss = self.model.loss(X_t, y_t)
        self.history['train_losses'].append(train_loss)
        self.history['weights'].append(self.model.get_weights().copy())
        
        # 如果提供了测试集，计算测试损失
        if X_test is not None and y_test is not None:
            test_loss = self.model.loss(X_test, y_test)
            self.history['test_losses'].append(test_loss)
        
        return train_loss
    
    def get_history(self):
        """获取训练历史"""
        return self.history


class OnePassOMD:
    """
    一次通过在线镜像下降(One-pass OMD)方法
    使用二阶泰勒展开，只保留当前点的梯度信息和Hessian曲率信息
    实现公式5: θ_{t+1} = argmin_θ { <g_t(θ_t), θ> + (1/2η)||θ-θ_t||^2_H_t }
    其中 H_t = H_{t-1} + η * H_t(θ_t)
    """
    def __init__(self, input_dim, learning_rate=0.1, reg_param=1e-6):
        """
        初始化One-pass OMD方法
        
        参数:
            input_dim: 输入特征维度
            learning_rate: 学习率η
            reg_param: 正则化参数，用于确保Hessian矩阵可逆
        """
        self.model = LinearRegression(input_dim)
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        
        # 初始化Hessian矩阵为单位矩阵乘以正则化参数
        self.H_t = np.eye(input_dim) * self.reg_param
        
        # 累积数据计数
        self.t = 0
        
        self.history = {
            'train_losses': [],
            'test_losses': [],
            'weights': []
        }
    
    def _compute_loss_gradient(self, x, y):
        """
        计算单个样本的损失函数梯度
        
        参数:
            x: 单个样本的特征向量，形状为 (input_dim,)
            y: 单个样本的标签
            
        返回:
            形状为 (input_dim,) 的梯度向量
        """
        # 对于均方误差损失，梯度为 -2(y-w^T*x)*x
        pred = self.model.predict(x.reshape(1, -1))[0]
        error = pred - y
        return 2 * error * x
    
    def _compute_hessian(self, x):
        """
        计算单个样本的Hessian矩阵
        参数:
            x: 单个样本的特征向量，形状为 (input_dim,)     
        返回:
            形状为 (input_dim, input_dim) 的Hessian矩阵
        """
        # 对于均方误差损失，Hessian为 2*x*x^T
        return 2 * np.outer(x, x)
    
    def update(self, X_t, y_t, X_test=None, y_test=None):
        """
        在线更新：接收新的数据后更新模型
        
        参数:
            X_t: 当前时间步的完整历史特征矩阵
            y_t: 当前时间步的完整历史标签向量
            X_test: 测试集特征矩阵
            y_test: 测试集标签向量
            
        返回:
            当前步的训练损失
        """
        # 只使用最新的样本点更新模型
        if self.t < len(X_t):
            x_new = X_t[self.t]
            y_new = y_t[self.t]
            
            # 获取当前权重
            theta_t = self.model.get_weights()
            
            # 计算当前样本点的梯度
            g_t = self._compute_loss_gradient(x_new, y_new)
            
            # 计算当前样本点的Hessian矩阵
            H_new = self._compute_hessian(x_new)
            
            # 更新累积Hessian矩阵: H_t = H_{t-1} + η * H_new
            self.H_t += self.learning_rate * H_new
            
            # 计算新的权重向量 θ_{t+1} = θ_t - η * H_t^(-1) * g_t
            H_inv = np.linalg.inv(self.H_t)
            theta_new = theta_t - self.learning_rate * np.dot(H_inv, g_t)
            
            # 更新模型参数
            self.model.set_weights(theta_new)
            
            # 增加时间步
            self.t += 1
        
        # 计算训练损失
        train_loss = self.model.loss(X_t, y_t)
        self.history['train_losses'].append(train_loss)
        self.history['weights'].append(self.model.get_weights().copy())
        
        # 如果提供了测试集，计算测试损失
        if X_test is not None and y_test is not None:
            test_loss = self.model.loss(X_test, y_test)
            self.history['test_losses'].append(test_loss)
        
        return train_loss
    
    def get_history(self):
        """获取训练历史"""
        return self.history 