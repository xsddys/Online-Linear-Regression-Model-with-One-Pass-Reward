import numpy as np
from model import LinearRegression
import time
import psutil
import os

def get_memory_usage():
    """获取当前进程的内存占用（MB）"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # 转换为MB

class OnlineGradientDescent:
    """
    在线梯度下降(OGD)方法
    每接收一个新样本点后，重新拟合整个历史数据集
    """
    def __init__(self, input_dim, learning_rate=0.01, max_iter=100, tol=1e-3, weight_decay=0.0):
        """
        初始化OGD方法
        
        参数:
            input_dim: 输入特征维度
            learning_rate: 学习率
            max_iter: 每轮最大迭代次数
            tol: 收敛容差
            weight_decay: 权重衰减系数（L2正则化强度）
        """
        self.model = LinearRegression(input_dim)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weight_decay = weight_decay  # 添加权重衰减参数
        self.history = {
            'train_losses': [],
            'test_losses': [],
            'weights': [],
            'time_records': [],        # 记录每20步的累积运行时间
            'memory_records': [],      # 记录每20步的内存占用
            'time_record_steps': [],   # 记录时间的步数
            'memory_record_steps': []  # 记录内存的步数
        }
        
        # 初始化计时器和初始内存占用
        self.start_time = time.time()
        self.total_time = 0.0
    
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
        
        # 添加权重衰减项的梯度 (L2正则化)
        if self.weight_decay > 0:
            weights = self.model.get_weights()
            gradient += 2 * self.weight_decay * weights
            
        return gradient
    
    def _compute_loss(self, X, y, weights):
        """
        计算带有权重衰减的损失函数值
        
        参数:
            X: 形状为 (n_samples, input_dim) 的特征矩阵
            y: 形状为 (n_samples,) 的标签向量
            weights: 模型权重
            
        返回:
            损失值
        """
        predictions = np.dot(X, weights)
        mse = np.mean((predictions - y) ** 2)
        
        # 添加权重衰减项 (L2正则化)
        if self.weight_decay > 0:
            l2_reg = self.weight_decay * np.sum(weights ** 2)
            return mse + l2_reg
        
        return mse
    
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
        # 记录开始时间
        iter_start_time = time.time()
        
        # 使用梯度下降重新拟合整个历史数据集
        weights = self._gradient_descent(X_t, y_t)
        
        # 更新模型参数
        self.model.set_weights(weights)
        
        # 计算训练损失
        if self.weight_decay > 0:
            train_loss = self._compute_loss(X_t, y_t, weights)
        else:
            train_loss = self.model.loss(X_t, y_t)
            
        self.history['train_losses'].append(train_loss)
        self.history['weights'].append(weights.copy())
        
        # 如果提供了测试集，计算测试损失
        if X_test is not None and y_test is not None:
            if self.weight_decay > 0:
                test_loss = self._compute_loss(X_test, y_test, weights)
            else:
                test_loss = self.model.loss(X_test, y_test)
            self.history['test_losses'].append(test_loss)
        
        # 更新累积运行时间
        self.total_time += (time.time() - iter_start_time)
        
        # 每20步记录一次时间和内存
        current_step = len(self.history['train_losses'])
        if current_step % 20 == 0:
            self.history['time_records'].append(self.total_time)
            self.history['time_record_steps'].append(current_step)
            
            self.history['memory_records'].append(get_memory_usage())
            self.history['memory_record_steps'].append(current_step)
        
        return train_loss
    
    def get_history(self):
        """获取训练历史"""
        self.history['total_time'] = self.total_time
        return self.history


class ImplicitOMD:
    """
    隐式在线镜像下降(Implicit OMD)方法
    使用Hessian矩阵代替历史数据点信息，避免存储完整的数据集
    实现公式4: θ_{t+1} = argmin_θ { ℓ_t(θ) + (1/2η)||θ-θ_t||^2_H_t }
    """
    def __init__(self, input_dim, learning_rate=0.01, reg_param=1e-6, implicit_iters=20, tol=1e-8, beta=0):
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
        self.beta = beta
        
        # 初始化Hessian矩阵为单位矩阵乘以正则化参数
        self.H_t = np.eye(input_dim) * self.reg_param
        
        # 累积数据计数
        self.t = 0
        
        self.history = {
            'train_losses': [],
            'test_losses': [],
            'weights': [],
            'time_records': [],        # 记录每20步的累积运行时间
            'memory_records': [],      # 记录每20步的内存占用
            'time_record_steps': [],   # 记录时间的步数
            'memory_record_steps': []  # 记录内存的步数
        }
        
        # 初始化计时器和初始内存占用
        self.start_time = time.time()
        self.total_time = 0.0
    
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
    
    def _compute_full_objective_gradient(self, x, y, theta, theta_t):
        """
        计算公式4中完整目标函数的梯度
        
        参数:
            x: 单个样本的特征向量
            y: 单个样本的标签
            theta: 当前迭代的参数
            theta_t: 初始参数θ_t
            
        返回:
            目标函数的梯度
        """
        # 计算损失函数梯度 ∇ℓ_t(θ)
        loss_grad = self._compute_loss_gradient(x, y, theta)
        
        # 计算正则化项梯度 η*H_t(θ-θ_t)
        reg_term = np.dot(self.H_t, (theta - theta_t)) * self.learning_rate
        
        # 总梯度 = ∇ℓ_t(θ) + η*H_t(θ-θ_t)
        return loss_grad 
    
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
        
        # 使用梯度下降方法直接最小化完整目标函数
        # f(θ) = ℓ_t(θ) + (1/2η)||θ-θ_t||^2_H_t
        
        # 内部优化的学习率（可调整）
        inner_lr = 0.01
        
        for i in range(self.implicit_iters):
            # 计算完整目标函数的梯度
            full_grad = self._compute_full_objective_gradient(x_new, y_new, theta, theta_t)
            
            # 梯度下降更新
            theta_new = theta - inner_lr * full_grad
            
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
        # 记录开始时间
        iter_start_time = time.time()
        
        # 只使用最新的样本点更新模型
        if self.t < len(X_t):
            x_new = X_t[self.t]
            y_new = y_t[self.t]
            
            # 获取当前权重
            theta_t = self.model.get_weights()
            
            # 计算当前样本的Hessian矩阵
            hessian = self._compute_hessian(x_new)
            
            # 更新累积Hessian矩阵
            if self.beta == 0:
                self.H_t += hessian
            else:
                self.H_t = (1-self.beta)*self.H_t + self.beta*hessian
                
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
        
        # 更新累积运行时间
        self.total_time += (time.time() - iter_start_time)
        
        # 每20步记录一次时间和内存
        current_step = len(self.history['train_losses'])
        if current_step % 20 == 0:
            self.history['time_records'].append(self.total_time)
            self.history['time_record_steps'].append(current_step)
            
            self.history['memory_records'].append(get_memory_usage())
            self.history['memory_record_steps'].append(current_step)
        
        return train_loss
    
    def get_history(self):
        """获取训练历史"""
        self.history['total_time'] = self.total_time
        return self.history


class OnePassOMD:
    """
    一次通过在线镜像下降(One-pass OMD)方法
    使用二阶泰勒展开，只保留当前点的梯度信息和Hessian曲率信息
    实现公式5: θ_{t+1} = argmin_θ { <g_t(θ_t), θ> + (1/2η)||θ-θ_t||^2_H_t }
    严格遵循算法1流程:
    1. 定义损失函数
    2. 更新 H_t = H_{t-1} + η * H_t(θ_t)
    3. 计算 θ'_{t+1} = θ_t - η * H_t^{-1} * g_t(θ_t)
    4. 计算 θ_{t+1} = argmin_θ ||θ - θ'_{t+1}||^2_{H_t} subject to ||θ||_2 ≤ B
    5. 更新 H_{t+1} = H_t + H_t(θ_{t+1})
    """
    def __init__(self, input_dim, learning_rate=0.1, reg_param=1e-6, beta=0, param_bound=50.0):
        """
        初始化One-pass OMD方法
        
        参数:
            input_dim: 输入特征维度
            learning_rate: 学习率η
            reg_param: 正则化参数，用于确保Hessian矩阵可逆
            beta: 指数加权平均系数，用于Hessian矩阵更新
            param_bound: 参数范数上界B，用于约束参数在L2球内
        """
        self.model = LinearRegression(input_dim)
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.beta = beta
        self.param_bound = param_bound  # 参数范数上界
        
        # 初始化Hessian矩阵为单位矩阵乘以正则化参数
        self.H_t = np.eye(input_dim) * self.reg_param
        self.H_tilde_t = np.eye(input_dim) * self.reg_param
        
        # 累积数据计数
        self.t = 0
        
        self.history = {
            'train_losses': [],
            'test_losses': [],
            'weights': [],
            'time_records': [],        # 记录每20步的累积运行时间
            'memory_records': [],      # 记录每20步的内存占用
            'time_record_steps': [],   # 记录时间的步数
            'memory_record_steps': []  # 记录内存的步数
        }
        
        # 初始化计时器和初始内存占用
        self.start_time = time.time()
        self.total_time = 0.0
    
    def _compute_loss_gradient(self, x, y, theta=None):
        """
        计算单个样本的损失函数梯度
        
        参数:
            x: 单个样本的特征向量，形状为 (input_dim,)
            y: 单个样本的标签
            theta: 可选，如果提供则使用此参数计算梯度，否则使用模型当前参数
            
        返回:
            形状为 (input_dim,) 的梯度向量
        """
        if theta is None:
            # 使用当前模型参数
            pred = self.model.predict(x.reshape(1, -1))[0]
        else:
            # 使用指定参数
            pred = np.dot(x, theta)
            
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
    
    def _project_to_l2_ball(self, theta, center, radius, H):
        """
        将参数投影到以center为中心，H范数下半径为radius的球内
        
        参数:
            theta: 要投影的参数向量
            center: 球的中心点
            radius: 球的半径
            H: 定义范数的矩阵
            
        返回:
            投影后的参数向量
        """
        # 计算 ||θ - center||_H
        diff = theta - center
        norm_h = np.sqrt(np.dot(diff, np.dot(H, diff)))
        
        # 如果已经在球内，直接返回
        if norm_h <= radius:
            return theta
        
        # 否则进行投影，保持方向但缩放长度
        return center + (radius / norm_h) * diff
    
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
        # 记录开始时间
        iter_start_time = time.time()
        
        # 只使用最新的样本点更新模型
        if self.t < len(X_t):
            x_new = X_t[self.t]
            y_new = y_t[self.t]
            
            # 获取当前权重 (θ_t)
            theta_t = self.model.get_weights()
            
            # 步骤1: 定义损失函数 (已隐含在代码中)
            
            # 计算当前样本点的Hessian矩阵 H_t(θ_t)
            H_new = self._compute_hessian(x_new)
            
            # 步骤2: 更新 H_tilde_t = H_t + η * H_t(θ_t)

            self.H_tilde_t += self.learning_rate * H_new

            
            # 计算当前样本点的梯度 g_t(θ_t)
            g_t = self._compute_loss_gradient(x_new, y_new, theta_t)
            
            # 步骤3: 计算 θ'_{t+1} = θ_t - η * H_tilde_t^{-1} * g_t(θ_t)
            try:
                H_tilde_inv = np.linalg.inv(self.H_tilde_t)
                theta_prime = theta_t - self.learning_rate * np.dot(H_tilde_inv, g_t)
            except np.linalg.LinAlgError:
                # 如果矩阵接近奇异，添加正则化
                H_reg = self.H_tilde_t + np.eye(self.input_dim) * self.reg_param * 10
                H_tilde_inv = np.linalg.inv(H_reg)
                theta_prime = theta_t - self.learning_rate * np.dot(H_tilde_inv, g_t)
            
            # 步骤4: 计算 θ_{t+1} = argmin_θ ||θ - θ'_{t+1}||^2_{H_tilde_t} subject to ||θ||_2 ≤ B
            # 这等价于投影到L2球内
            theta_new = self._project_to_l2_ball(theta_prime, 0, self.param_bound, np.eye(self.input_dim))
            
            # 更新模型参数
            self.model.set_weights(theta_new)
            
            # 步骤5: 更新 H_{t+1} = H_t + H_t(θ_{t+1})
            H_new_final = self._compute_hessian(x_new)  # 使用更新后的θ计算Hessian
            if self.beta == 0:
                self.H_t = self.H_t + H_new_final
            else:
                self.H_t = (1-self.beta) * self.H_t + self.beta * H_new_final
            
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
        
        # 更新累积运行时间
        self.total_time += (time.time() - iter_start_time)
        
        # 每20步记录一次时间和内存
        current_step = len(self.history['train_losses'])
        if current_step % 20 == 0:
            self.history['time_records'].append(self.total_time)
            self.history['time_record_steps'].append(current_step)
            
            self.history['memory_records'].append(get_memory_usage())
            self.history['memory_record_steps'].append(current_step)
        
        return train_loss
    
    def get_history(self):
        """获取训练历史"""
        self.history['total_time'] = self.total_time
        return self.history
