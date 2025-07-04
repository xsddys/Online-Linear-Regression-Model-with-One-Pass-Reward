import numpy as np

class DataGenerator:
    """
    数据生成器：生成符合线性模型的高斯分布数据，并支持增量式数据提供
    """
    def __init__(self, data_dim=5, n_samples=1000, test_ratio=0.2, noise_scale=0.05, seed=100):
        """
        初始化数据生成器
        
        参数:
            data_dim: 特征维度
            n_samples: 总样本数
            test_ratio: 测试集占比
            noise_scale: 噪声尺度
            seed: 随机种子
        """
        self.data_dim = data_dim
        self.n_samples = n_samples
        self.test_ratio = test_ratio
        self.noise_scale = noise_scale
        self.rng = np.random.RandomState(seed)
        
        # 生成真实权重向量 w*
        self.true_weights = self.rng.normal(0, 1, data_dim)
        
        # 生成所有数据
        self._generate_data()
        
        # 在线学习相关
        self.current_index = 0
        self.train_data_seen = []
        self.train_labels_seen = []
    
    def _generate_data(self):
        """生成完整的数据集"""
        # 生成特征矩阵 X
        self.X = self.rng.normal(0, 1, (self.n_samples, self.data_dim))
        
        # 生成标签 y = X*w* + ε, ε ~ N(0, noise_scale)
        noise = self.rng.normal(0, self.noise_scale, self.n_samples)
        self.y = np.dot(self.X, self.true_weights) + noise
        
        # 划分训练集和测试集
        n_test = int(self.n_samples * self.test_ratio)
        n_train = self.n_samples - n_test
        
        self.X_train = self.X[:n_train]
        self.y_train = self.y[:n_train]
        self.X_test = self.X[n_train:]
        self.y_test = self.y[n_train:]
        
        self.n_train = n_train
    
    def get_test_data(self):
        """获取测试集数据"""
        return self.X_test, self.y_test
    
    def get_next_sample(self):
        """
        获取下一个训练样本
        
        返回:
            x_t: 当前时间步的特征向量
            y_t: 当前时间步的标签
            done: 是否已经遍历完所有训练样本
        """
        if self.current_index >= self.n_train:
            return None, None, True
        
        x_t = self.X_train[self.current_index]
        y_t = self.y_train[self.current_index]
        
        # 将当前样本加入已见样本集
        self.train_data_seen.append(x_t)
        self.train_labels_seen.append(y_t)
        
        self.current_index += 1
        done = self.current_index >= self.n_train
        
        return x_t, y_t, done
    
    def get_current_dataset(self):
        """
        获取当前时间步的完整已见数据集
        
        返回:
            X_t: 当前已见的特征矩阵
            y_t: 当前已见的标签向量
        """
        X_t = np.array(self.train_data_seen)
        y_t = np.array(self.train_labels_seen)
        return X_t, y_t
    
    def reset(self):
        """重置数据生成器的状态"""
        self.current_index = 0
        self.train_data_seen = []
        self.train_labels_seen = [] 