import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, X, y):
    """
    评估模型性能
    
    参数:
        model: 线性回归模型
        X: 形状为 (n_samples, input_dim) 的特征矩阵
        y: 形状为 (n_samples,) 的标签向量
        
    返回:
        均方误差和R^2评分
    """
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    
    # 计算R^2评分
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    return mse, r2

def plot_learning_curves(history):
    """
    绘制学习曲线
    
    参数:
        history: 包含训练历史的字典
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='train_losses')
    if 'test_losses' in history and history['test_losses']:
        plt.plot(history['test_losses'], label='test_losses')
    plt.xlabel('time step')
    plt.ylabel('mse')
    plt.title('loss curve')
    plt.legend()
    
    # 绘制权重收敛曲线
    plt.subplot(1, 2, 2)
    weights = np.array(history['weights'])
    for i in range(weights.shape[1]):
        plt.plot(weights[:, i], label=f'weight {i+1}')
    plt.xlabel('time step')
    plt.ylabel('weights')
    plt.title('weights convergence curve')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_true_vs_pred(true_weights, estimated_weights):
    """
    绘制真实权重与估计权重的对比图
    
    参数:
        true_weights: 真实权重向量
        estimated_weights: 估计权重向量
    """
    plt.figure(figsize=(8, 6))
    
    indices = np.arange(len(true_weights))
    bar_width = 0.35
    
    plt.bar(indices - bar_width/2, true_weights, bar_width, label='true weights')
    plt.bar(indices + bar_width/2, estimated_weights, bar_width, label='estimated weights')
    
    plt.xlabel('feature index')
    plt.ylabel('weights')
    plt.title('true weights vs estimated weights')
    plt.xticks(indices)
    plt.legend()
    
    plt.tight_layout()
    plt.show() 