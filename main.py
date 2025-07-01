import numpy as np
import matplotlib.pyplot as plt
from data import DataGenerator
from model import LinearRegression
from online_update import OnlineGradientDescent
from utils import evaluate_model, plot_learning_curves, plot_true_vs_pred

def run_experiment(data_dim=5, n_samples=500, learning_rate=0.01, max_iter=100, seed=42):
    """
    运行在线梯度下降(OGD)实验
    
    参数:
        data_dim: 数据特征维度
        n_samples: 总样本数
        learning_rate: 学习率
        max_iter: 每轮最大迭代次数
        seed: 随机种子
        
    返回:
        experiment_results: 实验结果字典
    """
    print("初始化数据生成器...")
    data_generator = DataGenerator(data_dim=data_dim, n_samples=n_samples, seed=seed)
    
    print("初始化在线梯度下降算法...")
    ogd = OnlineGradientDescent(
        input_dim=data_dim, 
        learning_rate=learning_rate,
        max_iter=max_iter
    )
    
    # 获取测试集数据
    X_test, y_test = data_generator.get_test_data()
    
    print("开始在线学习过程...")
    t = 0
    done = False
    
    while not done:
        # 获取下一个训练样本
        x_t, y_t, done = data_generator.get_next_sample()
        
        if done:
            break
        
        # 获取当前时间步的完整历史数据集
        X_t, y_t_full = data_generator.get_current_dataset()
        
        # 更新模型
        train_loss = ogd.update(X_t, y_t_full, X_test, y_test)
        
        # 每10步输出一次进度
        if (t + 1) % 1 == 0:
            print(f"时间步 {t+1}: 训练损失 = {train_loss:.6f}")
        
        t += 1
    
    print("\n在线学习完成！")
    
    # 获取训练历史
    history = ogd.get_history()
    
    # 评估最终模型性能
    final_model = ogd.model
    train_mse, train_r2 = evaluate_model(final_model, X_t, y_t_full)
    test_mse, test_r2 = evaluate_model(final_model, X_test, y_test)
    
    print("\n最终模型性能评估:")
    print(f"训练集: MSE = {train_mse:.6f}, R^2 = {train_r2:.6f}")
    print(f"测试集: MSE = {test_mse:.6f}, R^2 = {test_r2:.6f}")
    
    # 比较估计权重与真实权重
    estimated_weights = final_model.get_weights()
    true_weights = data_generator.true_weights
    weight_error = np.linalg.norm(estimated_weights - true_weights)
    print(f"权重估计误差: {weight_error:.6f}")
    
    # 返回实验结果
    experiment_results = {
        'history': history,
        'final_model': final_model,
        'true_weights': true_weights,
        'estimated_weights': estimated_weights,
        'train_mse': train_mse,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'weight_error': weight_error
    }
    
    return experiment_results

def main():
    # 设置随机种子
    np.random.seed(42)
    
    # 设置实验参数
    data_dim = 3          # 特征维度
    n_samples = 200       # 样本数量
    learning_rate = 0.01  # 学习率
    max_iter = 50         # 每轮最大迭代次数
    
    print("=" * 50)
    print("在线梯度下降(OGD)实验")
    print("=" * 50)
    print(f"特征维度: {data_dim}")
    print(f"样本数量: {n_samples}")
    print(f"学习率: {learning_rate}")
    print(f"每轮最大迭代次数: {max_iter}")
    print("=" * 50)
    
    # 运行实验
    results = run_experiment(
        data_dim=data_dim,
        n_samples=n_samples,
        learning_rate=learning_rate,
        max_iter=max_iter
    )
    
    # 绘制学习曲线
    print("\n绘制学习曲线...")
    plot_learning_curves(results['history'])
    
    # 绘制真实权重与估计权重对比图
    print("绘制权重对比图...")
    plot_true_vs_pred(results['true_weights'], results['estimated_weights'])

if __name__ == "__main__":
    main() 