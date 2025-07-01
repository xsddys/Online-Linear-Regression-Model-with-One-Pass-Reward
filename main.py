import numpy as np
import matplotlib.pyplot as plt
import argparse
from data import DataGenerator
from model import LinearRegression
from online_update import OnlineGradientDescent, ImplicitOMD, OnePassOMD
from utils import evaluate_model, plot_learning_curves, plot_true_vs_pred

def run_experiment(method='ogd', data_dim=5, n_samples=500, learning_rate=0.01, max_iter=100, seed=42):
    """
    运行在线学习实验
    
    参数:
        method: 在线学习方法，可选值为 'ogd', 'implicit_omd', 'one_pass_omd'
        data_dim: 数据特征维度
        n_samples: 总样本数
        learning_rate: 学习率
        max_iter: 每轮最大迭代次数(仅用于OGD)
        seed: 随机种子
        
    返回:
        experiment_results: 实验结果字典
    """
    print("初始化数据生成器...")
    data_generator = DataGenerator(data_dim=data_dim, n_samples=n_samples, seed=seed)
    
    print(f"初始化在线学习方法: {method}...")
    if method == 'ogd':
        # 在线梯度下降(OGD)
        online_learner = OnlineGradientDescent(
            input_dim=data_dim, 
            learning_rate=learning_rate,
            max_iter=max_iter
        )
    elif method == 'implicit_omd':
        # 隐式在线镜像下降(Implicit OMD)
        online_learner = ImplicitOMD(
            input_dim=data_dim, 
            learning_rate=learning_rate
        )
    elif method == 'one_pass_omd':
        # 一次通过在线镜像下降(One-pass OMD)
        online_learner = OnePassOMD(
            input_dim=data_dim, 
            learning_rate=learning_rate
        )
    else:
        raise ValueError(f"不支持的方法: {method}")
    
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
        train_loss = online_learner.update(X_t, y_t_full, X_test, y_test)
        
        # 每1步输出一次进度
        if (t + 1) % 20 == 0:
            print(f"时间步 {t+1}: 训练损失 = {train_loss:.6f}")
        
        t += 1
    
    print("\n在线学习完成！")
    
    # 获取训练历史
    history = online_learner.get_history()
    
    # 评估最终模型性能
    final_model = online_learner.model
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
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='在线学习方法比较实验')
    parser.add_argument('--method', type=str, default='ogd', choices=['ogd', 'implicit_omd', 'one_pass_omd'],
                      help='在线学习方法: ogd (在线梯度下降), implicit_omd (隐式OMD), one_pass_omd (一次通过OMD)')
    parser.add_argument('--dim', type=int, default=3, help='特征维度')
    parser.add_argument('--samples', type=int, default=200, help='样本数量')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--max_iter', type=int, default=50, help='每轮最大迭代次数(仅用于OGD)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    print("=" * 50)
    print(f"在线学习方法: {args.method}")
    print("=" * 50)
    print(f"特征维度: {args.dim}")
    print(f"样本数量: {args.samples}")
    print(f"学习率: {args.lr}")
    if args.method == 'ogd':
        print(f"每轮最大迭代次数: {args.max_iter}")
    print("=" * 50)
    
    # 运行实验
    results = run_experiment(
        method=args.method,
        data_dim=args.dim,
        n_samples=args.samples,
        learning_rate=args.lr,
        max_iter=args.max_iter,
        seed=args.seed
    )
    
    # 绘制学习曲线
    print("\n绘制学习曲线...")
    plot_learning_curves(results['history'])
    
    # 绘制真实权重与估计权重对比图
    print("绘制权重对比图...")
    plot_true_vs_pred(results['true_weights'], results['estimated_weights'])

if __name__ == "__main__":
    main() 