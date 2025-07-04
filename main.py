import numpy as np
import matplotlib.pyplot as plt
import argparse
from data import DataGenerator
from model import LinearRegression
from online_update import OnlineGradientDescent, ImplicitOMD, OnePassOMD
from utils import evaluate_model, plot_learning_curves, plot_true_vs_pred, plot_algorithm_comparison

def run_experiment(method='ogd', data_dim=5, n_samples=500, learning_rate=0.01, max_iter=100, seed=42, beta=0, weight_decay=0.0, param_bound=50.0):
    """
    运行在线学习实验
    
    参数:
        method: 在线学习方法，可选值为 'ogd', 'implicit_omd', 'one_pass_omd'
        data_dim: 数据特征维度
        n_samples: 总样本数
        learning_rate: 学习率
        max_iter: 每轮最大迭代次数(仅用于OGD)
        seed: 随机种子
        beta: 指数加权平均系数，用于OMD方法中的Hessian矩阵更新
        weight_decay: 权重衰减系数，用于OGD的L2正则化
        param_bound: 参数范数上界，用于One-pass OMD
        
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
            max_iter=max_iter,
            weight_decay=weight_decay  # 添加权重衰减参数
        )
    elif method == 'implicit_omd':
        # 隐式在线镜像下降(Implicit OMD)
        online_learner = ImplicitOMD(
            input_dim=data_dim, 
            learning_rate=learning_rate,
            beta=beta  # 添加beta参数
        )
    elif method == 'one_pass_omd':
        # One-pass在线镜像下降(One-pass OMD)
        online_learner = OnePassOMD(
            input_dim=data_dim, 
            learning_rate=learning_rate,
            beta=beta,  # 添加beta参数
            param_bound=param_bound  # 添加参数范数上界
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
        
        # 每30步输出一次进度
        if (t + 1) % 30 == 0:
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

def run_algorithm_comparison(data_dim=3, n_samples=200, learning_rate=0.01, max_iter=50, seed=42, beta=0, weight_decay=0.0, param_bound=50.0):
    """
    运行算法比较实验，比较三种算法的性能
    
    参数:
        data_dim: 数据特征维度
        n_samples: 总样本数
        learning_rate: 学习率
        max_iter: 每轮最大迭代次数(仅用于OGD)
        seed: 随机种子
        beta: 指数加权平均系数，用于OMD方法中的Hessian矩阵更新
        weight_decay: 权重衰减系数，用于OGD的L2正则化
        param_bound: 参数范数上界，用于One-pass OMD
        
    返回:
        results_dict: 包含三种算法结果的字典
    """
    # 设置随机种子
    np.random.seed(seed)
    
    # 算法列表
    algorithms = ['ogd', 'implicit_omd', 'one_pass_omd']
    algorithm_names = {
        'ogd': 'Online Gradient Descent (OGD)',
        'implicit_omd': 'Implicit OMD',
        'one_pass_omd': 'One-pass OMD'
    }
    
    # 存储每种算法的结果
    results_dict = {}
    
    for method in algorithms:
        print("\n" + "=" * 50)
        print(f"运行算法: {algorithm_names[method]}")
        print("=" * 50)
        
        # 运行实验
        results = run_experiment(
            method=method,
            data_dim=data_dim,
            n_samples=n_samples,
            learning_rate=learning_rate,
            max_iter=max_iter,
            seed=seed,
            beta=beta,
            weight_decay=weight_decay,
            param_bound=param_bound
        )
        
        # 存储结果
        results_dict[algorithm_names[method]] = results
    
    return results_dict

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='在线学习方法比较实验')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'compare'],
                      help='实验模式: single (单个算法), compare (多算法比较)')
    parser.add_argument('--method', type=str, default='ogd', choices=['ogd', 'implicit_omd', 'one_pass_omd'],
                      help='在线学习方法: ogd (在线梯度下降), implicit_omd (隐式OMD), one_pass_omd (One-Pass OMD)')
    parser.add_argument('--dim', type=int, default=10, help='特征维度')
    parser.add_argument('--samples', type=int, default=500, help='样本数量')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--max_iter', type=int, default=50, help='每轮最大迭代次数(仅用于OGD)')
    parser.add_argument('--seed', type=int, default=100, help='随机种子')
    parser.add_argument('--beta', type=float, default=0.0, help='指数加权平均系数，用于OMD方法中的Hessian矩阵更新')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减系数，用于OGD的L2正则化')
    parser.add_argument('--param_bound', type=float, default=50.0, help='参数范数上界，用于One-pass OMD')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # 运行单个算法
        print("=" * 50)
        print(f"在线学习方法: {args.method}")
        print("=" * 50)
        print(f"特征维度: {args.dim}")
        print(f"样本数量: {args.samples}")
        print(f"学习率: {args.lr}")
        if args.method == 'ogd':
            print(f"每轮最大迭代次数: {args.max_iter}")
            print(f"权重衰减系数: {args.weight_decay}")
        if args.method in ['implicit_omd', 'one_pass_omd']:
            print(f"Hessian衰减系数(beta): {args.beta}")
        if args.method == 'one_pass_omd':
            print(f"参数范数上界: {args.param_bound}")
        print("=" * 50)
        
        # 运行实验
        results = run_experiment(
            method=args.method,
            data_dim=args.dim,
            n_samples=args.samples,
            learning_rate=args.lr,
            max_iter=args.max_iter,
            seed=args.seed,
            beta=args.beta,
            weight_decay=args.weight_decay,
            param_bound=args.param_bound
        )
        
        # 绘制学习曲线
        print("\n绘制学习曲线...")
        plot_learning_curves(results['history'])
        
        # 绘制真实权重与估计权重对比图
        print("绘制权重对比图...")
        plot_true_vs_pred(results['true_weights'], results['estimated_weights'])
    
    elif args.mode == 'compare':
        # 运行算法比较
        print("=" * 50)
        print("算法比较模式")
        print("=" * 50)
        print(f"特征维度: {args.dim}")
        print(f"样本数量: {args.samples}")
        print(f"学习率: {args.lr}")
        print(f"OGD每轮最大迭代次数: {args.max_iter}")
        print(f"OGD权重衰减系数: {args.weight_decay}")
        print(f"OMD方法Hessian衰减系数(beta): {args.beta}")
        print(f"One-pass OMD参数范数上界: {args.param_bound}")
        print("=" * 50)
        
        # 运行比较实验
        results_dict = run_algorithm_comparison(
            data_dim=args.dim,
            n_samples=args.samples,
            learning_rate=args.lr,
            max_iter=args.max_iter,
            seed=args.seed,
            beta=args.beta,
            weight_decay=args.weight_decay,
            param_bound=args.param_bound
        )
        
        # 绘制算法比较图
        print("\n绘制算法比较图...")
        plot_algorithm_comparison(results_dict)

if __name__ == "__main__":
    main() 