# 在线学习方法实现与比较

## 项目简介
本项目实现了多种在线学习方法，包括在线梯度下降法(Online Gradient Descent, OGD)、隐式在线镜像下降法(Implicit Online Mirror Descent, Implicit OMD)和一次通过在线镜像下降法(One-pass Online Mirror Descent, One-pass OMD)，并对它们在线性回归问题上的表现进行了比较。

## 方法介绍
1. **在线梯度下降(OGD)**: 使用最大似然估计(MLE)进行迭代下降。在每一轮接收到新数据点后，重新拟合整个历史数据集。支持L2正则化（权重衰减）以提高鲁棒性。

2. **隐式在线镜像下降(Implicit OMD)**: 使用Hessian矩阵代替历史数据点的信息，避免存储完整的数据集。通过求解以下优化问题更新参数：
   
   $\bar{\theta}_{t+1}=\underset{\theta\in\Theta}{\arg\min}\left\{\ell_t(\theta)+\frac{1}{2\eta}\left\|\theta-\bar{\theta}_t\right\|_{\overline{\mathcal{H}}_t}^2\right\}$
   
   其中$\ell_t(\theta)$是瞬时损失项，表示当前样本的完整损失函数。支持指数加权平均（beta参数）来调整历史Hessian矩阵的影响。

3. **一次通过在线镜像下降(One-pass OMD)**: 通过二阶泰勒展开进一步优化，仅保留当前点的梯度部分和二阶Hessian曲率信息，求解以下优化问题：
   
   $\widetilde{\theta}_{t+1}=\underset{\theta\in\Theta}{\arg\min}\left\{\left\langle g_t\left(\widetilde{\theta}_t\right),\theta\right\rangle+\frac{1}{2\eta}\left\|\theta-\widetilde{\theta}_t\right\|_{\widetilde{\mathcal{H}}_t}^2\right\}$
   
   其中$\widetilde{\mathcal{H}}_t=\mathcal{H}_t + \eta H_t(\widetilde{\theta}_t)$，$g_t$是当前点的梯度。该方法实现了每一代O(1)的计算和存储效率。支持参数范数约束和Hessian矩阵的指数加权平均更新。

## 项目结构
```
.
├── README.md            # 项目说明文档
├── data.py              # 数据生成和管理模块
├── model.py             # 模型定义模块
├── online_update.py     # 在线学习算法实现模块
├── main.py              # 主程序入口
└── utils.py             # 工具函数模块
```

## 功能特点
- 生成服从高斯分布的随机数据
- 实现增量式数据输入机制
- 支持三种不同的在线学习算法
- 提供训练集和测试集的性能评估
- 可视化学习过程和结果
- 支持算法性能比较：测试损失、运行时间和内存占用对比
- 支持OGD的L2正则化（权重衰减）
- 支持OMD方法的Hessian矩阵指数加权平均更新
- 支持One-pass OMD的参数范数约束调整

## 使用方法
### 单个算法运行
```bash
# 使用在线梯度下降(OGD)
python main.py --mode single --method ogd

# 使用隐式在线镜像下降(Implicit OMD)
python main.py --mode single --method implicit_omd

# 使用一次通过在线镜像下降(One-pass OMD)
python main.py --mode single --method one_pass_omd
```

### 算法性能比较
```bash
# 运行三种算法并比较它们的性能
python main.py --mode compare
```

### 可调整的命令行参数
- `--mode`: 实验模式，可选 'single' (运行单个算法), 'compare' (多算法比较)
- `--method`: 在线学习方法，可选 'ogd', 'implicit_omd', 'one_pass_omd'
- `--dim`: 数据特征维度，默认为10
- `--samples`: 总样本数量，默认为500
- `--lr`: 学习率，默认为0.01
- `--max_iter`: OGD方法的每轮最大迭代次数，默认为50
- `--seed`: 随机种子，默认为100
- `--beta`: 指数加权平均系数，用于OMD方法中的Hessian矩阵更新，默认为0.0
- `--weight_decay`: 权重衰减系数，用于OGD的L2正则化，默认为0.0
- `--param_bound`: 参数范数上界，用于One-pass OMD，默认为50.0

例如：
```bash
# 运行单个算法，调整参数
python main.py --mode single --method implicit_omd --dim 10 --samples 300 --lr 0.01 --beta 0.1

python main.py --mode single --method implicit_omd --dim 10 --samples 300 --lr 0.01

# One-Pass OMD
python main.py --mode single --method one_pass_omd --dim 10 --samples 300 --lr 0.01 --beta 0.5

python main.py --mode single --method one_pass_omd --dim 10 --samples 300 --lr 0.01

# 运行OGD并添加L2正则化
python main.py --mode single --method ogd --weight_decay 0.01

# 运行算法比较，调整参数
python main.py --mode compare --dim 5 --samples 300 --lr 0.005 --beta 0.1 --weight_decay 0.01
```

## 依赖库
- numpy
- matplotlib
- argparse
- psutil（用于内存使用监控）

## 实验结果
运行实验后，程序会输出每个方法的训练过程和最终性能指标，并生成各种可视化图表：

### 单个算法模式
- 学习曲线：训练损失和测试损失随时间步的变化
- 权重收敛曲线：模型参数随时间步的变化
- 权重对比图：真实权重与估计权重的对比

### 算法比较模式
- 测试损失对比图：三种算法的测试损失随时间步的变化
- 运行时间对比图：三种算法的累积运行时间随时间步的变化
- 内存占用对比图：三种算法的内存占用随时间步的变化

## 性能对比
三种算法的主要区别：

1. **时间复杂度**:
   - OGD: 每次更新需要重新拟合整个历史数据集，计算复杂度随数据集大小增长 O(t)
   - Implicit OMD: 每次只处理最新的数据点，但需要迭代求解隐式更新方程，复杂度为 O(k)，其中k为迭代次数
   - One-pass OMD: 进一步优化的OMD方法，通过闭式解实现O(1)的计算效率

2. **空间复杂度**:
   - OGD: 需要存储所有历史数据点，空间复杂度为O(t)
   - Implicit OMD: 只需存储Hessian矩阵和当前参数，空间复杂度为O(d²)，其中d为特征维度
   - One-pass OMD: 同样只需存储Hessian矩阵和当前参数，空间复杂度为O(d²)

3. **正则化与稳定性**:
   - OGD: 可通过权重衰减(L2正则化)提高稳定性，但在高维特征空间中仍可能不稳定
   - OMD方法: 通过累积Hessian矩阵提供自适应正则化效果，可使用beta参数控制历史信息的衰减速率
   - One-pass OMD: 额外提供参数范数约束，进一步增强稳定性

通过运行比较模式，可以直观地观察到这些差异在实际应用中的表现。 