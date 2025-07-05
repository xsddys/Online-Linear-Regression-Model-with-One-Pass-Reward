# 在线学习方法实现与比较

## 项目简介
Implementation of "Provably Efficient Online RLHF with One-Pass Reward Modeling" for Linear Model 

本项目基于["Provably Efficient Online RLHF with One-Pass Reward Modeling"](https://arxiv.org/pdf/2502.07193v2) 实现了多种在线学习方法，包括在线梯度下降法(Online Gradient Descent, OGD)、隐式在线镜像下降法(Implicit Online Mirror Descent, Implicit OMD)和一次通过在线镜像下降法(One-pass Online Mirror Descent, One-pass OMD)，并对它们在线性回归问题上的表现进行了比较。

## 方法介绍
1. **在线梯度下降(OGD)**: 使用最大似然估计(MLE)进行迭代下降。在每一轮接收到新数据点后，重新拟合整个历史数据集。支持L2正则化（权重衰减）以提高鲁棒性。

2. **隐式在线镜像下降(Implicit OMD)**: [Logistic Bandits[Faury et al., 2022]](https://proceedings.mlr.press/v151/faury22a) 在Linear Model中的实现。 使用Hessian矩阵代替历史数据点的信息，避免存储完整的数据集。
   

3. **一次通过在线镜像下降(One-pass OMD)**: 通过二阶泰勒展开进一步优化，仅保留当前点的梯度部分和二阶Hessian曲率信息，实现O(1)的时间复杂度。

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
相对于原论文的要求，添加了以下可调整的参数
- 支持OGD的L2正则化（权重衰减），测试正则化对于高维空间中OGD的鲁棒性的影响（×）
- 支持OMD方法的Hessian矩阵指数加权平均（EMA）更新，测试近期数据高权重的平均方法的影响（×）
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

# 运行OGD并添加L2正则化 （发现其实加了正则化也没用）
python main.py --mode single --method ogd --weight_decay 0.01

# 运行算法比较，调整参数
python main.py --mode compare --dim 10 --samples 300 --lr 0.01 
```

dim 与 步长影响巨大！！

## 依赖库
- numpy
- matplotlib
- argparse
- psutil（用于内存使用监控）

## 性能对比
三种算法的主要区别：

1. **时间复杂度**:
   - OGD: 每次更新需要重新拟合整个历史数据集，计算复杂度随数据集大小增长 O(tlogt)
   - Implicit OMD: 每次只处理最新的数据点，但需要迭代求解隐式更新方程，复杂度为 O(logt)
   - One-pass OMD: 进一步优化的OMD方法，通过闭式解实现O(1)的计算效率

2. **空间复杂度**:
   - OGD: 需要存储所有历史数据点，空间复杂度为O(t)
   - Implicit OMD: 只需存储Hessian矩阵和当前参数，空间复杂度为O(d²)，其中d为特征维度
   - One-pass OMD: 同样只需存储Hessian矩阵和当前参数，空间复杂度为O(d²)

3. **正则化与稳定性**:
   - OGD: 尝试了使用权重衰减(L2正则化)提高稳定性，但经过测试其高维特征空间中仍不稳定
   - OMD方法: 通过累积Hessian矩阵提供自适应正则化效果，可使用beta参数控制历史信息的衰减速率
   - One-pass OMD: 额外提供参数范数约束，进一步增强稳定性

通过运行比较模式，可以直观地观察到这些差异在实际应用中的表现。 