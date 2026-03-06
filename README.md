# LBCE-ESM (B细胞表位预测模型)

基于XGBoost和ESMC-600M特征的B细胞表位预测模型。

## 模型概述

LBCE-ESM (Enhanced Sequence Modeling) 是一个结合了深度学习表征和传统生物信息学特征的B细胞表位预测模型。该模型使用XGBoost分类器，结合ESMC-600M预训练特征和多种传统特征，实现高精度的B细胞表位预测。

## 特征工程

模型使用以下特征组合：

- **ESMC-600M**: 预训练蛋白质语言模型特征 (1152维)
- **AAC**: 氨基酸组成特征 (20维)
- **DPC**: 二肽组成特征 (400维)
- **TPC**: 三肽组成特征 (100维)
- **理化性质**: 疏水性、电荷、分子量、极性 (4维)
- **序列统计**: 长度、组成比例等 (4维)

## 模型架构

- **主分类器**: XGBoost
- **特征选择**: SelectKBest (k=1300)
- **降维**: PCA (n_components=650)
- **优化策略**: 多指标阈值优化 (ACC, MCC, AUROC)

## 依赖安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```python
from LBCE-ESM(BCpred) import main

# 运行模型
model, scaler, selector, pca, results = main()
```

### 自定义使用

```python
from LBCE-ESM(BCpred) import load_and_combine_features, XGBClassifier
import numpy as np

# 加载数据
X, y = load_and_combine_features(
    esmc_filepath='esmc_600mfeatures/BCPreds/BCPreds_CLS_fea.txt',
    seq_folder='data/datasets/BCPreds',
    dataset_name='BCPreds'
)

# 训练模型
model = XGBClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    random_state=42
)
model.fit(X, y)

# 预测
predictions = model.predict(X_test)
```

## 数据集

模型支持以下数据集：

- **BCPreds**: 训练集
- **Chen**: 测试集
- **ABCPred**: 测试集
- **Blind387**: 测试集
- **iBCE-EL_independent**: 测试集
- **iBCE-EL_training**: 测试集
- **LBtope**: 测试集

### 数据集结构

```
data/
└── datasets/
    ├── ABCPred/
    ├── BCPreds/
    ├── Blind387/
    ├── Chen/
    ├── LBtope/
    ├── iBCE-EL_independent/
    └── iBCE-EL_training/

esmc_600mfeatures/
└── ABCPred/
    ├── BCPreds/
    ├── Blind387/
    ├── Chen/
    ├── LBtope/
    ├── iBCE-EL_independent/
    └── iBCE-EL_training/
```

## 性能指标

模型在测试集上的性能指标：

- **ACC**: Accuracy (准确率)
- **Pre**: Precision (精确率)
- **Sn**: Sensitivity (敏感性/召回率)
- **F1**: F1-Score (F1分数)
- **MCC**: Matthews Correlation Coefficient (马修斯相关系数)
- **AUROC**: Area Under ROC Curve (ROC曲线下面积)

## 项目结构

```
LBCE-ESM/
├── LBCE-ESM(BCpred).py      # 主模型文件
├── robust_data_loader.py    # 数据加载模块
├── requirements.txt         # Python依赖
├── README.md               # 项目说明
├── .gitignore              # Git忽略配置
├── data/                   # 数据集文件
│   └── datasets/          # 原始数据
│       ├── ABCPred/
│       ├── BCPreds/
│       ├── Blind387/
│       ├── Chen/
│       ├── LBtope/
│       ├── iBCE-EL_independent/
│       └── iBCE-EL_training/
└── esmc_600mfeatures/     # ESMC-600M特征文件
    ├── ABCPred/
    ├── BCPreds/
    ├── Blind387/
    ├── Chen/
    ├── LBtope/
    ├── iBCE-EL_independent/
    └── iBCE-EL_training/
```

## 注意事项

### 数据文件

- **原始数据**位于 `data/datasets/` 目录
- **ESMC特征**位于 `esmc_600mfeatures/` 目录
- 数据文件较大（通常数百MB），请确保有足够的磁盘空间
- 如需从GitHub下载，请使用 LFS (Large File Storage)

### Git配置

本项目已配置 `.gitignore`，以下文件不会被提交到Git：

- 所有 `.txt`、`.csv`、`.fasta` 数据文件
- ESMC特征文件（`.txt`）
- 生成的模型文件（`.pkl`）
- 虚拟环境和缓存文件

## 许可证

本项目仅供研究使用。

## 引用

如使用本模型，请引用相关论文。

## 联系方式

如有问题，请联系项目维护者。
