# 快速开始指南

## 1. 环境准备

```bash
# 克隆或下载本项目
cd e:\工\b细胞表位预测\LBCE-ESM

# 创建虚拟环境（可选但推荐）
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 2. 运行模型

### 方式一：直接运行（使用默认配置）

```bash
python "LBCE-ESM(BCpred).py"
```

### 方式二：在Python中导入使用

```python
import sys
sys.path.append('.')

from LBCE-ESM(BCpred) import main

# 运行模型
model, scaler, selector, pca, results = main()

# 查看结果
print(results)
```

### 方式三：自定义数据集

```python
import sys
sys.path.append('.')

from LBCE-ESM(BCpred) import load_and_combine_features, XGBClassifier
from sklearn.model_selection import train_test_split

# 加载数据（使用本地路径）
X, y = load_and_combine_features(
    esmc_filepath='esmc_600mfeatures/BCPreds/BCPreds_CLS_fea.txt',
    seq_folder='data/datasets/BCPreds',
    dataset_name='BCPreds'
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练模型
model = XGBClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 评估
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

print(f"ACC:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Pre:  {precision_score(y_test, y_pred):.4f}")
print(f"Sn:   {recall_score(y_test, y_pred):.4f}")
print(f"F1:   {f1_score(y_test, y_pred):.4f}")
print(f"MCC:  {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"AUROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

## 3. 数据文件说明

### 原始数据
- **位置**: `data/datasets/`
- **内容**: 包含正负样本序列文件
- **格式**: `.txt`, `.fasta`, `.csv`

### ESMC特征
- **位置**: `esmc_600mfeatures/`
- **内容**: ESMC-600M预训练模型提取的特征
- **格式**: `.txt` (每行: 序列ID,特征1,特征2,...)

## 4. 项目结构

```
LBCE-ESM/
├── LBCE-ESM(BCpred).py      # 主模型文件
├── robust_data_loader.py    # 数据加载模块
├── requirements.txt         # Python依赖
├── README.md               # 项目说明
├── QUICKSTART.md          # 本快速开始指南
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

## 5. 常见问题

### Q: 数据文件太大，下载太慢怎么办？

A: 
1. 可以只下载需要的数据集
2. 使用Git LFS (Large File Storage)
3. 从其他来源获取数据文件

### Q: 如何使用其他数据集？

A: 修改 `load_and_combine_features()` 函数中的路径参数：
```python
X, y = load_and_combine_features(
    esmc_filepath='esmc_600mfeatures/YourDataset/YourDataset_CLS_fea.txt',
    seq_folder='data/datasets/YourDataset',
    dataset_name='YourDataset'
)
```

### Q: 如何调整模型参数？

A: 修改 `main()` 函数中的XGBoost参数：
```python
model = XGBClassifier(
    n_estimators=1000,    # 树的数量
    max_depth=8,          # 树的最大深度
    learning_rate=0.05,   # 学习率
    random_state=42
)
```

### Q: 如何在其他测试集上评估？

A: 模型会自动在所有配置的数据集上进行评估。查看 `main()` 函数中的 `datasets_config` 配置。

## 6. 下一步

- 阅读 [README.md](README.md) 了解详细信息
- 阅读 [DEPLOYMENT.md](DEPLOYMENT.md) 了解部署到GitHub的详细步骤
- 查看代码注释了解实现细节

## 7. 技术支持

如有问题，请检查：
1. Python版本是否 >= 3.8
2. 所有依赖是否正确安装
3. 数据文件路径是否正确
4. 查看错误信息进行调试
