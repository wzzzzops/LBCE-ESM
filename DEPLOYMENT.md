# GitHub部署指南

## 部署内容

### 核心文件（必须提交）

1. **LBCE-ESM(BCpred).py** - 主模型文件
   - 包含完整的模型训练、评估和预测逻辑
   - 实现特征工程、模型训练、性能评估等功能

2. **robust_data_loader.py** - 数据加载模块
   - 提供鲁棒的数据加载功能
   - 支持多种数据集格式（LBtope, ABCPred, Blind387等）

### 配置文件（建议提交）

3. **requirements.txt** - Python依赖
   - numpy>=1.21.0
   - pandas>=1.3.0
   - scikit-learn>=1.0.0
   - xgboost>=1.5.0
   - scipy>=1.7.0
   - joblib>=1.1.0

4. **README.md** - 项目说明文档
   - 模型概述
   - 特征工程说明
   - 使用方法
   - 性能指标

5. **.gitignore** - Git忽略配置
   - 忽略Python缓存文件
   - 忽略虚拟环境
   - 忽略IDE配置文件

## 部署步骤

### 1. 初始化Git仓库

```bash
cd e:\工\b细胞表位预测\LBCE-ESM
git init
```

### 2. 添加远程仓库

```bash
git remote add origin https://github.com/your-username/LBCE-ESM.git
```

### 3. 添加文件到暂存区

```bash
git add .
```

### 4. 提交更改

```bash
git commit -m "Initial commit: LBCE-ESM model for B-cell epitope prediction"
```

### 5. 推送到GitHub

```bash
git branch -M main
git push -u origin main
```

## 注意事项

### 文件名特殊字符

注意：主模型文件名 `LBCE-ESM(BCpred).py` 包含括号 `()`，在某些系统中可能引起问题。

**解决方案**：
- 在Windows系统中通常没有问题
- 在Linux/Mac系统中，建议使用引号包裹文件名
- 在代码中引用时，使用相对路径或绝对路径

### 数据文件

**重要提示**：数据文件（如ESMC特征文件、序列文件等）**不应**提交到GitHub，原因：

1. 文件体积大（通常数百MB到数GB）
2. 数据可能包含敏感信息
3. 数据文件会频繁更新

**推荐做法**：
- 在README中提供数据下载链接
- 提供数据预处理脚本
- 使用 `.gitignore` 忽略数据文件

### 环境配置

建议在README中明确说明：

1. Python版本要求（建议3.8+）
2. 依赖安装命令
3. 环境配置说明

## 文件结构

```
LBCE-ESM/
├── LBCE-ESM(BCpred).py      # 主模型文件
├── robust_data_loader.py    # 数据加载模块
├── requirements.txt         # Python依赖
├── README.md               # 项目说明
├── .gitignore              # Git忽略配置
└── DEPLOYMENT.md          # 本部署指南
```

## 常见问题

### Q: 文件名包含括号会不会有问题？

A: 在大多数情况下没有问题，但在某些shell环境中可能需要使用引号。例如：
```bash
python "LBCE-ESM(BCpred).py"
```

### Q: 如何处理数据文件？

A: 
1. 将数据文件存储在云存储（如Google Drive、Dropbox）
2. 在README中提供下载链接
3. 创建数据下载脚本
4. 在 `.gitignore` 中添加数据文件模式

### Q: 如何更新模型？

A: 
1. 修改模型文件
2. 测试新版本
3. 提交更改：`git add . && git commit -m "Update model" && git push`

## 许可证

建议在仓库根目录添加LICENSE文件，选择合适的开源许可证。

## 贡献指南

建议创建CONTRIBUTING.md文件，说明如何贡献代码。
