# 银行信用风险评估

基于机器学习的银行信用风险评估项目，使用 Kaggle **Give Me Some Credit** 数据集，对比多种经典ML模型的预测效果。

## 项目简介

本项目实现了完整的机器学习流程：

- 数据加载与探索性分析 (EDA)
- 特征工程与数据预处理
- 多模型训练与对比（Logistic Regression / Random Forest / XGBoost / LightGBM）
- 模型评估与可视化

## 数据集

使用 [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) 数据集：

- 15万条客户记录
- 10个特征（信用使用率、年龄、逾期次数、负债率、收入等）
- 目标变量：是否违约（6.68% 违约率）

## 项目结构

```
credit_risk/
├── data/                    # 数据目录
├── models/                  # 训练好的模型
├── outputs/                 # 可视化图表
├── src/
│   ├── data_generator.py    # 合成数据生成
│   ├── data_loader.py       # Kaggle数据加载
│   ├── eda.py               # 探索性数据分析
│   ├── feature_engineering.py # 特征工程
│   ├── model_training.py    # 模型训练
│   └── evaluation.py        # 模型评估
├── main.py                  # 主入口
└── requirements.txt         # 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

从 [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data) 下载 `cs-training.csv`，放到 `data/` 目录下。

### 3. 运行项目

```bash
python main.py
```

## 模型效果

| 模型 | 准确率 | F1 (违约类) | AUC |
|------|--------|-------------|-----|
| Logistic Regression | 0.756 | 0.272 | 0.795 |
| Random Forest | 0.861 | 0.381 | 0.857 |
| XGBoost | 0.920 | 0.384 | 0.843 |
| LightGBM | 0.927 | 0.380 | 0.845 |

## 输出示例

- ROC 曲线对比
- 混淆矩阵
- 特征重要性排序
- 特征相关性热力图

## 技术栈

- Python 3.11
- scikit-learn
- XGBoost / LightGBM
- pandas / numpy
- matplotlib / seaborn
- imbalanced-learn (SMOTE)

## License

MIT
