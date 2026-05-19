# 银行信用风险评估

基于机器学习的银行信用风险评估项目，使用 Kaggle **Give Me Some Credit** 数据集，对比多种经典ML模型的预测效果。

## 项目简介

本项目实现了完整的机器学习流程：

- 数据加载与探索性分析 (EDA)
- 数据质量检查（缺失值、异常值、重复值检测）
- 特征工程与数据预处理
- 超参数调优（Optuna贝叶斯优化）
- 多模型训练与对比（Logistic Regression / Random Forest / XGBoost / LightGBM）
- 模型评估与可视化
- SHAP模型解释性分析
- 日志系统
- 单元测试
- FastAPI模型部署

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
├── logs/                    # 日志文件
├── tests/                   # 单元测试
│   ├── test_data_generator.py
│   ├── test_feature_engineering.py
│   └── test_model_training.py
├── src/
│   ├── data_generator.py    # 合成数据生成
│   ├── data_loader.py       # Kaggle数据加载
│   ├── data_quality.py      # 数据质量检查
│   ├── eda.py               # 探索性数据分析
│   ├── feature_engineering.py # 特征工程
│   ├── hyperparameter_tuning.py # 超参数调优
│   ├── logger.py            # 日志系统
│   ├── model_training.py    # 模型训练
│   ├── shap_analysis.py     # SHAP模型解释
│   └── evaluation.py        # 模型评估
├── app.py                   # FastAPI部署服务
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

### 4. 运行单元测试

```bash
pytest tests/ -v
```

### 5. 启动API服务

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API文档: http://localhost:8000/docs

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
- Optuna (超参数优化)
- SHAP (模型解释)
- FastAPI (API部署)
- pandas / numpy
- matplotlib / seaborn
- imbalanced-learn (SMOTE)
- pytest (单元测试)

## License

MIT
