import os
import pandas as pd
from src.data_generator import generate_credit_data
from src.data_loader import load_give_me_some_credit
from src.eda import run_eda
from src.feature_engineering import feature_engineering, prepare_data
from src.model_training import train_models, save_models
from src.evaluation import evaluate_models


def main():
    # 创建目录
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. 加载数据（优先使用Kaggle数据集）
    print("=" * 60)
    print("步骤 1: 加载数据")
    print("=" * 60)

    df = load_give_me_some_credit('data')

    if df is None:
        print("\n使用合成数据替代...")
        df = generate_credit_data(n_samples=5000)
        df.to_csv('data/credit_data.csv', index=False, encoding='utf-8-sig')
        print(f"合成数据已生成: {df.shape[0]} 行, {df.shape[1]} 列")

    print(f"数据集: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"违约率: {df['是否违约'].mean():.2%}")

    # 2. 探索性数据分析
    print("\n" + "=" * 60)
    print("步骤 2: 探索性数据分析")
    print("=" * 60)
    run_eda(df, output_dir='outputs')

    # 3. 特征工程
    print("\n" + "=" * 60)
    print("步骤 3: 特征工程")
    print("=" * 60)
    df_processed, scaler, encoders = feature_engineering(df)
    X, y = prepare_data(df_processed)
    print(f"特征数量: {X.shape[1]}")
    print(f"特征列: {list(X.columns)}")

    # 4. 模型训练
    print("\n" + "=" * 60)
    print("步骤 4: 模型训练")
    print("=" * 60)
    results, X_train, X_test, y_train, y_test = train_models(X, y)

    # 5. 模型评估
    print("\n" + "=" * 60)
    print("步骤 5: 模型评估")
    print("=" * 60)
    comparison = evaluate_models(results, list(X.columns), output_dir='outputs')

    # 6. 保存模型
    print("\n" + "=" * 60)
    print("步骤 6: 保存模型")
    print("=" * 60)
    save_models(results, output_dir='models')

    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)
    print(f"输出图表: outputs/")
    print(f"训练模型: models/")


if __name__ == '__main__':
    main()
