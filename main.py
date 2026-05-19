import os
import pandas as pd
from src.data_generator import generate_credit_data
from src.data_loader import load_give_me_some_credit
from src.eda import run_eda
from src.feature_engineering import feature_engineering, prepare_data
from src.model_training import train_models, save_models
from src.evaluation import evaluate_models
from src.data_quality import generate_quality_report
from src.logger import setup_logger
import logging


def main():
    # 初始化日志系统
    setup_logger()
    logger = logging.getLogger(__name__)

    # 创建目录
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # 1. 加载数据（优先使用Kaggle数据集）
    logger.info("=" * 60)
    logger.info("步骤 1: 加载数据")
    logger.info("=" * 60)

    df = load_give_me_some_credit('data')

    if df is None:
        logger.info("\n使用合成数据替代...")
        df = generate_credit_data(n_samples=5000)
        df.to_csv('data/credit_data.csv', index=False, encoding='utf-8-sig')
        logger.info(f"合成数据已生成: {df.shape[0]} 行, {df.shape[1]} 列")

    logger.info(f"数据集: {df.shape[0]} 行, {df.shape[1]} 列")
    logger.info(f"违约率: {df['是否违约'].mean():.2%}")

    # 2. 数据质量检查
    logger.info("\n" + "=" * 60)
    logger.info("步骤 2: 数据质量检查")
    logger.info("=" * 60)
    quality_report = generate_quality_report(df, output_dir='outputs')

    # 3. 探索性数据分析
    logger.info("\n" + "=" * 60)
    logger.info("步骤 3: 探索性数据分析")
    logger.info("=" * 60)
    run_eda(df, output_dir='outputs')

    # 4. 特征工程
    logger.info("\n" + "=" * 60)
    logger.info("步骤 4: 特征工程")
    logger.info("=" * 60)
    df_processed, scaler, encoders = feature_engineering(df)
    X, y = prepare_data(df_processed)
    logger.info(f"特征数量: {X.shape[1]}")
    logger.info(f"特征列: {list(X.columns)}")

    # 5. 模型训练
    logger.info("\n" + "=" * 60)
    logger.info("步骤 5: 模型训练")
    logger.info("=" * 60)
    results, X_train, X_test, y_train, y_test = train_models(X, y)

    # 6. 模型评估
    logger.info("\n" + "=" * 60)
    logger.info("步骤 6: 模型评估")
    logger.info("=" * 60)
    comparison = evaluate_models(results, list(X.columns), output_dir='outputs')

    # 7. SHAP模型解释（可选）
    logger.info("\n" + "=" * 60)
    logger.info("步骤 7: SHAP模型解释")
    logger.info("=" * 60)
    try:
        from src.shap_analysis import explain_model
        # 为XGBoost和LightGBM生成SHAP分析
        for name in ['XGBoost', 'LightGBM']:
            if name in results:
                explain_model(
                    results[name]['model'],
                    X_train, X_test,
                    list(X.columns),
                    name,
                    output_dir='outputs'
                )
    except ImportError:
        logger.warning("SHAP未安装，跳过模型解释步骤")
    except Exception as e:
        logger.error(f"SHAP分析失败: {e}")

    # 8. 保存模型
    logger.info("\n" + "=" * 60)
    logger.info("步骤 8: 保存模型")
    logger.info("=" * 60)
    save_models(results, output_dir='models')

    logger.info("\n" + "=" * 60)
    logger.info("全部完成!")
    logger.info("=" * 60)
    logger.info(f"输出图表: outputs/")
    logger.info(f"训练模型: models/")
    logger.info(f"日志文件: logs/")


if __name__ == '__main__':
    main()
