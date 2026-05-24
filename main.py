import os
import logging
from sklearn.model_selection import train_test_split
from src.config import get_config, load_config
from src.data_generator import generate_credit_data
from src.data_loader import load_give_me_some_credit
from src.eda import run_eda
from src.feature_engineering import (
    prepare_data, feature_engineering_fit_transform,
    feature_engineering_transform, save_preprocessor
)
from src.model_training import get_default_models, train_models, save_models
from src.evaluation import evaluate_models
from src.data_quality import generate_quality_report
from src.logger import setup_logger


def build_models(config, random_state):
    """根据配置构建模型字典"""
    models = {}
    mc = config.get('models', {})

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    lr_params = mc.get('logistic_regression', {})
    models['Logistic Regression'] = LogisticRegression(
        max_iter=lr_params.get('max_iter', 1000), random_state=random_state
    )

    rf_params = mc.get('random_forest', {})
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=rf_params.get('n_estimators', 200),
        max_depth=rf_params.get('max_depth', 10),
        random_state=random_state, n_jobs=-1
    )

    xgb_params = mc.get('xgboost', {})
    models['XGBoost'] = XGBClassifier(
        n_estimators=xgb_params.get('n_estimators', 200),
        max_depth=xgb_params.get('max_depth', 6),
        learning_rate=xgb_params.get('learning_rate', 0.1),
        random_state=random_state,
        eval_metric=xgb_params.get('eval_metric', 'logloss')
    )

    lgbm_params = mc.get('lightgbm', {})
    models['LightGBM'] = LGBMClassifier(
        n_estimators=lgbm_params.get('n_estimators', 200),
        max_depth=lgbm_params.get('max_depth', 6),
        learning_rate=lgbm_params.get('learning_rate', 0.1),
        random_state=random_state,
        verbose=lgbm_params.get('verbose', -1)
    )

    return models


def main():
    config = load_config()
    setup_logger()
    logger = logging.getLogger(__name__)

    paths = config['paths']
    data_cfg = config['data']
    train_cfg = config['training']
    random_state = data_cfg.get('random_state', 42)

    # 创建目录
    for d in paths.values():
        os.makedirs(d, exist_ok=True)

    # 1. 加载数据
    logger.info("=" * 60)
    logger.info("步骤 1: 加载数据")
    logger.info("=" * 60)

    df = load_give_me_some_credit(paths['data_dir'])

    if df is None:
        logger.info("使用合成数据替代...")
        df = generate_credit_data(n_samples=data_cfg.get('synthetic_samples', 5000))
        df.to_csv(os.path.join(paths['data_dir'], 'credit_data.csv'),
                  index=False, encoding='utf-8-sig')
        logger.info(f"合成数据已生成: {df.shape[0]} 行, {df.shape[1]} 列")

    logger.info(f"数据集: {df.shape[0]} 行, {df.shape[1]} 列")
    logger.info(f"违约率: {df['是否违约'].mean():.2%}")

    # 2. 数据质量检查
    logger.info("\n" + "=" * 60)
    logger.info("步骤 2: 数据质量检查")
    logger.info("=" * 60)
    generate_quality_report(df, output_dir=paths['output_dir'])

    # 3. 探索性数据分析
    logger.info("\n" + "=" * 60)
    logger.info("步骤 3: 探索性数据分析")
    logger.info("=" * 60)
    run_eda(df, output_dir=paths['output_dir'])

    # 4. 分离特征和目标，切分训练/测试集
    logger.info("\n" + "=" * 60)
    logger.info("步骤 4: 数据切分与特征工程")
    logger.info("=" * 60)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_cfg.get('test_size', 0.2),
        random_state=random_state, stratify=y
    )
    logger.info(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 特征工程：在训练集上fit，转换训练集和测试集
    X_train_processed, scaler, encoders, feature_columns = \
        feature_engineering_fit_transform(X_train, target='是否违约')
    X_test_processed = feature_engineering_transform(
        X_test, scaler, encoders, feature_columns, target='是否违约'
    )

    # 保存预处理器
    save_preprocessor(scaler, encoders, feature_columns, output_dir=paths['model_dir'])
    logger.info(f"预处理器已保存到 {paths['model_dir']}/preprocessor.pkl")
    logger.info(f"特征数量: {len(feature_columns)}")
    logger.info(f"特征列: {feature_columns}")

    # 5. 构建模型（可选：超参数调优）
    models = build_models(config, random_state)

    if train_cfg.get('use_hyperparameter_tuning', False):
        logger.info("\n" + "=" * 60)
        logger.info("步骤 5: 超参数调优")
        logger.info("=" * 60)
        from src.hyperparameter_tuning import tune_all_models
        best_params = tune_all_models(
            X_train_processed, y_train,
            n_trials=train_cfg.get('tuning_trials', 30)
        )
        # 用调优后的参数覆盖默认模型
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        if 'XGBoost' in best_params:
            models['XGBoost'] = XGBClassifier(
                **best_params['XGBoost'],
                random_state=random_state, eval_metric='logloss'
            )
        if 'LightGBM' in best_params:
            models['LightGBM'] = LGBMClassifier(
                **best_params['LightGBM'],
                random_state=random_state, verbose=-1
            )

    # 6. 模型训练
    logger.info("\n" + "=" * 60)
    logger.info("步骤 6: 模型训练")
    logger.info("=" * 60)
    results = train_models(
        X_train_processed, X_test_processed,
        y_train, y_test,
        models=models, random_state=random_state,
        cv_n_splits=train_cfg.get('cv_n_splits', 5)
    )

    # 7. 模型评估
    logger.info("\n" + "=" * 60)
    logger.info("步骤 7: 模型评估")
    logger.info("=" * 60)
    evaluate_models(results, feature_columns, output_dir=paths['output_dir'])

    # 8. SHAP模型解释（可选）
    logger.info("\n" + "=" * 60)
    logger.info("步骤 8: SHAP模型解释")
    logger.info("=" * 60)
    try:
        from src.shap_analysis import explain_model
        for name in ['XGBoost', 'LightGBM']:
            if name in results:
                explain_model(
                    results[name]['model'],
                    X_train_processed, X_test_processed,
                    feature_columns, name,
                    output_dir=paths['output_dir']
                )
    except ImportError:
        logger.warning("SHAP未安装，跳过模型解释步骤")
    except Exception as e:
        logger.error(f"SHAP分析失败: {e}")

    # 9. 保存模型
    logger.info("\n" + "=" * 60)
    logger.info("步骤 9: 保存模型")
    logger.info("=" * 60)
    save_models(results, output_dir=paths['model_dir'])

    logger.info("\n" + "=" * 60)
    logger.info("全部完成!")
    logger.info("=" * 60)
    logger.info(f"输出图表: {paths['output_dir']}/")
    logger.info(f"训练模型: {paths['model_dir']}/")
    logger.info(f"日志文件: {paths['log_dir']}/")


if __name__ == '__main__':
    main()
