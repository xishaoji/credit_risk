import os
import logging
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def get_default_models(random_state=42):
    """返回默认超参数的模型字典"""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=random_state, eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=random_state, verbose=-1
        )
    }


def train_models(X_train, X_test, y_train, y_test, models=None,
                 random_state=42, cv_n_splits=5):
    """训练多个模型，执行交叉验证并评估

    Args:
        X_train: 训练特征
        X_test: 测试特征
        y_train: 训练目标
        y_test: 测试目标
        models: 模型字典，None则使用默认模型
        random_state: 随机种子
        cv_n_splits: 交叉验证折数

    Returns:
        results: 模型结果字典
    """
    if models is None:
        models = get_default_models(random_state)

    # SMOTE处理类别不平衡
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(f"SMOTE前: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logger.info(f"SMOTE后: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

    cv = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)

    results = {}
    for name, model in models.items():
        logger.info(f"训练 {name}...")

        # 交叉验证
        cv_scores = cross_val_score(model, X_train_res, y_train_res,
                                    cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_auc_mean = cv_scores.mean()
        cv_auc_std = cv_scores.std()
        logger.info(f"  CV AUC: {cv_auc_mean:.4f} (+/- {cv_auc_std:.4f})")

        # 在完整训练集上训练
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'y_test': y_test,
            'cv_auc_mean': cv_auc_mean,
            'cv_auc_std': cv_auc_std
        }
        logger.info(f"  训练完成")

    return results


def save_models(results, output_dir='models'):
    """保存训练好的模型"""
    os.makedirs(output_dir, exist_ok=True)
    for name, res in results.items():
        path = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(res['model'], path)
    logger.info(f"模型已保存到 {output_dir}/ 目录")
