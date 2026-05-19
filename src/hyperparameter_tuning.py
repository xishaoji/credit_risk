"""超参数调优模块 - 使用Optuna进行贝叶斯优化"""
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import logging

logger = logging.getLogger(__name__)


def objective_xgboost(trial, X, y):
    """XGBoost超参数优化目标函数"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }

    model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    return scores.mean()


def objective_lightgbm(trial, X, y):
    """LightGBM超参数优化目标函数"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
    }

    model = LGBMClassifier(**params, random_state=42, verbose=-1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    return scores.mean()


def tune_xgboost(X, y, n_trials=50):
    """调优XGBoost超参数"""
    logger.info("开始XGBoost超参数调优...")

    study = optuna.create_study(direction='maximize', study_name='xgboost_tuning')
    study.optimize(lambda trial: objective_xgboost(trial, X, y), n_trials=n_trials, show_progress_bar=True)

    logger.info(f"最佳AUC: {study.best_value:.4f}")
    logger.info(f"最佳参数: {study.best_params}")

    return study.best_params


def tune_lightgbm(X, y, n_trials=50):
    """调优LightGBM超参数"""
    logger.info("开始LightGBM超参数调优...")

    study = optuna.create_study(direction='maximize', study_name='lightgbm_tuning')
    study.optimize(lambda trial: objective_lightgbm(trial, X, y), n_trials=n_trials, show_progress_bar=True)

    logger.info(f"最佳AUC: {study.best_value:.4f}")
    logger.info(f"最佳参数: {study.best_params}")

    return study.best_params


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('data/credit_data.csv')
    X = df.drop('是否违约', axis=1)
    y = df['是否违约']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("调优XGBoost...")
    xgb_params = tune_xgboost(X_train, y_train, n_trials=30)
    print(f"XGBoost最佳参数: {xgb_params}")

    print("\n调优LightGBM...")
    lgbm_params = tune_lightgbm(X_train, y_train, n_trials=30)
    print(f"LightGBM最佳参数: {lgbm_params}")
