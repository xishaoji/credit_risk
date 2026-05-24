"""配置管理模块"""
import os
import yaml


_CONFIG = None
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')


def load_config(config_path=None):
    """加载配置文件"""
    global _CONFIG
    path = config_path or _CONFIG_PATH
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            _CONFIG = yaml.safe_load(f)
    else:
        _CONFIG = _default_config()
    return _CONFIG


def get_config():
    """获取当前配置，未加载则自动加载"""
    if _CONFIG is None:
        return load_config()
    return _CONFIG


def _default_config():
    """默认配置"""
    return {
        'paths': {
            'data_dir': 'data',
            'output_dir': 'outputs',
            'model_dir': 'models',
            'log_dir': 'logs'
        },
        'data': {
            'kaggle_filename': 'cs-training.csv',
            'synthetic_samples': 5000,
            'random_state': 42
        },
        'training': {
            'test_size': 0.2,
            'cv_n_splits': 5,
            'smote_random_state': 42,
            'use_hyperparameter_tuning': True,
            'tuning_trials': 30
        },
        'models': {
            'logistic_regression': {'max_iter': 1000},
            'random_forest': {'n_estimators': 200, 'max_depth': 10},
            'xgboost': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'eval_metric': 'logloss'},
            'lightgbm': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'verbose': -1}
        }
    }
