"""单元测试 - 模型训练模块"""
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.model_training import train_models, get_default_models
from src.data_generator import generate_credit_data
from src.feature_engineering import (
    feature_engineering_fit_transform, feature_engineering_transform, prepare_data
)


class TestModelTraining:
    """测试模型训练"""

    @pytest.fixture
    def processed_data(self):
        """创建预处理后的测试数据"""
        df = generate_credit_data(n_samples=300)
        X, y = prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_proc, scaler, encoders, feature_cols = \
            feature_engineering_fit_transform(X_train, target='是否违约')
        X_test_proc = feature_engineering_transform(
            X_test, scaler, encoders, feature_cols, target='是否违约'
        )
        return X_train_proc, X_test_proc, y_train, y_test

    def test_train_models_returns(self, processed_data):
        """测试训练返回值"""
        X_train, X_test, y_train, y_test = processed_data
        results = train_models(X_train, X_test, y_train, y_test)
        assert isinstance(results, dict)
        assert len(results) == 4

    def test_all_models_trained(self, processed_data):
        """测试所有模型都已训练"""
        X_train, X_test, y_train, y_test = processed_data
        results = train_models(X_train, X_test, y_train, y_test)
        expected_models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']
        for name in expected_models:
            assert name in results

    def test_predictions_exist(self, processed_data):
        """测试预测结果存在"""
        X_train, X_test, y_train, y_test = processed_data
        results = train_models(X_train, X_test, y_train, y_test)
        for name, res in results.items():
            assert 'y_pred' in res
            assert 'y_prob' in res
            assert len(res['y_pred']) == len(res['y_test'])
            assert len(res['y_prob']) == len(res['y_test'])

    def test_cv_scores_exist(self, processed_data):
        """测试交叉验证分数存在"""
        X_train, X_test, y_train, y_test = processed_data
        results = train_models(X_train, X_test, y_train, y_test)
        for name, res in results.items():
            assert 'cv_auc_mean' in res
            assert 'cv_auc_std' in res
            assert 0 <= res['cv_auc_mean'] <= 1
            assert res['cv_auc_std'] >= 0

    def test_custom_models(self, processed_data):
        """测试自定义模型"""
        X_train, X_test, y_train, y_test = processed_data
        from sklearn.linear_model import LogisticRegression
        custom_models = {
            'Logistic Regression': LogisticRegression(max_iter=500, random_state=42)
        }
        results = train_models(X_train, X_test, y_train, y_test, models=custom_models)
        assert len(results) == 1
        assert 'Logistic Regression' in results

    def test_random_state_reproducibility(self, processed_data):
        """测试结果可重复性"""
        X_train, X_test, y_train, y_test = processed_data
        results1 = train_models(X_train, X_test, y_train, y_test, random_state=42)
        results2 = train_models(X_train, X_test, y_train, y_test, random_state=42)
        for name in results1:
            np.testing.assert_array_equal(
                results1[name]['y_pred'],
                results2[name]['y_pred']
            )

    def test_get_default_models(self):
        """测试默认模型获取"""
        models = get_default_models()
        assert len(models) == 4
        assert 'Logistic Regression' in models
        assert 'XGBoost' in models
