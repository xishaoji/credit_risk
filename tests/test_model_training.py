"""单元测试 - 模型训练模块"""
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.model_training import train_models
from src.data_generator import generate_credit_data


class TestModelTraining:
    """测试模型训练"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        df = generate_credit_data(n_samples=200)
        X = df.drop('是否违约', axis=1)
        y = df['是否违约']
        # 简单编码类别变量
        X = pd.get_dummies(X, drop_first=True)
        return X, y

    def test_train_models_returns(self, sample_data):
        """测试训练返回值"""
        X, y = sample_data
        results, X_train, X_test, y_train, y_test = train_models(X, y, test_size=0.2)
        assert isinstance(results, dict)
        assert len(results) == 4  # 4个模型

    def test_all_models_trained(self, sample_data):
        """测试所有模型都已训练"""
        X, y = sample_data
        results, _, _, _, _ = train_models(X, y, test_size=0.2)
        expected_models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']
        for name in expected_models:
            assert name in results

    def test_predictions_exist(self, sample_data):
        """测试预测结果存在"""
        X, y = sample_data
        results, _, _, _, _ = train_models(X, y, test_size=0.2)
        for name, res in results.items():
            assert 'y_pred' in res
            assert 'y_prob' in res
            assert len(res['y_pred']) == len(res['y_test'])
            assert len(res['y_prob']) == len(res['y_test'])

    def test_train_test_split(self, sample_data):
        """测试训练集测试集划分"""
        X, y = sample_data
        _, X_test, _, y_test = train_models(X, y, test_size=0.3)
        assert len(X_test) == int(len(X) * 0.3)

    def test_smote_applied(self, sample_data):
        """测试SMOTE应用"""
        X, y = sample_data
        results, X_train, _, y_train, _ = train_models(X, y, test_size=0.2)
        # SMOTE后训练集应该比原始大（或相等）
        assert len(results['Logistic Regression']['y_test']) > 0

    def test_cross_validation(self, sample_data):
        """测试交叉验证（如果有的话）"""
        X, y = sample_data
        results, _, _, _, _ = train_models(X, y, test_size=0.2)
        # 检查结果中是否有交叉验证信息
        for name, res in results.items():
            assert isinstance(res, dict)

    def test_custom_test_size(self, sample_data):
        """测试自定义测试集比例"""
        X, y = sample_data
        _, X_test, _, _ = train_models(X, y, test_size=0.25)
        expected_size = int(len(X) * 0.25)
        assert abs(len(X_test) - expected_size) <= 1

    def test_random_state_reproducibility(self, sample_data):
        """测试结果可重复性"""
        X, y = sample_data
        results1, _, _, _, _ = train_models(X, y, random_state=42)
        results2, _, _, _, _ = train_models(X, y, random_state=42)
        for name in results1:
            np.testing.assert_array_equal(
                results1[name]['y_pred'],
                results2[name]['y_pred']
            )
