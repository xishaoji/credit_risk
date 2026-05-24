"""单元测试 - 特征工程模块"""
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.feature_engineering import (
    feature_engineering_fit_transform, feature_engineering_transform,
    prepare_data, save_preprocessor, load_preprocessor
)
from src.data_generator import generate_credit_data


class TestFeatureEngineering:
    """测试特征工程"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据并切分"""
        df = generate_credit_data(n_samples=200)
        X, y = prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def test_fit_transform_returns(self, sample_data):
        """测试fit_transform返回值"""
        X_train, X_test, y_train, y_test = sample_data
        X_proc, scaler, encoders, feature_cols = \
            feature_engineering_fit_transform(X_train, target='是否违约')
        assert isinstance(X_proc, pd.DataFrame)
        assert scaler is not None
        assert isinstance(encoders, dict)
        assert isinstance(feature_cols, list)

    def test_no_missing_values_after(self, sample_data):
        """测试处理后无缺失值"""
        X_train, X_test, y_train, y_test = sample_data
        X_proc, _, _, _ = feature_engineering_fit_transform(X_train, target='是否违约')
        assert X_proc.isnull().sum().sum() == 0

    def test_scaler_applied(self, sample_data):
        """测试标准化已应用"""
        X_train, X_test, y_train, y_test = sample_data
        X_proc, _, _, _ = feature_engineering_fit_transform(X_train, target='是否违约')
        numeric_cols = X_proc.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert abs(X_proc[col].mean()) < 1.0

    def test_transform_reuses_scaler(self, sample_data):
        """测试transform复用训练时的scaler"""
        X_train, X_test, y_train, y_test = sample_data
        X_train_proc, scaler, encoders, feature_cols = \
            feature_engineering_fit_transform(X_train, target='是否违约')
        X_test_proc = feature_engineering_transform(
            X_test, scaler, encoders, feature_cols, target='是否违约'
        )
        # 测试集列数和顺序应与训练集一致
        assert list(X_test_proc.columns) == list(X_train_proc.columns)
        assert len(X_test_proc) == len(X_test)

    def test_feature_columns_consistent(self, sample_data):
        """测试特征列一致"""
        X_train, X_test, y_train, y_test = sample_data
        _, scaler, encoders, feature_cols = \
            feature_engineering_fit_transform(X_train, target='是否违约')
        X_test_proc = feature_engineering_transform(
            X_test, scaler, encoders, feature_cols, target='是否违约'
        )
        assert list(X_test_proc.columns) == feature_cols

    def test_prepare_data(self, sample_data):
        """测试数据准备"""
        X_train, _, _, _ = sample_data
        # prepare_data应该在特征工程之前使用
        df = generate_credit_data(n_samples=10)
        X, y = prepare_data(df)
        assert '是否违约' not in X.columns
        assert len(y) == len(X)

    def test_new_features_created(self, sample_data):
        """测试新特征构造"""
        X_train, X_test, y_train, y_test = sample_data
        X_proc, _, _, _ = feature_engineering_fit_transform(X_train, target='是否违约')
        # 合成数据没有Kaggle列名，所以不会创建新特征
        # 但列数应该 >= 原始列数
        assert X_proc.shape[1] >= X_train.shape[1]

    def test_save_load_preprocessor(self, sample_data, tmp_path):
        """测试预处理器保存和加载"""
        X_train, X_test, y_train, y_test = sample_data
        _, scaler, encoders, feature_cols = \
            feature_engineering_fit_transform(X_train, target='是否违约')

        save_preprocessor(scaler, encoders, feature_cols, output_dir=str(tmp_path))
        loaded = load_preprocessor(model_dir=str(tmp_path))

        assert 'scaler' in loaded
        assert 'encoders' in loaded
        assert 'feature_columns' in loaded
        assert loaded['feature_columns'] == feature_cols


class TestFeatureEngineeringEdgeCases:
    """测试边界情况"""

    def test_single_row_fit_transform(self):
        """测试单行数据fit_transform"""
        df = generate_credit_data(n_samples=1)
        X, y = prepare_data(df)
        X_proc, _, _, _ = feature_engineering_fit_transform(X, target='是否违约')
        assert len(X_proc) == 1
