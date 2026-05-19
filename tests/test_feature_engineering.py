"""单元测试 - 特征工程模块"""
import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import feature_engineering, prepare_data
from src.data_generator import generate_credit_data


class TestFeatureEngineering:
    """测试特征工程"""

    @pytest.fixture
    def sample_df(self):
        """创建测试数据"""
        return generate_credit_data(n_samples=100)

    def test_feature_engineering_returns(self, sample_df):
        """测试特征工程返回值"""
        result, scaler, encoders = feature_engineering(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert scaler is not None
        assert isinstance(encoders, dict)

    def test_no_missing_values_after(self, sample_df):
        """测试处理后无缺失值"""
        result, _, _ = feature_engineering(sample_df)
        assert result.isnull().sum().sum() == 0

    def test_target_column_preserved(self, sample_df):
        """测试目标列保留"""
        result, _, _ = feature_engineering(sample_df)
        assert '是否违约' in result.columns

    def test_scaler_applied(self, sample_df):
        """测试标准化已应用"""
        result, _, _ = feature_engineering(sample_df)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != '是否违约']
        # 标准化后均值应该接近0
        for col in numeric_cols:
            assert abs(result[col].mean()) < 0.5

    def test_label_encoding(self, sample_df):
        """测试标签编码"""
        _, _, encoders = feature_engineering(sample_df)
        # 编码器应该包含类别列
        assert len(encoders) > 0

    def test_prepare_data(self, sample_df):
        """测试数据准备"""
        result, _, _ = feature_engineering(sample_df)
        X, y = prepare_data(result)
        assert '是否违约' not in X.columns
        assert len(y) == len(X)

    def test_new_features_created(self, sample_df):
        """测试新特征创建"""
        result, _, _ = feature_engineering(sample_df)
        # 应该有一些新的组合特征
        assert result.shape[1] >= sample_df.shape[1]


class TestFeatureEngineeringEdgeCases:
    """测试边界情况"""

    def test_empty_dataframe(self):
        """测试空DataFrame"""
        df = pd.DataFrame()
        # 空DataFrame应该抛出异常或返回空结果
        with pytest.raises((ValueError, KeyError)):
            feature_engineering(df)

    def test_single_row(self):
        """测试单行数据"""
        df = generate_credit_data(n_samples=1)
        result, _, _ = feature_engineering(df)
        assert len(result) == 1
