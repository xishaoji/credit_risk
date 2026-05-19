"""单元测试 - 数据生成模块"""
import pytest
import pandas as pd
from src.data_generator import generate_credit_data


class TestCreditDataGenerator:
    """测试信用数据生成器"""

    def test_generate_default_samples(self):
        """测试默认生成样本数量"""
        df = generate_credit_data()
        assert df.shape[0] == 5000

    def test_generate_custom_samples(self):
        """测试自定义样本数量"""
        n_samples = 1000
        df = generate_credit_data(n_samples=n_samples)
        assert df.shape[0] == n_samples

    def test_column_names(self):
        """测试列名"""
        df = generate_credit_data()
        expected_columns = [
            '年龄', '收入', '工作年限', '学历', '婚姻状况',
            '贷款金额', '贷款期限_月', '利率', '信用评分',
            '历史违约次数', '信用卡数量', '是否有房产', '是否违约'
        ]
        assert list(df.columns) == expected_columns

    def test_data_types(self):
        """测试数据类型"""
        df = generate_credit_data()
        assert df['年龄'].dtype in ['int32', 'int64']
        assert df['收入'].dtype in ['int32', 'int64']
        assert df['是否违约'].dtype in ['int32', 'int64']

    def test_no_missing_values(self):
        """测试无缺失值"""
        df = generate_credit_data()
        assert df.isnull().sum().sum() == 0

    def test_target_distribution(self):
        """测试目标变量分布"""
        df = generate_credit_data(n_samples=10000)
        default_rate = df['是否违约'].mean()
        # 违约率应该在合理范围内
        assert 0.01 < default_rate < 0.5

    def test_random_state_reproducibility(self):
        """测试随机数可重复性"""
        df1 = generate_credit_data(random_state=42)
        df2 = generate_credit_data(random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_age_range(self):
        """测试年龄范围"""
        df = generate_credit_data()
        assert df['年龄'].min() >= 22
        assert df['年龄'].max() <= 65

    def test_credit_score_range(self):
        """测试信用评分范围"""
        df = generate_credit_data()
        assert df['信用评分'].min() >= 300
        assert df['信用评分'].max() <= 850

    def test_categorical_columns(self):
        """测试类别型列"""
        df = generate_credit_data()
        assert df['学历'].dtype == 'object'
        assert df['婚姻状况'].dtype == 'object'
