import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def feature_engineering(df):
    """特征工程：缺失值处理、编码、标准化、特征构造"""
    df = df.copy()

    # 缺失值处理
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())

    # 类别变量编码
    label_encoders = {}
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 特征构造（根据可用列动态选择）
    cols = set(df.columns)
    if '月收入' in cols and '贷款金额' in cols:
        df['负债收入比'] = df['贷款金额'] / (df['月收入'] + 1)
    if '月收入' in cols and '负债比率' in cols:
        df['月负债比'] = df['负债比率'] / (df['月收入'] + 1)
    if '信用额度使用率' in cols and '负债比率' in cols:
        df['信用风险综合'] = df['信用额度使用率'] * df['负债比率']
    if '逾期30-59天次数' in cols and '逾期90天以上次数' in cols:
        df['总逾期次数'] = df['逾期30-59天次数'] + df['逾期90天以上次数']

    # 数值特征标准化
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove('是否违约')
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler, label_encoders


def prepare_data(df, target='是否违约'):
    """分离特征和目标变量"""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
