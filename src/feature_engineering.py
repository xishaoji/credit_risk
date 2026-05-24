import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder


def _handle_missing_values(df):
    """缺失值处理：数值列用中位数填充"""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '')
    return df


def _encode_categoricals(df, encoders=None, fit=True):
    """类别变量编码，fit=True时训练encoder，否则复用已有encoder"""
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns
    if encoders is None:
        encoders = {}
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            # 处理未见过的类别
            df[col] = df[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    return df, encoders


def _construct_features(df):
    """特征构造（根据可用列动态选择）"""
    df = df.copy()
    cols = set(df.columns)
    if '月收入' in cols and '贷款金额' in cols:
        df['负债收入比'] = df['贷款金额'] / (df['月收入'] + 1)
    if '月收入' in cols and '负债比率' in cols:
        df['月负债比'] = df['负债比率'] / (df['月收入'] + 1)
    if '信用额度使用率' in cols and '负债比率' in cols:
        df['信用风险综合'] = df['信用额度使用率'] * df['负债比率']
    if '逾期30-59天次数' in cols and '逾期90天以上次数' in cols:
        df['总逾期次数'] = df['逾期30-59天次数'] + df['逾期90天以上次数']
    return df


def _scale_numerics(df, scaler=None, fit=True, target='是否违约'):
    """数值特征标准化"""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler


def feature_engineering_fit_transform(X_train, y_train=None, target='是否违约'):
    """在训练集上拟合并转换特征工程全流程

    Args:
        X_train: 训练特征
        y_train: 训练目标（仅用于特征构造时的信息，不参与fit）
        target: 目标列名

    Returns:
        X_train_processed: 处理后的训练特征
        scaler: 拟合好的StandardScaler
        encoders: 拟合好的LabelEncoder字典
        feature_columns: 最终特征列名列表
    """
    df = X_train.copy()

    df = _handle_missing_values(df)
    df, encoders = _encode_categoricals(df, encoders=None, fit=True)
    df = _construct_features(df)
    df, scaler = _scale_numerics(df, scaler=None, fit=True, target=target)

    feature_columns = df.columns.tolist()
    return df, scaler, encoders, feature_columns


def feature_engineering_transform(X, scaler, encoders, feature_columns, target='是否违约'):
    """使用训练时的scaler和encoder转换数据（用于测试集或新数据）

    Args:
        X: 待转换的特征
        scaler: 训练时拟合的StandardScaler
        encoders: 训练时拟合的LabelEncoder字典
        feature_columns: 训练时的最终特征列名列表
        target: 目标列名

    Returns:
        X_processed: 转换后的特征DataFrame
    """
    df = X.copy()

    df = _handle_missing_values(df)
    df, _ = _encode_categoricals(df, encoders=encoders, fit=False)
    df = _construct_features(df)
    df, _ = _scale_numerics(df, scaler=scaler, fit=False, target=target)

    # 确保列顺序与训练时一致
    df = df[feature_columns]
    return df


def save_preprocessor(scaler, encoders, feature_columns, output_dir='models'):
    """保存预处理器到磁盘"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    preprocessor = {
        'scaler': scaler,
        'encoders': encoders,
        'feature_columns': feature_columns
    }
    path = os.path.join(output_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, path)
    return path


def load_preprocessor(model_dir='models'):
    """从磁盘加载预处理器"""
    import os
    path = os.path.join(model_dir, 'preprocessor.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"预处理器文件不存在: {path}")
    return joblib.load(path)


def prepare_data(df, target='是否违约'):
    """分离特征和目标变量"""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
