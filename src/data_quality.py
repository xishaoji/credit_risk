"""数据质量检查模块"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def check_missing_values(df):
    """检查缺失值"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        '缺失数量': missing,
        '缺失比例(%)': missing_pct
    })
    missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失比例(%)', ascending=False)

    if len(missing_df) > 0:
        logger.warning(f"发现 {len(missing_df)} 列存在缺失值")
    else:
        logger.info("无缺失值")

    return missing_df


def detect_outliers(df, method='iqr', threshold=1.5):
    """检测异常值"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_info = {}

    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df[col][z_scores > threshold]

        if len(outliers) > 0:
            outliers_info[col] = {
                '数量': len(outliers),
                '比例(%)': round(len(outliers) / len(df) * 100, 2),
                '最小值': df[col].min(),
                '最大值': df[col].max(),
                '均值': round(df[col].mean(), 2),
                '标准差': round(df[col].std(), 2)
            }

    return outliers_info


def check_duplicates(df):
    """检查重复行"""
    n_duplicates = df.duplicated().sum()
    return n_duplicates


def check_data_types(df):
    """检查数据类型"""
    type_info = pd.DataFrame({
        '数据类型': df.dtypes,
        '唯一值数量': df.nunique(),
        '示例值': [df[col].iloc[0] if len(df) > 0 else None for col in df.columns]
    })
    return type_info


def check_class_balance(y, threshold=0.3):
    """检查类别平衡"""
    class_counts = y.value_counts()
    class_pct = y.value_counts(normalize=True) * 100

    imbalance_ratio = class_counts.max() / class_counts.min()
    is_imbalanced = imbalance_ratio > (1 / threshold)

    return {
        '类别分布': class_counts.to_dict(),
        '类别比例(%)': class_pct.round(2).to_dict(),
        '不平衡比例': round(imbalance_ratio, 2),
        '是否不平衡': is_imbalanced
    }


def plot_outliers(df, output_dir='outputs'):
    """可视化异常值"""
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.boxplot(df[col].dropna())
        ax.set_title(f'{col} - Boxplot')
        ax.set_ylabel('Value')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Features Outlier Detection', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outlier_detection.png'), dpi=150)
    plt.close()

    logger.info("异常值检测图已保存")


def generate_quality_report(df, target_col='是否违约', output_dir='outputs'):
    """生成完整的数据质量报告"""
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("数据质量报告")
    logger.info("=" * 60)

    # 基本信息
    logger.info(f"\n数据形状: {df.shape}")
    logger.info(f"内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # 缺失值
    missing_df = check_missing_values(df)
    if len(missing_df) > 0:
        logger.warning(f"\n缺失值统计:\n{missing_df}")
    else:
        logger.info("\n无缺失值")

    # 重复值
    n_duplicates = check_duplicates(df)
    if n_duplicates > 0:
        logger.warning(f"\n重复行数量: {n_duplicates} ({n_duplicates/len(df)*100:.2f}%)")
    else:
        logger.info("\n无重复行")

    # 异常值
    outliers_info = detect_outliers(df.drop(columns=[target_col], errors='ignore'))
    if outliers_info:
        logger.warning(f"\n异常值检测 (IQR方法):")
        for col, info in outliers_info.items():
            logger.warning(f"  {col}: {info['数量']}个异常值 ({info['比例(%)']}%)")

    # 类别平衡
    if target_col in df.columns:
        balance_info = check_class_balance(df[target_col])
        logger.info(f"\n类别平衡检查:")
        logger.info(f"  类别分布: {balance_info['类别分布']}")
        logger.info(f"  不平衡比例: {balance_info['不平衡比例']}")
        logger.info(f"  是否不平衡: {'是' if balance_info['是否不平衡'] else '否'}")

    # 生成可视化
    plot_outliers(df.drop(columns=[target_col], errors='ignore'), output_dir)

    logger.info(f"\n数据质量报告已保存到 {output_dir}/")

    return {
        'missing': missing_df,
        'duplicates': n_duplicates,
        'outliers': outliers_info,
        'balance': balance_info if target_col in df.columns else None
    }


if __name__ == '__main__':
    df = pd.read_csv('data/credit_data.csv')
    report = generate_quality_report(df)
