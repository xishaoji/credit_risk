import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_eda(df, output_dir='outputs'):
    """执行探索性数据分析并保存图表"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 数据概览
    print("=" * 60)
    print("数据概览")
    print("=" * 60)
    print(f"数据形状: {df.shape}")
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n缺失值统计:\n{df.isnull().sum()}")
    print(f"\n基本统计:\n{df.describe()}")

    # 2. 目标变量分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    default_counts = df['是否违约'].value_counts()
    axes[0].bar(['未违约', '违约'], default_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('违约分布')
    axes[0].set_ylabel('数量')
    for i, v in enumerate(default_counts.values):
        axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')
    axes[1].pie(default_counts.values, labels=['未违约', '违约'],
                autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
    axes[1].set_title('违约比例')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=150)
    plt.close()

    # 3. 数值特征分布
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove('是否违约')
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('数值特征分布', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'numeric_distributions.png'), dpi=150)
    plt.close()

    # 4. 类别特征分布
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        fig, axes = plt.subplots(1, len(cat_cols), figsize=(6 * len(cat_cols), 5))
        if len(cat_cols) == 1:
            axes = [axes]
        for i, col in enumerate(cat_cols):
            df[col].value_counts().plot(kind='bar', ax=axes[i], edgecolor='black')
            axes[i].set_title(f'{col} 分布')
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'categorical_distributions.png'), dpi=150)
        plt.close()

    # 5. 相关性热力图
    plt.figure(figsize=(12, 10))
    corr = df[num_cols + ['是否违约']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5)
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()

    # 6. 特征与违约的关系
    num_cols_for_target = [c for c in num_cols if c != '是否违约'][:6]
    n_feats = len(num_cols_for_target)
    rows = (n_feats + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten() if n_feats > 1 else [axes]
    for i, feat in enumerate(num_cols_for_target):
        ax = axes[i]
        df.groupby('是否违约')[feat].plot(kind='hist', alpha=0.6, ax=ax, legend=True)
        ax.set_title(f'{feat} vs 违约')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_vs_target.png'), dpi=150)
    plt.close()

    print(f"\nEDA图表已保存到 {output_dir}/ 目录")
