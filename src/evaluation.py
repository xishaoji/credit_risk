import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_models(results, feature_names, output_dir='outputs'):
    """评估所有模型并生成可视化"""
    os.makedirs(output_dir, exist_ok=True)

    # 混淆矩阵
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(res['y_test'], res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['未违约', '违约'], yticklabels=['未违约', '违约'])
        axes[i].set_title(f'{name}\n混淆矩阵')
        axes[i].set_ylabel('真实值')
        axes[i].set_xlabel('预测值')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=150)
    plt.close()

    # ROC曲线
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC 曲线')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150)
    plt.close()

    # Precision-Recall曲线
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        precision, recall, _ = precision_recall_curve(res['y_test'], res['y_prob'])
        ap = average_precision_score(res['y_test'], res['y_prob'])
        plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall 曲线')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=150)
    plt.close()

    # 分类报告
    print("\n" + "=" * 60)
    print("模型评估报告")
    print("=" * 60)
    for name, res in results.items():
        print(f"\n--- {name} ---")
        print(classification_report(res['y_test'], res['y_pred'],
                                    target_names=['未违约', '违约']))

    # 特征重要性（树模型）
    tree_models = {k: v for k, v in results.items()
                   if k in ['Random Forest', 'XGBoost', 'LightGBM']}
    if tree_models:
        fig, axes = plt.subplots(1, len(tree_models), figsize=(6 * len(tree_models), 6))
        if len(tree_models) == 1:
            axes = [axes]
        for i, (name, res) in enumerate(tree_models.items()):
            importance = res['model'].feature_importances_
            indices = np.argsort(importance)[::-1][:10]
            axes[i].barh(range(len(indices)),
                         importance[indices][::-1],
                         align='center')
            axes[i].set_yticks(range(len(indices)))
            axes[i].set_yticklabels([feature_names[j] for j in indices][::-1])
            axes[i].set_title(f'{name} Top 10 特征重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
        plt.close()

    # 模型对比表
    print("\n" + "=" * 60)
    print("模型对比")
    print("=" * 60)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    comparison = []
    for name, res in results.items():
        comparison.append({
            '模型': name,
            '准确率': accuracy_score(res['y_test'], res['y_pred']),
            'F1分数': f1_score(res['y_test'], res['y_pred']),
            'AUC': roc_auc_score(res['y_test'], res['y_prob'])
        })
    import pandas as pd
    df_comp = pd.DataFrame(comparison)
    print(df_comp.to_string(index=False))

    print(f"\n评估图表已保存到 {output_dir}/ 目录")
    return df_comp
