"""SHAP模型解释性分析模块"""
import os
import shap
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def explain_model(model, X_train, X_test, feature_names, model_name, output_dir='outputs'):
    """使用SHAP解释模型预测"""
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"生成SHAP解释: {model_name}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 1. SHAP Summary Plot (Feature Importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f'{model_name} - SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_shap_summary.png'), dpi=150)
    plt.close()

    # 2. SHAP Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f'{model_name} - SHAP Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_shap_bar.png'), dpi=150)
    plt.close()

    # 3. SHAP Dependence Plots for top features
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[::-1][:3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, feat_idx in enumerate(top_indices):
        plt.sca(axes[idx])
        shap.dependence_plot(feat_idx, shap_values, X_test, feature_names=feature_names, show=False)
        axes[idx].set_title(f'{feature_names[feat_idx]} dependence')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_shap_dependence.png'), dpi=150)
    plt.close()

    # 4. SHAP Waterfall for first prediction
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test.iloc[0],
            feature_names=feature_names
        ),
        show=False,
        max_display=10
    )
    plt.title(f'{model_name} - SHAP Waterfall Plot (Sample 0)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_shap_waterfall.png'), dpi=150)
    plt.close()

    logger.info(f"SHAP图表已保存到 {output_dir}/")

    return shap_values


def compare_shap_across_models(models_dict, X_train, X_test, feature_names, output_dir='outputs'):
    """对比多个模型的SHAP特征重要性"""
    os.makedirs(output_dir, exist_ok=True)

    importances = {}
    for name, model in models_dict.items():
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        importances[name] = np.abs(shap_values).mean(axis=0)

    # 归一化对比
    fig, axes = plt.subplots(1, len(models_dict), figsize=(6 * len(models_dict), 6))
    if len(models_dict) == 1:
        axes = [axes]

    for idx, (name, imp) in enumerate(importances.items()):
        sorted_idx = np.argsort(imp)[::-1][:10]
        axes[idx].barh(range(len(sorted_idx)), imp[sorted_idx][::-1])
        axes[idx].set_yticks(range(len(sorted_idx)))
        axes[idx].set_yticklabels([feature_names[i] for i in sorted_idx][::-1])
        axes[idx].set_title(f'{name}\nTop 10 Features')

    plt.suptitle('Cross-Model SHAP Feature Importance Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_model_shap_comparison.png'), dpi=150)
    plt.close()

    logger.info("模型间SHAP对比图已保存")


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    df = pd.read_csv('data/credit_data.csv')
    X = df.drop('是否违约', axis=1)
    y = df['是否违约']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    xgb = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    lgbm = LGBMClassifier(n_estimators=200, max_depth=6, random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)

    feature_names = list(X.columns)

    explain_model(xgb, X_train, X_test, feature_names, 'XGBoost')
    explain_model(lgbm, X_train, X_test, feature_names, 'LightGBM')

    compare_shap_across_models(
        {'XGBoost': xgb, 'LightGBM': lgbm},
        X_train, X_test, feature_names
    )
