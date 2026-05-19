import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE


def train_models(X, y, test_size=0.2, random_state=42):
    """训练多个模型并返回结果"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # SMOTE处理类别不平衡
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE前: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"SMOTE后: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=random_state, use_label_encoder=False, eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=random_state, verbose=-1
        )
    }

    results = {}
    for name, model in models.items():
        print(f"\n训练 {name}...")
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'y_test': y_test
        }
        print(f"  训练完成")

    return results, X_train, X_test, y_train, y_test


def save_models(results, output_dir='models'):
    """保存训练好的模型"""
    os.makedirs(output_dir, exist_ok=True)
    for name, res in results.items():
        path = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(res['model'], f)
    print(f"\n模型已保存到 {output_dir}/ 目录")
