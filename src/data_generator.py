import numpy as np
import pandas as pd


def generate_credit_data(n_samples=5000, random_state=42):
    """生成合成银行信用风险数据集"""
    np.random.seed(random_state)

    # 客户基本信息
    age = np.random.randint(22, 65, n_samples)
    income = np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples).astype(int)
    work_years = np.random.randint(0, 40, n_samples)
    education = np.random.choice(
        ['高中', '大专', '本科', '硕士', '博士'], n_samples,
        p=[0.15, 0.25, 0.35, 0.18, 0.07]
    )
    marital_status = np.random.choice(
        ['单身', '已婚', '离异'], n_samples,
        p=[0.35, 0.50, 0.15]
    )

    # 贷款信息
    loan_amount = np.random.lognormal(mean=11, sigma=0.8, size=n_samples).astype(int)
    loan_term = np.random.choice([12, 24, 36, 48, 60], n_samples)
    interest_rate = np.random.uniform(3.5, 15.0, n_samples).round(2)

    # 信用特征
    credit_score = np.random.normal(650, 100, n_samples).clip(300, 850).astype(int)
    past_defaults = np.random.poisson(0.3, n_samples)
    num_credit_cards = np.random.poisson(2, n_samples)
    has_house = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])

    # 生成违约概率（逻辑函数）
    logit = (
        -2.5
        + 0.02 * (age - 40)
        - 0.00001 * income
        - 0.03 * work_years
        + 0.00005 * loan_amount
        + 0.05 * loan_term
        + 0.1 * interest_rate
        - 0.005 * credit_score
        + 0.8 * past_defaults
        - 0.05 * num_credit_cards
        - 0.3 * has_house
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = (np.random.random(n_samples) < prob_default).astype(int)

    df = pd.DataFrame({
        '年龄': age,
        '收入': income,
        '工作年限': work_years,
        '学历': education,
        '婚姻状况': marital_status,
        '贷款金额': loan_amount,
        '贷款期限_月': loan_term,
        '利率': interest_rate,
        '信用评分': credit_score,
        '历史违约次数': past_defaults,
        '信用卡数量': num_credit_cards,
        '是否有房产': has_house,
        '是否违约': default
    })

    return df


if __name__ == '__main__':
    df = generate_credit_data()
    df.to_csv('data/credit_data.csv', index=False, encoding='utf-8-sig')
    print(f"数据已生成: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"违约率: {df['是否违约'].mean():.2%}")
    print(df.head())
