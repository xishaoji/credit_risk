import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_give_me_some_credit(data_dir='data'):
    """
    加载 Give Me Some Credit 数据集

    下载地址: https://www.kaggle.com/c/GiveMeSomeCredit/data
    需要将 cs-training.csv 放到 data/ 目录下
    """
    file_path = os.path.join(data_dir, 'cs-training.csv')

    if not os.path.exists(file_path):
        logger.warning("=" * 60)
        logger.warning("数据集未找到！")
        logger.warning("=" * 60)
        logger.warning("请按以下步骤操作:")
        logger.warning("1. 访问 https://www.kaggle.com/c/GiveMeSomeCredit/data")
        logger.warning("2. 下载 cs-training.csv 文件")
        logger.warning(f"3. 将文件放到 {os.path.abspath(data_dir)}/ 目录下")
        logger.warning("4. 重新运行程序")
        return None

    logger.info(f"加载数据集: {file_path}")
    df = pd.read_csv(file_path, index_col=0)

    # 重命名列（中文）
    column_mapping = {
        'SeriousDlqin2yrs': '是否违约',
        'RevolvingUtilizationOfUnsecuredLines': '信用额度使用率',
        'age': '年龄',
        'NumberOfTime30-59DaysPastDueNotWorse': '逾期30-59天次数',
        'DebtRatio': '负债比率',
        'MonthlyIncome': '月收入',
        'NumberOfOpenCreditLinesAndLoans': '信用账户数',
        'NumberRealEstateLoansOrLines': '房贷数量',
        'NumberOfTimes90DaysLate': '逾期90天以上次数',
        'NumberOfTime60-89DaysPastDueNotWorse': '逾期60-89天次数',
        'NumberOfDependents': '家属人数'
    }
    df = df.rename(columns=column_mapping)

    logger.info(f"数据形状: {df.shape}")
    logger.info(f"违约率: {df['是否违约'].mean():.2%}")
    logger.info(f"缺失值:\n{df.isnull().sum()}")

    return df


def load_home_credit(data_dir='data'):
    """
    加载 Home Credit Default Risk 数据集

    下载地址: https://www.kaggle.com/c/home-credit-default-risk/data
    需要将 application_train.csv 放到 data/ 目录下
    """
    file_path = os.path.join(data_dir, 'application_train.csv')

    if not os.path.exists(file_path):
        logger.warning("=" * 60)
        logger.warning("数据集未找到！")
        logger.warning("=" * 60)
        logger.warning("请按以下步骤操作:")
        logger.warning("1. 访问 https://www.kaggle.com/c/home-credit-default-risk/data")
        logger.warning("2. 下载 application_train.csv 文件")
        logger.warning(f"3. 将文件放到 {os.path.abspath(data_dir)}/ 目录下")
        logger.warning("4. 重新运行程序")
        return None

    logger.info(f"加载数据集: {file_path}")
    df = pd.read_csv(file_path)

    # 重命名目标列
    df = df.rename(columns={'TARGET': '是否违约'})

    logger.info(f"数据形状: {df.shape}")
    logger.info(f"违约率: {df['是否违约'].mean():.2%}")

    return df


if __name__ == '__main__':
    df = load_give_me_some_credit()
    if df is not None:
        print("\n前5行数据:")
        print(df.head())
        print("\n数据类型:")
        print(df.dtypes)
