"""日志系统模块"""
import os
import logging
from datetime import datetime


def setup_logger(name='credit_risk', log_dir='logs', level=logging.INFO):
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Specific module loggers
    loggers = {
        'data_loader': logging.getLogger('src.data_loader'),
        'eda': logging.getLogger('src.eda'),
        'feature_engineering': logging.getLogger('src.feature_engineering'),
        'model_training': logging.getLogger('src.model_training'),
        'evaluation': logging.getLogger('src.evaluation'),
        'shap_analysis': logging.getLogger('src.shap_analysis'),
        'data_quality': logging.getLogger('src.data_quality'),
        'hyperparameter_tuning': logging.getLogger('src.hyperparameter_tuning'),
    }

    for logger_name, logger_obj in loggers.items():
        logger_obj.setLevel(level)

    logging.info(f"日志系统已初始化，日志文件: {log_file}")

    return loggers


if __name__ == '__main__':
    loggers = setup_logger()
    logging.info("测试日志系统")
    loggers['data_loader'].info("数据加载模块测试")
    loggers['model_training'].info("模型训练模块测试")
