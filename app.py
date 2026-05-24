"""FastAPI模型部署服务"""
import os
import logging
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Prediction API",
    description="银行信用风险评估预测API",
    version="2.0.0"
)

MODEL_DIR = "models"
DEFAULT_MODEL = "xgboost"


class CreditFeatures(BaseModel):
    """Kaggle Give Me Some Credit 数据集特征"""
    信用额度使用率: float = Field(..., description="信用额度使用率 (0~1)")
    年龄: int = Field(..., ge=18, le=100, description="年龄")
    逾期30_59天次数: int = Field(..., alias="逾期30-59天次数", ge=0, description="逾期30-59天次数")
    负债比率: float = Field(..., ge=0, description="负债比率")
    月收入: float = Field(..., ge=0, description="月收入")
    信用账户数: int = Field(..., ge=0, description="信用账户数")
    房贷数量: int = Field(..., ge=0, description="房地产贷款数量")
    逾期90天以上次数: int = Field(..., ge=0, description="逾期90天以上次数")
    逾期60_89天次数: int = Field(..., alias="逾期60-89天次数", ge=0, description="逾期60-89天次数")
    家属人数: int = Field(..., ge=0, description="家属人数")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    """预测响应"""
    model_name: str
    default_probability: float
    is_default: bool
    risk_level: str
    features_used: List[str]


class BatchPredictionRequest(BaseModel):
    """批量预测请求"""
    instances: List[CreditFeatures]


class BatchPredictionResponse(BaseModel):
    """批量预测响应"""
    predictions: List[PredictionResponse]
    total_count: int
    default_count: int
    default_rate: float


# 缓存
loaded_models = {}
loaded_preprocessor = None


def load_model(model_name: str):
    """加载指定模型"""
    if model_name not in loaded_models:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
        loaded_models[model_name] = joblib.load(model_path)
        logger.info(f"已加载模型: {model_name}")
    return loaded_models[model_name]


def load_preprocessor():
    """加载预处理器（scaler + encoder）"""
    global loaded_preprocessor
    if loaded_preprocessor is None:
        from src.feature_engineering import load_preprocessor as _load
        loaded_preprocessor = _load(MODEL_DIR)
        logger.info("已加载预处理器")
    return loaded_preprocessor


def preprocess_features(features: CreditFeatures, preprocessor) -> pd.DataFrame:
    """使用训练时的预处理器转换特征"""
    data = features.model_dump(by_alias=True)
    df = pd.DataFrame([data])

    # 使用训练时的encoder处理类别变量
    encoders = preprocessor['encoders']
    for col, le in encoders.items():
        if col in df.columns:
            # 处理未见过的类别
            df[col] = df[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    # 使用训练时的scaler标准化数值特征
    scaler = preprocessor['scaler']
    feature_columns = preprocessor['feature_columns']

    # 特征构造（与训练时一致）
    cols = set(df.columns)
    if '月收入' in cols and '负债比率' in cols:
        df['月负债比'] = df['负债比率'] / (df['月收入'] + 1)
    if '信用额度使用率' in cols and '负债比率' in cols:
        df['信用风险综合'] = df['信用额度使用率'] * df['负债比率']
    if '逾期30-59天次数' in cols and '逾期90天以上次数' in cols:
        df['总逾期次数'] = df['逾期30-59天次数'] + df['逾期90天以上次数']

    # 标准化
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = scaler.transform(df[num_cols])

    # 确保列顺序与训练时一致
    df = df[feature_columns]
    return df


def get_risk_level(probability: float) -> str:
    """根据概率判断风险等级"""
    if probability < 0.1:
        return "低风险"
    elif probability < 0.3:
        return "中等风险"
    elif probability < 0.5:
        return "较高风险"
    else:
        return "高风险"


@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "Credit Risk Prediction API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/models")
async def list_models():
    """列出可用模型"""
    available_models = []
    if os.path.exists(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith('.pkl') and f != 'preprocessor.pkl':
                available_models.append(f.replace('.pkl', ''))
    return {"models": available_models}


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CreditFeatures, model_name: str = DEFAULT_MODEL):
    """单个预测"""
    try:
        model = load_model(model_name)
        preprocessor = load_preprocessor()
        df = preprocess_features(features, preprocessor)

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return PredictionResponse(
            model_name=model_name,
            default_probability=round(float(probability), 4),
            is_default=bool(prediction),
            risk_level=get_risk_level(probability),
            features_used=list(df.columns)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, model_name: str = DEFAULT_MODEL):
    """批量预测"""
    try:
        model = load_model(model_name)
        preprocessor = load_preprocessor()

        # 批量构建DataFrame后一次性预测
        all_data = [inst.model_dump() for inst in request.instances]
        df = pd.DataFrame(all_data)

        # 逐样本预处理（因为有特征构造逻辑）
        dfs = []
        for features in request.instances:
            row_df = preprocess_features(features, preprocessor)
            dfs.append(row_df)
        df_batch = pd.concat(dfs, ignore_index=True)

        predictions_raw = model.predict(df_batch)
        probabilities_raw = model.predict_proba(df_batch)[:, 1]

        predictions = []
        for pred, prob in zip(predictions_raw, probabilities_raw):
            predictions.append(PredictionResponse(
                model_name=model_name,
                default_probability=round(float(prob), 4),
                is_default=bool(pred),
                risk_level=get_risk_level(prob),
                features_used=list(df_batch.columns)
            ))

        default_count = sum(1 for p in predictions if p.is_default)

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            default_count=default_count,
            default_rate=round(default_count / len(predictions), 4) if predictions else 0
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
