"""FastAPI模型部署服务"""
import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Prediction API",
    description="银行信用风险评估预测API",
    version="1.0.0"
)

# 模型路径
MODEL_DIR = "models"
DEFAULT_MODEL = "xgboost"


class CreditFeatures(BaseModel):
    """信用风险特征"""
    年龄: int = Field(..., ge=18, le=100, description="年龄")
    收入: int = Field(..., ge=0, description="月收入")
    工作年限: int = Field(..., ge=0, le=50, description="工作年限")
    学历: str = Field(..., description="学历: 高中/大专/本科/硕士/博士")
    婚姻状况: str = Field(..., description="婚姻状况: 单身/已婚/离异")
    贷款金额: int = Field(..., ge=0, description="贷款金额")
    贷款期限_月: int = Field(..., ge=1, le=60, description="贷款期限(月)")
    利率: float = Field(..., ge=0, le=100, description="利率(%)")
    信用评分: int = Field(..., ge=300, le=850, description="信用评分")
    历史违约次数: int = Field(..., ge=0, description="历史违约次数")
    信用卡数量: int = Field(..., ge=0, description="信用卡数量")
    是否有房产: int = Field(..., ge=0, le=1, description="是否有房产(0/1)")


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


# 加载模型
loaded_models = {}


def load_model(model_name: str):
    """加载指定模型"""
    if model_name not in loaded_models:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
        with open(model_path, 'rb') as f:
            loaded_models[model_name] = pickle.load(f)
        logger.info(f"已加载模型: {model_name}")
    return loaded_models[model_name]


def preprocess_features(features: CreditFeatures) -> pd.DataFrame:
    """预处理特征"""
    data = features.model_dump()
    df = pd.DataFrame([data])

    # 类别变量编码（与训练时一致）
    education_map = {'高中': 0, '大专': 1, '本科': 2, '硕士': 3, '博士': 4}
    marital_map = {'单身': 0, '已婚': 1, '离异': 2}

    df['学历'] = education_map.get(features.学历, 2)
    df['婚姻状况'] = marital_map.get(features.婚姻状况, 1)

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
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/models")
async def list_models():
    """列出可用模型"""
    available_models = []
    if os.path.exists(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith('.pkl'):
                available_models.append(f.replace('.pkl', ''))
    return {"models": available_models}


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CreditFeatures, model_name: str = DEFAULT_MODEL):
    """单个预测"""
    try:
        model = load_model(model_name)
        df = preprocess_features(features)

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return PredictionResponse(
            model_name=model_name,
            default_probability=round(float(probability), 4),
            is_default=bool(prediction),
            risk_level=get_risk_level(probability),
            features_used=list(df.columns)
        )
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, model_name: str = DEFAULT_MODEL):
    """批量预测"""
    try:
        model = load_model(model_name)
        predictions = []

        for features in request.instances:
            df = preprocess_features(features)
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]

            predictions.append(PredictionResponse(
                model_name=model_name,
                default_probability=round(float(probability), 4),
                is_default=bool(prediction),
                risk_level=get_risk_level(probability),
                features_used=list(df.columns)
            ))

        default_count = sum(1 for p in predictions if p.is_default)

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            default_count=default_count,
            default_rate=round(default_count / len(predictions), 4) if predictions else 0
        )
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
