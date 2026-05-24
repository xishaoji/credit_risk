"""单元测试 - FastAPI部署服务"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd


@pytest.fixture
def mock_model():
    """创建mock模型"""
    model = MagicMock()
    # 返回与输入行数匹配的结果
    def mock_predict(X):
        return np.zeros(len(X), dtype=int)
    def mock_predict_proba(X):
        return np.column_stack([np.full(len(X), 0.85), np.full(len(X), 0.15)])
    model.predict.side_effect = mock_predict
    model.predict_proba.side_effect = mock_predict_proba
    return model


@pytest.fixture
def mock_preprocessor():
    """创建mock预处理器"""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    # 用假数据fit scaler
    fake_data = pd.DataFrame({
        '信用额度使用率': [0.5], '年龄': [35], '逾期30-59天次数': [1],
        '负债比率': [0.3], '月收入': [5000], '信用账户数': [5],
        '房贷数量': [1], '逾期90天以上次数': [0], '逾期60-89天次数': [0],
        '家属人数': [2], '月负债比': [0.0001], '信用风险综合': [0.15],
        '总逾期次数': [1]
    })
    scaler.fit(fake_data)
    return {
        'scaler': scaler,
        'encoders': {},
        'feature_columns': list(fake_data.columns)
    }


@pytest.fixture
def client(mock_model, mock_preprocessor):
    """创建测试客户端"""
    import app as app_module
    # 清除缓存
    app_module.loaded_models.clear()
    app_module.loaded_preprocessor = None

    with patch.object(app_module, 'load_model', return_value=mock_model), \
         patch.object(app_module, 'load_preprocessor', return_value=mock_preprocessor):
        yield TestClient(app_module.app)

    # 清理
    app_module.loaded_models.clear()
    app_module.loaded_preprocessor = None


class TestAPIRoot:
    """测试根路径"""

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestHealthCheck:
    """测试健康检查"""

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestListModels:
    """测试模型列表"""

    def test_list_models(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        assert "models" in response.json()


class TestPredict:
    """测试单条预测"""

    def test_predict_success(self, client):
        valid_request = {
            "信用额度使用率": 0.5,
            "年龄": 35,
            "逾期30-59天次数": 1,
            "负债比率": 0.3,
            "月收入": 5000.0,
            "信用账户数": 5,
            "房贷数量": 1,
            "逾期90天以上次数": 0,
            "逾期60-89天次数": 0,
            "家属人数": 2
        }
        response = client.post("/predict", json=valid_request)
        assert response.status_code == 200
        data = response.json()
        assert "default_probability" in data
        assert "risk_level" in data
        assert "is_default" in data
        assert 0 <= data["default_probability"] <= 1

    def test_predict_invalid_age(self, client):
        invalid_request = {
            "信用额度使用率": 0.5,
            "年龄": 10,  # 低于最小值18
            "逾期30-59天次数": 1,
            "负债比率": 0.3,
            "月收入": 5000.0,
            "信用账户数": 5,
            "房贷数量": 1,
            "逾期90天以上次数": 0,
            "逾期60-89天次数": 0,
            "家属人数": 2
        }
        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422  # Pydantic验证失败

    def test_predict_missing_field(self, client):
        incomplete_request = {
            "年龄": 35
            # 缺少其他必填字段
        }
        response = client.post("/predict", json=incomplete_request)
        assert response.status_code == 422


class TestBatchPredict:
    """测试批量预测"""

    def test_batch_predict(self, client):
        batch_request = {
            "instances": [
                {
                    "信用额度使用率": 0.5, "年龄": 35, "逾期30-59天次数": 1,
                    "负债比率": 0.3, "月收入": 5000.0, "信用账户数": 5,
                    "房贷数量": 1, "逾期90天以上次数": 0,
                    "逾期60-89天次数": 0, "家属人数": 2
                },
                {
                    "信用额度使用率": 0.8, "年龄": 50, "逾期30-59天次数": 3,
                    "负债比率": 0.6, "月收入": 3000.0, "信用账户数": 8,
                    "房贷数量": 2, "逾期90天以上次数": 1,
                    "逾期60-89天次数": 2, "家属人数": 3
                }
            ]
        }
        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert "predictions" in data
        assert "default_rate" in data
