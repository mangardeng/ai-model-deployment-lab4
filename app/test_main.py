
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_valid_prediction():
    response = client.post(
        "/v1/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "setosa"
    assert data["predicted_class_id"] == 0
    assert 0 <= data["confidence"] <= 1
    assert data["model_version"] == "iris_v1"

def test_invalid_prediction():
    response = client.post(
        "/v1/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4
        }
    )

    assert response.status_code == 422

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == "iris_v1"
