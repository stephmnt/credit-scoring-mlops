import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _build_payload(preprocessor):
    data = {}
    for col in preprocessor.required_input_columns:
        if col in preprocessor.numeric_medians:
            data[col] = preprocessor.numeric_medians[col]
        elif col in preprocessor.categorical_columns:
            data[col] = "Unknown"
        else:
            data[col] = 0
    data["SK_ID_CURR"] = int(data.get("SK_ID_CURR", 100001))
    return {"data": data}


def _pick_required_column(preprocessor, exclude=None):
    exclude = set(exclude or [])
    for col in preprocessor.required_input_columns:
        if col not in exclude:
            return col
    raise AssertionError("No required column available for test.")


def _pick_numeric_range(preprocessor):
    for col, bounds in preprocessor.numeric_ranges.items():
        if col in preprocessor.numeric_required_columns:
            return col, bounds
    raise AssertionError("No numeric range available for test.")


def _pick_numeric_required(preprocessor):
    for col in preprocessor.numeric_required_columns:
        if col != "SK_ID_CURR":
            return col
    raise AssertionError("No numeric required column available for test.")


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_features(client):
    resp = client.get("/features")
    assert resp.status_code == 200
    payload = resp.json()
    assert "input_features" in payload
    assert "required_input_features" in payload
    assert "SK_ID_CURR" in payload["input_features"]
    assert "EXT_SOURCE_2" in payload["input_features"]


def test_predict(client):
    preprocessor = client.app.state.preprocessor
    payload = _build_payload(preprocessor)
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    result = data["predictions"][0]
    assert "sk_id_curr" in result
    assert "prediction" in result
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_missing_required_field(client):
    preprocessor = client.app.state.preprocessor
    payload = _build_payload(preprocessor)
    missing_col = _pick_required_column(preprocessor, exclude={"SK_ID_CURR"})
    payload["data"].pop(missing_col, None)
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    detail = resp.json().get("detail", {})
    assert detail.get("message") == "Missing required input columns."


def test_predict_invalid_type(client):
    preprocessor = client.app.state.preprocessor
    payload = _build_payload(preprocessor)
    invalid_col = _pick_numeric_required(preprocessor)
    payload["data"][invalid_col] = "not_a_number"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    detail = resp.json().get("detail", {})
    assert detail.get("message") == "Invalid numeric values provided."


def test_predict_out_of_range(client):
    preprocessor = client.app.state.preprocessor
    payload = _build_payload(preprocessor)
    col, (min_val, max_val) = _pick_numeric_range(preprocessor)
    payload["data"][col] = max_val + 1
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    detail = resp.json().get("detail", {})
    assert detail.get("message") == "Input contains values outside expected ranges."
