from fastapi.testclient import TestClient
import pytest
from api_evaluator.main import app
from api_evaluator.schemas import EvalRequest, EvalResponse
from api_evaluator.model import BaseEvaluatorModel

# A tiny mock model to use during tests (deterministic)
class MockModel(BaseEvaluatorModel):
    def __init__(self):
        self._version = "mock-1"
    @property
    def version(self):
        return self._version
    def predict(self, criteria: str, resume: str) -> float:
        # produce deterministic results for tests:
        if "python" in criteria.lower():
            # pretend resume contains Python keyword => high match
            return 0.92
        if len(resume.strip()) == 0 or len(criteria.strip()) == 0:
            return 0.0
        # otherwise medium
        return 0.5

@pytest.fixture(autouse=True)
def override_model_dependency(monkeypatch):
    # override the get_model dependency to return MockModel
    from app.main import get_model
    # monkeypatching the dependency by setting app.dependency_overrides
    from app.main import app as _app
    _app.dependency_overrides[get_model] = lambda: MockModel()
    yield
    _app.dependency_overrides.clear()

def test_eval_python_match():
    client = TestClient(app)
    req = {"criteria_description": "Experience with Python and ML", "applicant_cv": "Skilled in Python, pandas, sklearn"}
    r = client.post("/eval", json=req)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "score" in data
    assert 0.0 <= data["score"] <= 1.0
    # our mock returns 0.92 for "python"
    assert abs(data["score"] - 0.92) < 1e-6
    assert data.get("model_version") == "mock-1"
    assert data.get("elapsed_seconds", 0) >= 0

def test_eval_empty_cv_or_criteria():
    client = TestClient(app)
    # empty resume
    req1 = {"criteria_description": "Python", "applicant_cv": ""}
    r1 = client.post("/eval", json=req1)
    assert r1.status_code == 200
    assert r1.json()["score"] == 0.0

    # empty criteria (should be validated by pydantic -> 422)
    req2 = {"criteria_description": "", "applicant_cv": "Some text"}
    r2 = client.post("/eval", json=req2)
    assert r2.status_code == 422

def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
