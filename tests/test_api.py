# /tests/test_api.py
from fastapi.testclient import TestClient
import pytest

from recruitair.api.dependencies import get_default_model
from recruitair.api.main import app
from recruitair.api.model import BaseEvaluatorModel


class MockModel(BaseEvaluatorModel):
    def __init__(self):
        self._version = "mock-1"

    @property
    def version(self):
        return self._version

    def predict(self, criteria: str, resume: str) -> float:
        if not criteria.strip() or not resume.strip():
            return 0.0
        if "python" in criteria.lower():
            return 0.92
        return 0.5


app.dependency_overrides[get_default_model] = lambda: MockModel()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_eval_python_match(client: TestClient):
    req = {
        "criteria_description": "Experience with Python and ML",
        "applicant_cv": "Skilled in Python",
    }
    r = client.post("/eval", json=req)
    assert r.status_code == 200, r.text
    data = r.json()
    assert abs(data["score"] - 0.92) < 1e-6
    assert data["model_version"] == "mock-1"


def test_eval_empty_cv_or_criteria(client: TestClient):
    req1 = {"criteria_description": "Python", "applicant_cv": ""}
    r1 = client.post("/eval", json=req1)
    assert r1.status_code == 422

    req2 = {"criteria_description": "", "applicant_cv": "Some text"}
    r2 = client.post("/eval", json=req2)
    assert r2.status_code == 422


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
