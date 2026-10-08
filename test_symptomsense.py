import json
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.ml_service import ml_engine
from backend.app.services.red_flags import check_emergency_red_flags, extract_symptoms_from_text

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_symptoms_list_endpoint():
    response = client.get("/api/predict/symptoms-list")
    assert response.status_code == 200
    data = response.json()
    assert "symptoms" in data
    assert len(data["symptoms"]) > 50

def test_model_metrics_endpoint():
    response = client.get("/api/predict/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "test_accuracy" in data
    assert data["test_accuracy"] >= 0.95

def test_emergency_red_flag_detection():
    # Test symptom match
    is_emerg, reason = check_emergency_red_flags(["chest_pain", "mild_fatigue"])
    assert is_emerg is True
    assert "chest pain" in reason.lower()
    
    # Test free-text match
    is_emerg_text, reason_text = check_emergency_red_flags([], "Patient is gasping and cannot breathe")
    assert is_emerg_text is True
    
    # Test safe symptoms
    is_safe, _ = check_emergency_red_flags(["sneezing", "runny_nose"])
    assert is_safe is False

def test_natural_language_symptom_extractor():
    text = "I have a terrible migraine headache, high fever, and I am coughing."
    extracted = extract_symptoms_from_text(text, ml_engine.symptoms)
    assert len(extracted) > 0
    assert any("headache" in s or "fever" in s or "cough" in s for s in extracted)

def test_guest_prediction():
    payload = {
        "symptoms": ["wheezing", "shortness_of_breath", "chest_tightness"],
        "description": "Difficulty breathing, whistling chest sound",
        "severity": 7,
        "duration": "2 days",
        "onset": "gradual",
        "include_profile_history": False
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_condition" in data
    assert "Bronchial Asthma" in data["top_condition"]
    assert len(data["predictions"]) >= 3
    assert len(data["explainability"]) > 0

def test_demo_user_auth_and_profile():
    # Test login with seeded demo user John Doe
    login_payload = {
        "email": "john.doe@example.com",
        "password": "Password123!"
    }
    response = client.post("/api/auth/login", json=login_payload)
    assert response.status_code == 200
    token_data = response.json()
    assert "access_token" in token_data
    token = token_data["access_token"]
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test /api/auth/me
    me_resp = client.get("/api/auth/me", headers=headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == "john.doe@example.com"
    
    # Test /api/profile
    profile_resp = client.get("/api/profile", headers=headers)
    assert profile_resp.status_code == 200
    profile_data = profile_resp.json()
    assert profile_data["age"] == 58
    assert "Hypertension" in profile_data["chronic_conditions"]
    
    # Test /api/records
    records_resp = client.get("/api/records", headers=headers)
    assert records_resp.status_code == 200
    assert len(records_resp.json()) >= 1
    
    # Test /api/analytics/dashboard
    analytics_resp = client.get("/api/analytics/dashboard", headers=headers)
    assert analytics_resp.status_code == 200
    analytics_data = analytics_resp.json()
    assert "vitals" in analytics_data
    assert len(analytics_data["vitals"]["bp_series"]) >= 1

def test_fhir_export_and_import():
    login_payload = {
        "email": "jane.smith@example.com",
        "password": "Password123!"
    }
    login_resp = client.post("/api/auth/login", json=login_payload)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Export FHIR Bundle
    export_resp = client.get("/api/fhir/export", headers=headers)
    assert export_resp.status_code == 200
    bundle = export_resp.json()
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert len(bundle["entry"]) >= 1
    
    # Import FHIR Bundle test
    sample_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Condition",
                    "code": {"text": "Seasonal Allergies"}
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {
                        "coding": [{"code": "8480-6", "display": "Systolic blood pressure"}],
                        "text": "Systolic Blood Pressure"
                    },
                    "valueQuantity": {"value": 122, "unit": "mmHg"}
                }
            }
        ]
    }
    import_resp = client.post("/api/fhir/import", json={"bundle": sample_bundle}, headers=headers)
    assert import_resp.status_code == 200
    assert import_resp.json()["status"] == "success"
