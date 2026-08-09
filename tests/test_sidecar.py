"""
Unit and Integration Tests for FastAPI Diagnostic Sidecar & Hook Client
Executes on Mac M4 Air using FastAPI TestClient and mock activation streams.
"""

import os
import sys
import pytest
import asyncio
from fastapi.testclient import TestClient

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from part_b_diagnostic.sidecar.main import app
from part_b_diagnostic.sidecar.hook_client import HookClient

client = TestClient(app)

def test_health_check_endpoint():
    """Verify GET /health returns 200 OK and expected service metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "SynapseGuard" in data["service"]

def test_score_token_safe():
    """Verify POST /score_token classifies high-confidence, low-entropy token as SAFE."""
    payload = {
        "token_text": " Paris",
        "logit_confidence": 0.99,
        "sae_entropy": 0.05
    }
    response = client.post("/score_token", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SAFE"
    assert data["psc_score"] < 0.65
    assert data["ssp_triggered"] is False

def test_score_token_warning_and_halt():
    """Verify POST /score_token triggers WARNING and HALT status bands."""
    # Warning band
    warn_payload = {
        "token_text": " Mars",
        "logit_confidence": 0.50,
        "sae_entropy": 0.85
    }
    res_warn = client.post("/score_token", json=warn_payload)
    assert res_warn.status_code == 200
    assert res_warn.json()["status"] == "WARNING"
    assert res_warn.json()["ssp_triggered"] is True
    
    # Halt band
    halt_payload = {
        "token_text": " hallucinated",
        "logit_confidence": 0.10,
        "sae_entropy": 0.95
    }
    res_halt = client.post("/score_token", json=halt_payload)
    assert res_halt.status_code == 200
    assert res_halt.json()["status"] == "HALT"
    assert res_halt.json()["ssp_triggered"] is True

def test_score_sequence():
    """Verify POST /score_sequence processes token streams and aggregates max PSC score."""
    payload = {
        "prompt": "Who was the first president of Mars in 1984?",
        "tokens": [
            {"token_text": "The", "logit_confidence": 0.95, "sae_entropy": 0.12},
            {"token_text": " president", "logit_confidence": 0.50, "sae_entropy": 0.85},
            {"token_text": " was", "logit_confidence": 0.10, "sae_entropy": 0.95}
        ]
    }
    response = client.post("/score_sequence", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["overall_status"] == "HALT"
    assert data["max_psc_score"] >= 0.85
    assert data["ssp_summary"]["ssp_applied"] is True

@pytest.mark.asyncio
async def test_hook_client_mock_stream():
    """Verify async HookClient yields activation tokens in mock mode."""
    hook = HookClient(mock_mode=True)
    tokens_received = []
    async for item in hook.stream_activations(prompt="Test Prompt"):
        tokens_received.append(item)
        
    assert len(tokens_received) == 4
    assert tokens_received[0]["token_text"] == " The"
