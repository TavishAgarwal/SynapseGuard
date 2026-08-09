"""
main.py — SynapseGuard FastAPI Diagnostic Sidecar Service

Exposes real-time PSC scoring endpoints for vLLM generation streams per architecture.md.
"""

import os
import sys
import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from part_b_diagnostic.sidecar.psc_score import (
    calculate_psc,
    classify_psc_status,
    evaluate_token_diagnostic
)
from part_b_diagnostic.sidecar.ssp import evaluate_ssp_trigger, get_ssp_perturbation

app = FastAPI(
    title="SynapseGuard Diagnostic Sidecar",
    description="Real-Time Hallucination Interception Endpoint via PSC Scoring",
    version="2.0.0"
)

class TokenScoreRequest(BaseModel):
    token_text: str = Field(..., description="Generated token string")
    logit_confidence: float = Field(..., ge=0.0, le=1.0, description="Top-1 token logit probability")
    sae_entropy: float = Field(..., ge=0.0, le=1.0, description="SAE latent entropy/sparsity score")
    alpha: Optional[float] = Field(0.5, description="Weight for logit confidence term")
    beta: Optional[float] = Field(0.5, description="Weight for SAE entropy term")

class SequenceScoreRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt text")
    tokens: List[TokenScoreRequest] = Field(..., description="List of generated tokens with activation metrics")

class TokenScoreResponse(BaseModel):
    token_text: str
    logit_confidence: float
    sae_entropy: float
    psc_score: float
    status: str # SAFE, WARNING, HALT
    ssp_triggered: bool
    timestamp: str

class SequenceScoreResponse(BaseModel):
    prompt: str
    token_diagnostics: List[TokenScoreResponse]
    overall_status: str
    max_psc_score: float
    ssp_summary: Dict[str, Any]

@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "SynapseGuard FastAPI Diagnostic Sidecar",
        "version": "2.0.0",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

@app.post("/score_token", response_model=TokenScoreResponse, tags=["Scoring"])
def score_token(request: TokenScoreRequest):
    """
    Computes PSC score and diagnostic status for a single token.
    """
    diag = evaluate_token_diagnostic(
        token_text=request.token_text,
        logit_confidence=request.logit_confidence,
        sae_entropy=request.sae_entropy,
        alpha=request.alpha,
        beta=request.beta
    )
    
    return TokenScoreResponse(
        token_text=diag["token_text"],
        logit_confidence=diag["logit_confidence"],
        sae_entropy=diag["sae_entropy"],
        psc_score=diag["psc_score"],
        status=diag["status"],
        ssp_triggered=diag["ssp_triggered"],
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

@app.post("/score_sequence", response_model=SequenceScoreResponse, tags=["Scoring"])
def score_sequence(request: SequenceScoreRequest):
    """
    Processes a full sequence stream of generated tokens, returning per-token diagnostics and aggregate risk status.
    """
    token_responses = []
    max_psc = 0.0
    ssp_any_triggered = False
    
    for tok in request.tokens:
        diag = evaluate_token_diagnostic(
            token_text=tok.token_text,
            logit_confidence=tok.logit_confidence,
            sae_entropy=tok.sae_entropy,
            alpha=tok.alpha,
            beta=tok.beta
        )
        
        if diag["psc_score"] > max_psc:
            max_psc = diag["psc_score"]
        if diag["ssp_triggered"]:
            ssp_any_triggered = True
            
        token_responses.append(TokenScoreResponse(
            token_text=diag["token_text"],
            logit_confidence=diag["logit_confidence"],
            sae_entropy=diag["sae_entropy"],
            psc_score=diag["psc_score"],
            status=diag["status"],
            ssp_triggered=diag["ssp_triggered"],
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        ))
        
    overall_status = classify_psc_status(max_psc)
    ssp_details = get_ssp_perturbation(request.prompt, overall_status)
    
    return SequenceScoreResponse(
        prompt=request.prompt,
        token_diagnostics=token_responses,
        overall_status=overall_status,
        max_psc_score=max_psc,
        ssp_summary=ssp_details
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
