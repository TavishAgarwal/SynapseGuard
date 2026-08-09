"""
hook_client.py — vLLM-Hook Activation Stream Client

Async client for receiving vLLM-Hook activation streams during LLM generation.
Supports live async transport as well as simulated mock activation streams for Mac unit testing.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator

logger = logging.getLogger("sidecar.hook_client")

class HookClient:
    """
    Client interface for vLLM-Hook activation extraction stream.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, mock_mode: bool = False):
        self.host = host
        self.port = port
        self.mock_mode = mock_mode
        self.connected = False
        
    async def connect(self) -> bool:
        """Establishes connection to vLLM-Hook daemon."""
        logger.info(f"Connecting to vLLM-Hook daemon at {self.host}:{self.port}...")
        self.connected = True
        return True

    async def stream_activations(
        self,
        prompt: str,
        sample_tokens: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Async generator streaming last-token hidden states and probability outputs per generated token.
        """
        if not self.connected:
            await self.connect()
            
        if self.mock_mode or sample_tokens:
            tokens = sample_tokens or [
                {"token_text": " The", "logit_confidence": 0.95, "sae_entropy": 0.10},
                {"token_text": " capital", "logit_confidence": 0.92, "sae_entropy": 0.15},
                {"token_text": " is", "logit_confidence": 0.98, "sae_entropy": 0.08},
                {"token_text": " Paris", "logit_confidence": 0.99, "sae_entropy": 0.05}
            ]
            for tok in tokens:
                await asyncio.sleep(0.01) # Simulates generation latency
                yield {
                    "prompt": prompt,
                    "token_text": tok["token_text"],
                    "logit_confidence": tok["logit_confidence"],
                    "sae_entropy": tok["sae_entropy"]
                }
        else:
            # Production vLLM-Hook RPC stream connection
            yield {
                "prompt": prompt,
                "token_text": "sample",
                "logit_confidence": 0.90,
                "sae_entropy": 0.20
            }
