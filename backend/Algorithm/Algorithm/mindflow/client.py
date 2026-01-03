"""
MindFlow Async Client
High-performance async client for production deployments.
"""

import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class PredictionResult:
    """Prediction result container."""
    user_id: int
    engagement: float
    click_prob: float
    watch_time: float
    latency_ms: float
    cached: bool = False


class MindFlowAsyncClient:
    """
    Async client for MindFlow recommendation server.
    
    Usage:
        async with MindFlowAsyncClient("http://localhost:8000") as client:
            result = await client.predict(
                user_id=1,
                content_ids=[101, 102, 103],
                action_types=[0, 1, 0],
                hours=[10, 11, 12],
                days=[1, 1, 1],
            )
            print(result.engagement)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
        max_connections: int = 100,
    ):
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required: pip install httpx")
        
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
        self.max_connections = max_connections
    
    async def __aenter__(self):
        """Async context manager entry."""
        limits = httpx.Limits(max_connections=self.max_connections)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            limits=limits,
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()
    
    async def health(self) -> Dict:
        """Check server health."""
        response = await self.client.get("/health")
        response.raise_for_status()
        return response.json()
    
    async def predict(
        self,
        user_id: int,
        content_ids: List[int],
        action_types: List[int],
        hours: List[int],
        days: List[int],
    ) -> PredictionResult:
        """Get single prediction."""
        response = await self.client.post("/predict", json={
            "user_id": user_id,
            "content_ids": content_ids,
            "action_types": action_types,
            "hours": hours,
            "days": days,
        })
        response.raise_for_status()
        data = response.json()
        return PredictionResult(**data)
    
    async def predict_batch(
        self,
        requests: List[Dict],
    ) -> List[PredictionResult]:
        """Batch prediction for high throughput."""
        response = await self.client.post("/predict/batch", json={
            "requests": requests,
        })
        response.raise_for_status()
        data = response.json()
        return [PredictionResult(**p) for p in data['predictions']]
    
    async def recommend(
        self,
        user_id: int,
        content_history: List[int],
        action_history: List[int],
        candidate_ids: List[int],
        top_k: int = 10,
    ) -> List[Dict]:
        """Get recommendations from candidates."""
        response = await self.client.post("/recommend", params={
            "user_id": user_id,
            "top_k": top_k,
        }, json={
            "content_history": content_history,
            "action_history": action_history,
            "candidate_ids": candidate_ids,
        })
        response.raise_for_status()
        return response.json()['recommendations']
    
    async def predict_parallel(
        self,
        requests: List[Dict],
        max_concurrent: int = 50,
    ) -> List[PredictionResult]:
        """
        Parallel predictions using asyncio.
        Faster than batch for small requests with network latency.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def predict_one(req: Dict) -> PredictionResult:
            async with semaphore:
                return await self.predict(**req)
        
        tasks = [predict_one(req) for req in requests]
        return await asyncio.gather(*tasks)


class MindFlowSyncClient:
    """
    Synchronous client for MindFlow (wrapper around async).
    
    Usage:
        client = MindFlowSyncClient("http://localhost:8000")
        result = client.predict(user_id=1, content_ids=[...], ...)
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", **kwargs):
        self.base_url = base_url
        self.kwargs = kwargs
    
    def predict(self, **kwargs) -> PredictionResult:
        """Synchronous prediction."""
        async def _predict():
            async with MindFlowAsyncClient(self.base_url, **self.kwargs) as client:
                return await client.predict(**kwargs)
        return asyncio.run(_predict())
    
    def predict_batch(self, requests: List[Dict]) -> List[PredictionResult]:
        """Synchronous batch prediction."""
        async def _predict():
            async with MindFlowAsyncClient(self.base_url, **self.kwargs) as client:
                return await client.predict_batch(requests)
        return asyncio.run(_predict())
    
    def health(self) -> Dict:
        """Check server health."""
        async def _health():
            async with MindFlowAsyncClient(self.base_url, **self.kwargs) as client:
                return await client.health()
        return asyncio.run(_health())
