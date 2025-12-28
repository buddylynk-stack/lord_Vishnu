"""
MindFlow Production Server
FastAPI-based high-performance recommendation API.
"""

import os
import time
import asyncio
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Install onnxruntime: pip install onnxruntime")

try:
    from cachetools import TTLCache, LRUCache
except ImportError:
    TTLCache = dict
    LRUCache = dict


# ============================================================================
# Configuration
# ============================================================================

class ServerConfig:
    MODEL_PATH = os.getenv("MINDFLOW_MODEL", "mindflow.onnx")
    HOST = os.getenv("MINDFLOW_HOST", "0.0.0.0")
    PORT = int(os.getenv("MINDFLOW_PORT", "8000"))
    WORKERS = int(os.getenv("MINDFLOW_WORKERS", "4"))
    CACHE_SIZE = int(os.getenv("MINDFLOW_CACHE_SIZE", "10000"))
    CACHE_TTL = int(os.getenv("MINDFLOW_CACHE_TTL", "300"))  # 5 minutes


# ============================================================================
# Request/Response Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Single prediction request."""
    user_id: int = Field(..., description="User ID")
    content_ids: List[int] = Field(..., min_length=1, max_length=20, description="Recent content IDs")
    action_types: List[int] = Field(..., min_length=1, max_length=20, description="Action types (0-9)")
    hours: List[int] = Field(..., min_length=1, max_length=20, description="Hours (0-23)")
    days: List[int] = Field(..., min_length=1, max_length=20, description="Days (0-6)")


class PredictionResponse(BaseModel):
    """Prediction response."""
    user_id: int
    engagement: float = Field(..., description="Engagement score (0-1)")
    click_prob: float = Field(..., description="Click probability (0-1)")
    watch_time: float = Field(..., description="Predicted watch time (seconds)")
    latency_ms: float = Field(..., description="Inference latency")
    cached: bool = Field(default=False, description="Whether result was cached")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    requests: List[PredictionRequest] = Field(..., max_length=100)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    cache_size: int
    uptime_seconds: float


# ============================================================================
# Inference Engine
# ============================================================================

class ProductionInferenceEngine:
    """High-performance production inference engine."""
    
    SEQUENCE_LENGTH = 20
    
    def __init__(self, model_path: str, num_threads: int = 4, cache_size: int = 10000, cache_ttl: int = 300):
        # ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider'],
        )
        
        # Prediction cache
        if TTLCache != dict:
            self.cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        else:
            self.cache = {}
        
        # Stats
        self.request_count = 0
        self.cache_hits = 0
        self.total_latency = 0.0
        
        # Warmup
        self._warmup()
    
    def _warmup(self, iterations: int = 10):
        """Warmup model for consistent latency."""
        dummy = self._create_dummy_input(1)
        for _ in range(iterations):
            self.session.run(None, dummy)
    
    def _create_dummy_input(self, batch_size: int) -> Dict[str, np.ndarray]:
        return {
            'user_ids': np.ones(batch_size, dtype=np.int64),
            'content_ids': np.ones((batch_size, self.SEQUENCE_LENGTH), dtype=np.int64),
            'action_types': np.zeros((batch_size, self.SEQUENCE_LENGTH), dtype=np.int64),
            'hours': np.zeros((batch_size, self.SEQUENCE_LENGTH), dtype=np.int64),
            'days': np.zeros((batch_size, self.SEQUENCE_LENGTH), dtype=np.int64),
        }
    
    def _pad_sequence(self, seq: List[int], default: int = 0) -> np.ndarray:
        arr = np.full(self.SEQUENCE_LENGTH, default, dtype=np.int64)
        seq = seq[-self.SEQUENCE_LENGTH:]
        arr[-len(seq):] = seq
        return arr
    
    def _make_cache_key(self, user_id: int, content_ids: tuple) -> str:
        return f"{user_id}:{hash(content_ids)}"
    
    def predict(self, request: PredictionRequest) -> Dict:
        """Single prediction with caching."""
        start = time.perf_counter()
        self.request_count += 1
        
        # Check cache
        cache_key = self._make_cache_key(request.user_id, tuple(request.content_ids))
        if cache_key in self.cache:
            self.cache_hits += 1
            result = self.cache[cache_key].copy()
            result['cached'] = True
            result['latency_ms'] = (time.perf_counter() - start) * 1000
            return result
        
        # Prepare inputs
        inputs = {
            'user_ids': np.array([request.user_id], dtype=np.int64),
            'content_ids': self._pad_sequence(request.content_ids).reshape(1, -1),
            'action_types': self._pad_sequence(request.action_types).reshape(1, -1),
            'hours': self._pad_sequence(request.hours).reshape(1, -1),
            'days': self._pad_sequence(request.days).reshape(1, -1),
        }
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        latency = (time.perf_counter() - start) * 1000
        self.total_latency += latency
        
        result = {
            'user_id': request.user_id,
            'engagement': float(outputs[0][0]),
            'click_prob': float(outputs[1][0]),
            'watch_time': float(outputs[2][0]),
            'latency_ms': latency,
            'cached': False,
        }
        
        # Cache result
        self.cache[cache_key] = result.copy()
        
        return result
    
    def predict_batch(self, requests: List[PredictionRequest]) -> List[Dict]:
        """Batch prediction for multiple requests."""
        start = time.perf_counter()
        batch_size = len(requests)
        
        # Prepare batch inputs
        user_ids = np.array([r.user_id for r in requests], dtype=np.int64)
        content_ids = np.array([self._pad_sequence(r.content_ids) for r in requests], dtype=np.int64)
        action_types = np.array([self._pad_sequence(r.action_types) for r in requests], dtype=np.int64)
        hours = np.array([self._pad_sequence(r.hours) for r in requests], dtype=np.int64)
        days = np.array([self._pad_sequence(r.days) for r in requests], dtype=np.int64)
        
        inputs = {
            'user_ids': user_ids,
            'content_ids': content_ids,
            'action_types': action_types,
            'hours': hours,
            'days': days,
        }
        
        # Run batch inference
        outputs = self.session.run(None, inputs)
        
        latency = (time.perf_counter() - start) * 1000
        per_item_latency = latency / batch_size
        
        results = []
        for i, request in enumerate(requests):
            results.append({
                'user_id': request.user_id,
                'engagement': float(outputs[0][i]),
                'click_prob': float(outputs[1][i]),
                'watch_time': float(outputs[2][i]),
                'latency_ms': per_item_latency,
                'cached': False,
            })
        
        return results
    
    @property
    def stats(self) -> Dict:
        return {
            'request_count': self.request_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.request_count),
            'avg_latency_ms': self.total_latency / max(1, self.request_count),
            'cache_size': len(self.cache),
        }


# ============================================================================
# FastAPI Application
# ============================================================================

# Global engine
engine: Optional[ProductionInferenceEngine] = None
start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global engine, start_time
    
    # Startup
    print(f"🚀 Loading model: {ServerConfig.MODEL_PATH}")
    engine = ProductionInferenceEngine(
        model_path=ServerConfig.MODEL_PATH,
        num_threads=ServerConfig.WORKERS,
        cache_size=ServerConfig.CACHE_SIZE,
        cache_ttl=ServerConfig.CACHE_TTL,
    )
    start_time = time.time()
    print(f"✅ Model loaded! Server ready.")
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")


app = FastAPI(
    title="MindFlow Recommendation API",
    description="Production-ready social media recommendation engine",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=engine is not None,
        cache_size=len(engine.cache) if engine else 0,
        uptime_seconds=time.time() - start_time,
    )


@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return engine.stats


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = engine.predict(request)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint for high throughput."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.perf_counter()
    results = engine.predict_batch(request.requests)
    total_latency = (time.perf_counter() - start) * 1000
    
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**r) for r in results],
        total_latency_ms=total_latency,
    )


@app.post("/recommend")
async def get_recommendations(
    user_id: int,
    content_history: List[int],
    action_history: List[int],
    candidate_ids: List[int],
    top_k: int = 10,
):
    """Get top-K recommendations from candidate content."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Score each candidate
    hour = time.localtime().tm_hour
    day = time.localtime().tm_wday
    
    scores = []
    for content_id in candidate_ids:
        request = PredictionRequest(
            user_id=user_id,
            content_ids=content_history[-19:] + [content_id],
            action_types=action_history[-19:] + [0],
            hours=[hour] * 20,
            days=[day] * 20,
        )
        result = engine.predict(request)
        scores.append({
            'content_id': content_id,
            'score': result['engagement'] * 0.4 + result['click_prob'] * 0.4 + min(result['watch_time'] / 60, 1) * 0.2,
        })
    
    # Sort and return top-K
    scores.sort(key=lambda x: x['score'], reverse=True)
    return {
        'recommendations': scores[:top_k],
        'user_id': user_id,
    }


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Run production server."""
    print("=" * 60)
    print("🧠 MindFlow Production Server")
    print("=" * 60)
    
    if not os.path.exists(ServerConfig.MODEL_PATH):
        print(f"❌ Model not found: {ServerConfig.MODEL_PATH}")
        print("\n📖 First train and export a model:")
        print("   python train.py --epochs 50")
        print("   python export_onnx.py --checkpoint models/best_model.pt")
        return
    
    print(f"📦 Model: {ServerConfig.MODEL_PATH}")
    print(f"🌐 Host: {ServerConfig.HOST}:{ServerConfig.PORT}")
    print(f"⚡ Workers: {ServerConfig.WORKERS}")
    print("-" * 60)
    
    uvicorn.run(
        "server:app",
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        workers=1,  # Use 1 worker since ONNX is already parallel
        reload=False,
    )


if __name__ == "__main__":
    main()
