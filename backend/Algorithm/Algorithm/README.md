# 🧠 MindFlow - Production Social Media Recommendation Algorithm

A high-performance neural network recommendation system with multi-processing, caching, and async support.

## ⚡ Performance

| Mode | Throughput | Latency |
|------|-----------|---------|
| Single inference | 250-300/sec | ~3-5ms |
| Batch (64) | 2000+/sec | <1ms/item |
| Worker pool (4) | 1000+/sec | ~2ms |
| Async parallel | 500+/sec | ~5ms |

## 🚀 Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Train & Export

```bash
# Train (production: 100K samples)
python train.py --epochs 50 --num-samples 100000 --output-dir models

# Export to ONNX
python export_onnx.py --checkpoint models/best_model.pt --output mindflow.onnx
```

### Run Server

```bash
# Start FastAPI server
python server.py

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## 📦 Production Features

### FastAPI Server (`server.py`)
- REST API with Swagger docs
- Request caching (TTL-based)
- Batch prediction endpoint
- Health checks & metrics
- CORS support

### Multi-Process Workers (`mindflow/worker_pool.py`)
- Parallel CPU inference
- Auto-scaling workers
- Load balancing

### Async Client (`mindflow/client.py`)
- Async/sync HTTP clients
- Connection pooling
- Parallel requests

### Docker Deployment
```bash
# Build and run
docker-compose up -d

# Scale workers
docker-compose up -d --scale mindflow=4
```

## 🔌 API Usage

### Python Client
```python
from mindflow import MindFlowInference

# Direct ONNX inference
engine = MindFlowInference('mindflow.onnx')
result = engine.predict(
    user_id=1,
    content_history=[101, 102, 103, ...],
    action_history=[0, 1, 0, ...],
    hour_history=[10, 11, 12, ...],
    day_history=[1, 1, 1, ...],
)
print(result)
# {'engagement': 0.65, 'click_prob': 0.42, 'watch_time': 25.3}
```

### HTTP API
```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "content_ids": [101, 102, 103],
    "action_types": [0, 1, 0],
    "hours": [10, 11, 12],
    "days": [1, 1, 1]
  }'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"requests": [...]}'

# Health check
curl http://localhost:8000/health
```

### Async Client
```python
from mindflow.client import MindFlowAsyncClient

async with MindFlowAsyncClient("http://localhost:8000") as client:
    result = await client.predict(user_id=1, ...)
```

## 📁 Project Structure

```
Algorithm/
├── train.py              # Training script
├── export_onnx.py        # ONNX export
├── server.py             # FastAPI server
├── demo.py               # Benchmarking
├── Dockerfile            # Container build
├── docker-compose.yml    # Multi-container
├── requirements.txt      # Dependencies
└── mindflow/
    ├── models/           # Neural networks
    ├── training/         # Training pipeline
    ├── onnx/            # ONNX utilities
    ├── inference.py      # Direct inference API
    ├── client.py         # HTTP clients
    └── worker_pool.py    # Multi-process pool
```

## 🔧 Configuration

Environment variables:
- `MINDFLOW_MODEL` - Model path (default: mindflow.onnx)
- `MINDFLOW_HOST` - Server host (default: 0.0.0.0)
- `MINDFLOW_PORT` - Server port (default: 8000)
- `MINDFLOW_WORKERS` - CPU threads (default: 4)
- `MINDFLOW_CACHE_SIZE` - Cache entries (default: 10000)
- `MINDFLOW_CACHE_TTL` - Cache TTL seconds (default: 300)

## License

MIT
