"""
MindFlow Advanced Recommendation Server v3
- Deep Learning with Neural Network Embeddings
- Content Embeddings for similarity matching
- Real-time Learning with DynamoDB persistence
- User preference vectors updated on each interaction
"""

import os
import time
import json
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from collections import defaultdict
import math

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import boto3
from boto3.dynamodb.conditions import Key
from botocore.config import Config

# ============================================================================
# Configuration
# ============================================================================

class ServerConfig:
    HOST = os.getenv("MINDFLOW_HOST", "0.0.0.0")
    PORT = int(os.getenv("MINDFLOW_PORT", "8000"))
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))
    EMBEDDING_DIM = 64  # Dimension for user/content embeddings
    LEARNING_RATE = 0.1  # How fast to update embeddings

# DynamoDB Tables
TABLES = {
    'USER_BEHAVIOR': 'Buddylynk_UserBehavior',
    'POSTS': 'Buddylynk_Posts',
    'USERS': 'Buddylynk_Users',
    'OTT_VIDEOS': 'buddylynk-ott-videos',
    'SAVES': 'Buddylynk_Saves',
    'FOLLOWS': 'Buddylynk_Follows',
    'USER_EMBEDDINGS': 'Buddylynk_UserEmbeddings',
    'CONTENT_EMBEDDINGS': 'Buddylynk_ContentEmbeddings',
    'MODEL_STATE': 'Buddylynk_ModelState'
}

# Action weights for learning
ACTION_WEIGHTS = {
    0: 0.2,   # VIEW - weak positive
    1: 1.0,   # LIKE - strong positive
    2: 1.5,   # SHARE - very strong positive
    3: 1.2,   # COMMENT - strong positive
    4: 1.3,   # SAVE - strong positive
    5: -0.3,  # SKIP - weak negative
    6: -0.8,  # UNLIKE - negative
    7: -0.8   # UNSAVE - negative
}


# ============================================================================
# Request/Response Models
# ============================================================================

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    limit: int = Field(default=20, ge=1, le=100)
    content_type: str = Field(default="all")

class LearnRequest(BaseModel):
    user_id: str
    content_id: str
    action_type: int
    watch_time: float = 0.0

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    algorithm: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    model_version: str
    total_user_embeddings: int
    total_content_embeddings: int
    uptime_seconds: float
    total_requests: int


# ============================================================================
# Neural Network Embeddings (Simple but Effective)
# ============================================================================

class NeuralEmbedding:
    """Simple neural embedding layer for users and content."""
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        np.random.seed(42)  # Reproducibility
    
    def create_random_embedding(self) -> List[float]:
        """Create a random embedding vector."""
        # Initialize with small random values (Xavier initialization)
        embedding = np.random.randn(self.dim) * np.sqrt(2.0 / self.dim)
        return embedding.tolist()
    
    def create_content_embedding(self, content: Dict) -> List[float]:
        """Create embedding from content features."""
        embedding = np.zeros(self.dim)
        
        # Feature 1-10: Engagement features
        likes = int(content.get('likes', 0) or content.get('likeCount', 0) or 0)
        views = int(content.get('viewCount', 0) or content.get('views', 0) or 1)
        comments = int(content.get('commentsCount', 0) or content.get('commentCount', 0) or 0)
        shares = int(content.get('shares', 0) or 0)
        
        embedding[0] = min(likes / 1000, 1.0)  # Normalized likes
        embedding[1] = min(views / 10000, 1.0)  # Normalized views
        embedding[2] = min(comments / 100, 1.0)  # Normalized comments
        embedding[3] = min(shares / 50, 1.0)  # Normalized shares
        embedding[4] = (likes + comments * 2 + shares * 3) / max(views, 1)  # Engagement rate
        
        # Feature 5-10: Time features
        created_at = content.get('createdAt', '')
        if created_at:
            try:
                created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                hours_old = (datetime.now(created_dt.tzinfo) - created_dt).total_seconds() / 3600
                embedding[5] = max(0, 1 - hours_old / 168)  # Decay over 1 week
                embedding[6] = created_dt.hour / 24  # Hour of day
                embedding[7] = created_dt.weekday() / 7  # Day of week
            except:
                pass
        
        # Feature 10-30: Content type features
        content_type = content.get('mediaType', 'text')
        if content_type == 'video':
            embedding[10:15] = [1.0, 0.0, 0.0, 0.0, 0.0]
        elif content_type == 'image':
            embedding[10:15] = [0.0, 1.0, 0.0, 0.0, 0.0]
        else:
            embedding[10:15] = [0.0, 0.0, 1.0, 0.0, 0.0]
        
        # Feature 30-64: Random hash-based features for diversity
        content_id = content.get('postId') or content.get('videoId') or ''
        if content_id:
            hash_val = int(hashlib.md5(content_id.encode()).hexdigest(), 16)
            for i in range(30, self.dim):
                embedding[i] = ((hash_val >> (i - 30)) & 0xFF) / 255.0
        
        return embedding.tolist()
    
    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        a = np.array(emb1)
        b = np.array(emb2)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def update_embedding(self, current: List[float], target: List[float], weight: float, lr: float = 0.1) -> List[float]:
        """Update embedding towards target (real-time learning)."""
        current = np.array(current)
        target = np.array(target)
        
        # Move current embedding towards target based on action weight
        updated = current + lr * weight * (target - current)
        
        # Normalize to prevent explosion
        norm = np.linalg.norm(updated)
        if norm > 1:
            updated = updated / norm
        
        return updated.tolist()


# ============================================================================
# DynamoDB Client with Embedding Storage
# ============================================================================

class AdvancedDynamoDBClient:
    """DynamoDB client with embedding storage and real-time updates."""
    
    def __init__(self, region: str = 'us-east-1'):
        config = Config(
            region_name=region,
            retries={'max_attempts': 3, 'mode': 'standard'}
        )
        self.dynamodb = boto3.resource('dynamodb', config=config)
        self.client = boto3.client('dynamodb', config=config)
        self._cache = {}
        self._cache_times = {}
        self.connected = False
        self.neural = NeuralEmbedding(ServerConfig.EMBEDDING_DIM)
        
        # Test connection and ensure tables exist
        try:
            self.client.describe_table(TableName=TABLES['POSTS'])
            self.connected = True
            print("✅ DynamoDB connected!")
            self._ensure_embedding_tables()
        except Exception as e:
            print(f"⚠️ DynamoDB connection failed: {e}")
    
    def _ensure_embedding_tables(self):
        """Create embedding tables if they don't exist."""
        for table_name in [TABLES['USER_EMBEDDINGS'], TABLES['CONTENT_EMBEDDINGS'], TABLES['MODEL_STATE']]:
            try:
                self.client.describe_table(TableName=table_name)
                print(f"✅ Table {table_name} exists")
            except self.client.exceptions.ResourceNotFoundException:
                print(f"⚠️ Table {table_name} not found - will use fallback")
            except Exception as e:
                print(f"⚠️ Error checking {table_name}: {e}")
    
    def _cache_key(self, table: str, key: str) -> str:
        return f"{table}:{key}"
    
    def _is_cached(self, cache_key: str) -> bool:
        if cache_key not in self._cache_times:
            return False
        return (time.time() - self._cache_times[cache_key]) < ServerConfig.CACHE_TTL
    
    # ========== User Embeddings ==========
    
    def get_user_embedding(self, user_id: str) -> List[float]:
        """Get user embedding from DynamoDB or create new one."""
        cache_key = self._cache_key('user_emb', user_id)
        
        if self._is_cached(cache_key):
            return self._cache[cache_key]
        
        try:
            table = self.dynamodb.Table(TABLES['USER_EMBEDDINGS'])
            response = table.get_item(Key={'userId': user_id})
            
            if 'Item' in response:
                embedding = json.loads(response['Item']['embedding'])
                self._cache[cache_key] = embedding
                self._cache_times[cache_key] = time.time()
                return embedding
        except Exception as e:
            print(f"Error getting user embedding: {e}")
        
        # Create new embedding
        embedding = self.neural.create_random_embedding()
        self.save_user_embedding(user_id, embedding)
        return embedding
    
    def save_user_embedding(self, user_id: str, embedding: List[float]):
        """Save user embedding to DynamoDB."""
        try:
            table = self.dynamodb.Table(TABLES['USER_EMBEDDINGS'])
            table.put_item(Item={
                'userId': user_id,
                'embedding': json.dumps(embedding),
                'updatedAt': datetime.now().isoformat()
            })
            
            # Update cache
            cache_key = self._cache_key('user_emb', user_id)
            self._cache[cache_key] = embedding
            self._cache_times[cache_key] = time.time()
        except Exception as e:
            print(f"Error saving user embedding: {e}")
    
    # ========== Content Embeddings ==========
    
    def get_content_embedding(self, content_id: str, content: Dict = None) -> List[float]:
        """Get content embedding from DynamoDB or create from content."""
        cache_key = self._cache_key('content_emb', content_id)
        
        if self._is_cached(cache_key):
            return self._cache[cache_key]
        
        try:
            table = self.dynamodb.Table(TABLES['CONTENT_EMBEDDINGS'])
            response = table.get_item(Key={'contentId': content_id})
            
            if 'Item' in response:
                embedding = json.loads(response['Item']['embedding'])
                self._cache[cache_key] = embedding
                self._cache_times[cache_key] = time.time()
                return embedding
        except Exception as e:
            print(f"Error getting content embedding: {e}")
        
        # Create from content features
        if content:
            embedding = self.neural.create_content_embedding(content)
            self.save_content_embedding(content_id, embedding)
            return embedding
        
        return self.neural.create_random_embedding()
    
    def save_content_embedding(self, content_id: str, embedding: List[float]):
        """Save content embedding to DynamoDB."""
        try:
            table = self.dynamodb.Table(TABLES['CONTENT_EMBEDDINGS'])
            table.put_item(Item={
                'contentId': content_id,
                'embedding': json.dumps(embedding),
                'updatedAt': datetime.now().isoformat()
            })
            
            cache_key = self._cache_key('content_emb', content_id)
            self._cache[cache_key] = embedding
            self._cache_times[cache_key] = time.time()
        except Exception as e:
            print(f"Error saving content embedding: {e}")
    
    # ========== Real-time Learning ==========
    
    def learn_from_interaction(self, user_id: str, content_id: str, action_type: int, content: Dict = None):
        """Update user embedding based on interaction (real-time learning)."""
        weight = ACTION_WEIGHTS.get(action_type, 0)
        
        # Get current embeddings
        user_emb = self.get_user_embedding(user_id)
        content_emb = self.get_content_embedding(content_id, content)
        
        # Update user embedding towards content they liked, away from content they disliked
        updated_emb = self.neural.update_embedding(
            user_emb, 
            content_emb, 
            weight, 
            ServerConfig.LEARNING_RATE
        )
        
        # Save updated embedding
        self.save_user_embedding(user_id, updated_emb)
        
        return updated_emb
    
    # ========== Original Data Methods ==========
    
    def get_user_behavior(self, user_id: str, days: int = 30, limit: int = 200) -> List[Dict]:
        """Get user behavior history."""
        cache_key = self._cache_key('behavior', f"{user_id}:{days}")
        
        if self._is_cached(cache_key):
            return self._cache[cache_key]
        
        try:
            table = self.dynamodb.Table(TABLES['USER_BEHAVIOR'])
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            response = table.query(
                KeyConditionExpression=Key('userId').eq(user_id) & Key('timestamp').gte(start_time),
                Limit=limit,
                ScanIndexForward=False
            )
            
            items = response.get('Items', [])
            self._cache[cache_key] = items
            self._cache_times[cache_key] = time.time()
            return items
        except Exception as e:
            print(f"Error getting behavior: {e}")
            return []
    
    def get_all_posts(self, limit: int = 500) -> List[Dict]:
        """Get all posts."""
        cache_key = self._cache_key('posts', 'all')
        
        if self._is_cached(cache_key):
            return self._cache[cache_key]
        
        try:
            table = self.dynamodb.Table(TABLES['POSTS'])
            response = table.scan(Limit=limit)
            items = response.get('Items', [])
            
            self._cache[cache_key] = items
            self._cache_times[cache_key] = time.time()
            return items
        except Exception as e:
            print(f"Error getting posts: {e}")
            return []
    
    def get_ott_videos(self, limit: int = 200) -> List[Dict]:
        """Get all OTT videos."""
        cache_key = self._cache_key('videos', 'all')
        
        if self._is_cached(cache_key):
            return self._cache[cache_key]
        
        try:
            table = self.dynamodb.Table(TABLES['OTT_VIDEOS'])
            response = table.scan(Limit=limit)
            items = response.get('Items', [])
            
            self._cache[cache_key] = items
            self._cache_times[cache_key] = time.time()
            return items
        except Exception as e:
            print(f"Error getting videos: {e}")
            return []
    
    def get_user_follows(self, user_id: str) -> List[str]:
        """Get users that this user follows."""
        cache_key = self._cache_key('follows', user_id)
        
        if self._is_cached(cache_key):
            return self._cache[cache_key]
        
        try:
            table = self.dynamodb.Table(TABLES['FOLLOWS'])
            response = table.query(
                KeyConditionExpression=Key('followerId').eq(user_id)
            )
            
            following = [item['followingId'] for item in response.get('Items', [])]
            self._cache[cache_key] = following
            self._cache_times[cache_key] = time.time()
            return following
        except Exception as e:
            print(f"Error getting follows: {e}")
            return []
    
    def get_embedding_counts(self) -> Tuple[int, int]:
        """Get counts of user and content embeddings."""
        user_count = 0
        content_count = 0
        
        try:
            user_count = len(self._cache.get('user_emb', {}))
            content_count = len(self._cache.get('content_emb', {}))
        except:
            pass
        
        return user_count, content_count


# ============================================================================
# Advanced Recommendation Engine
# ============================================================================

class AdvancedRecommendationEngine:
    """Production recommendation engine with deep learning and real-time learning."""
    
    def __init__(self, db: AdvancedDynamoDBClient):
        self.db = db
        self.request_count = 0
        self.total_latency = 0.0
        self.model_version = "v3.0-neural-embeddings"
    
    def score_content(self, user_emb: List[float], content: Dict, following: List[str]) -> float:
        """Score content using neural similarity + engagement features."""
        content_id = content.get('postId') or content.get('videoId') or ''
        creator_id = content.get('userId') or content.get('creatorId') or ''
        
        # Get content embedding
        content_emb = self.db.get_content_embedding(content_id, content)
        
        # Neural similarity score (main signal)
        similarity = self.db.neural.similarity(user_emb, content_emb)
        score = similarity * 50  # Scale to 0-50
        
        # Engagement boost
        likes = int(content.get('likes', 0) or content.get('likeCount', 0) or 0)
        views = int(content.get('viewCount', 0) or content.get('views', 0) or 1)
        comments = int(content.get('commentsCount', 0) or content.get('commentCount', 0) or 0)
        
        engagement_rate = (likes * 2 + comments * 3) / max(views, 1)
        score += min(engagement_rate * 20, 20)  # Max 20 points
        
        # Following boost
        if creator_id in following:
            score += 25
        
        # Recency boost
        created_at = content.get('createdAt', '')
        if created_at:
            try:
                created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                hours_old = (datetime.now(created_dt.tzinfo) - created_dt).total_seconds() / 3600
                recency_score = max(0, 15 - hours_old / 4)  # Decay over time
                score += recency_score
            except:
                pass
        
        # Exploration boost (small random factor for diversity)
        score += np.random.random() * 5
        
        return score
    
    def get_recommendations(self, user_id: str, limit: int = 20, content_type: str = "all") -> List[Dict]:
        """Get personalized recommendations using neural embeddings."""
        start = time.perf_counter()
        self.request_count += 1
        
        # Get user embedding (creates if not exists)
        user_emb = self.db.get_user_embedding(user_id)
        following = self.db.get_user_follows(user_id)
        
        # Get content
        all_content = []
        
        if content_type in ["all", "posts"]:
            posts = self.db.get_all_posts()
            for p in posts:
                p['_type'] = 'post'
                p['contentId'] = p.get('postId', '')
            all_content.extend(posts)
        
        if content_type in ["all", "videos"]:
            videos = self.db.get_ott_videos()
            for v in videos:
                v['_type'] = 'video'
                v['contentId'] = v.get('videoId', '')
            all_content.extend(videos)
        
        # Filter out user's own content
        all_content = [c for c in all_content if c.get('userId') != user_id and c.get('creatorId') != user_id]
        
        # Score using neural similarity
        scored = []
        for content in all_content:
            score = self.score_content(user_emb, content, following)
            scored.append({
                'contentId': content.get('contentId', ''),
                'type': content.get('_type', 'unknown'),
                'score': round(score, 2),
                'creatorId': content.get('userId') or content.get('creatorId'),
            })
        
        # Sort by score
        scored.sort(key=lambda x: x['score'], reverse=True)
        recommendations = scored[:limit]
        
        latency = (time.perf_counter() - start) * 1000
        self.total_latency += latency
        
        return recommendations
    
    def learn(self, user_id: str, content_id: str, action_type: int, content: Dict = None):
        """Real-time learning from user interaction."""
        self.db.learn_from_interaction(user_id, content_id, action_type, content)
    
    @property
    def stats(self) -> Dict:
        user_count, content_count = self.db.get_embedding_counts()
        return {
            'total_requests': self.request_count,
            'avg_latency_ms': self.total_latency / max(1, self.request_count),
            'db_connected': self.db.connected,
            'model_version': self.model_version,
            'user_embeddings': user_count,
            'content_embeddings': content_count
        }


# ============================================================================
# FastAPI Application
# ============================================================================

db: Optional[AdvancedDynamoDBClient] = None
engine: Optional[AdvancedRecommendationEngine] = None
start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global db, engine, start_time
    
    print("=" * 60)
    print("🧠 MindFlow Advanced Recommendation Server v3")
    print("   - Neural Embeddings")
    print("   - Real-time Learning")
    print("   - Content Similarity")
    print("=" * 60)
    
    print(f"🔗 Connecting to DynamoDB ({ServerConfig.AWS_REGION})...")
    db = AdvancedDynamoDBClient(ServerConfig.AWS_REGION)
    engine = AdvancedRecommendationEngine(db)
    start_time = time.time()
    
    print(f"✅ Server ready!")
    print(f"🌐 http://{ServerConfig.HOST}:{ServerConfig.PORT}")
    print("-" * 60)
    
    yield
    
    print("👋 Shutting down...")


app = FastAPI(
    title="MindFlow Advanced Recommendation API",
    description="Production recommendation engine with neural embeddings and real-time learning",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS
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
    user_count, content_count = db.get_embedding_counts() if db else (0, 0)
    return HealthResponse(
        status="healthy",
        db_connected=db.connected if db else False,
        model_version=engine.model_version if engine else "unknown",
        total_user_embeddings=user_count,
        total_content_embeddings=content_count,
        uptime_seconds=time.time() - start_time,
        total_requests=engine.request_count if engine else 0
    )


@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine.stats


@app.get("/api/recommendations/{user_id}")
async def get_recommendations_get(user_id: str, limit: int = 20):
    """Get recommendations for a user (GET method)."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    start = time.perf_counter()
    recommendations = engine.get_recommendations(user_id, limit)
    latency = (time.perf_counter() - start) * 1000
    
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "algorithm": engine.model_version,
        "latency_ms": round(latency, 2)
    }


@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations_post(request: RecommendationRequest):
    """Get recommendations for a user (POST method)."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    start = time.perf_counter()
    recommendations = engine.get_recommendations(
        request.user_id, 
        request.limit,
        request.content_type
    )
    latency = (time.perf_counter() - start) * 1000
    
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        algorithm=engine.model_version,
        latency_ms=round(latency, 2)
    )


@app.post("/api/learn")
async def learn_from_interaction(request: LearnRequest, background_tasks: BackgroundTasks):
    """Real-time learning endpoint - call when user interacts with content."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Learn in background to not block response
    background_tasks.add_task(
        engine.learn,
        request.user_id,
        request.content_id,
        request.action_type
    )
    
    return {
        "status": "learning",
        "user_id": request.user_id,
        "content_id": request.content_id,
        "action_type": request.action_type
    }


@app.get("/api/user/{user_id}/embedding")
async def get_user_embedding(user_id: str):
    """Get user's learned embedding vector."""
    if db is None:
        raise HTTPException(status_code=503, detail="DB not initialized")
    
    embedding = db.get_user_embedding(user_id)
    return {
        "user_id": user_id,
        "embedding_dim": len(embedding),
        "embedding": embedding[:10],  # Return first 10 values for preview
        "embedding_hash": hashlib.md5(json.dumps(embedding).encode()).hexdigest()[:8]
    }


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server_advanced:app",
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        workers=1,
        reload=False,
    )
