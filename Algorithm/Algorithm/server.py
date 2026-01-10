"""
MindFlow Production Recommendation Server
FastAPI-based recommendation engine with DynamoDB integration.
Connects to real user behavior data for personalized recommendations.
"""

import os
import time
import hashlib
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from fastapi import FastAPI, HTTPException
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
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes

# DynamoDB Tables
TABLES = {
    'USER_BEHAVIOR': 'Buddylynk_UserBehavior',
    'POSTS': 'Buddylynk_Posts',
    'USERS': 'Buddylynk_Users',
    'OTT_VIDEOS': 'buddylynk-ott-videos',
    'SAVES': 'Buddylynk_Saves',
    'FOLLOWS': 'Buddylynk_Follows'
}

# Action types and weights
ACTION_WEIGHTS = {
    0: 1.0,   # VIEW - base engagement
    1: 3.0,   # LIKE - strong positive
    2: 5.0,   # SHARE - very strong positive
    3: 4.0,   # COMMENT - high engagement
    4: 4.0,   # SAVE - strong intent
    5: -0.5,  # SKIP - mild negative
    6: -2.0,  # UNLIKE - negative
    7: -2.0   # UNSAVE - negative
}


# ============================================================================
# Request/Response Models
# ============================================================================

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    limit: int = Field(default=20, ge=1, le=100)
    content_type: str = Field(default="all", description="all, posts, or videos")


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    algorithm: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    uptime_seconds: float
    total_requests: int


# ============================================================================
# DynamoDB Client
# ============================================================================

class DynamoDBClient:
    """DynamoDB client with caching."""
    
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
        
        # Test connection
        try:
            self.client.describe_table(TableName=TABLES['POSTS'])
            self.connected = True
            print("✅ DynamoDB connected!")
        except Exception as e:
            print(f"⚠️ DynamoDB connection failed: {e}")
    
    def _cache_key(self, table: str, key: str) -> str:
        return f"{table}:{key}"
    
    def _is_cached(self, cache_key: str) -> bool:
        if cache_key not in self._cache_times:
            return False
        return (time.time() - self._cache_times[cache_key]) < ServerConfig.CACHE_TTL
    
    def get_user_behavior(self, user_id: str, days: int = 30, limit: int = 200) -> List[Dict]:
        """Get user behavior history from DynamoDB."""
        cache_key = self._cache_key('behavior', f"{user_id}:{days}")
        
        if self._is_cached(cache_key):
            return self._cache[cache_key]
        
        try:
            table = self.dynamodb.Table(TABLES['USER_BEHAVIOR'])
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            response = table.query(
                KeyConditionExpression=Key('userId').eq(user_id) & Key('timestamp').gte(start_time),
                Limit=limit,
                ScanIndexForward=False  # Most recent first
            )
            
            items = response.get('Items', [])
            self._cache[cache_key] = items
            self._cache_times[cache_key] = time.time()
            
            return items
        except Exception as e:
            print(f"Error getting behavior for {user_id}: {e}")
            return []
    
    def get_all_posts(self, limit: int = 500) -> List[Dict]:
        """Get all posts for scoring."""
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
        """Get all OTT videos for scoring."""
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
            print(f"Error getting follows for {user_id}: {e}")
            return []


# ============================================================================
# Recommendation Engine
# ============================================================================

class RecommendationEngine:
    """Production recommendation engine using real user behavior."""
    
    def __init__(self, db: DynamoDBClient):
        self.db = db
        self.request_count = 0
        self.total_latency = 0.0
    
    def compute_user_preferences(self, behavior: List[Dict]) -> Dict[str, float]:
        """Compute user preferences from behavior history."""
        content_scores = defaultdict(float)
        creator_scores = defaultdict(float)
        hour_activity = defaultdict(int)
        
        for item in behavior:
            content_id = item.get('contentId', '')
            action = int(item.get('actionType', 0))
            weight = ACTION_WEIGHTS.get(action, 0)
            creator_id = item.get('contentOwnerId', '')
            hour = int(item.get('hour', 12))
            watch_time = float(item.get('watchTime', 0))
            
            # Score content
            content_scores[content_id] += weight
            
            # Score creators
            if creator_id:
                creator_scores[creator_id] += weight * 0.5
            
            # Track active hours
            hour_activity[hour] += 1
            
            # Bonus for watch time (normalized)
            if watch_time > 0:
                content_scores[content_id] += min(watch_time / 60, 1) * 2
        
        return {
            'content': dict(content_scores),
            'creators': dict(creator_scores),
            'active_hours': dict(hour_activity),
            'total_interactions': len(behavior)
        }
    
    def score_content(self, content: Dict, preferences: Dict, following: List[str]) -> float:
        """Score a piece of content for a user."""
        score = 0.0
        
        content_id = content.get('postId') or content.get('videoId') or ''
        creator_id = content.get('userId') or content.get('creatorId') or ''
        
        # Base engagement score
        likes = int(content.get('likes', 0) or content.get('likeCount', 0) or 0)
        views = int(content.get('viewCount', 0) or content.get('views', 0) or 1)
        comments = int(content.get('commentsCount', 0) or content.get('commentCount', 0) or 0)
        shares = int(content.get('shares', 0) or 0)
        
        engagement_rate = (likes * 2 + comments * 3 + shares * 4) / max(views, 1)
        score += min(engagement_rate * 10, 30)  # Max 30 points for engagement
        
        # Boost if from followed user
        if creator_id in following:
            score += 25
        
        # Boost based on creator affinity
        creator_affinity = preferences.get('creators', {}).get(creator_id, 0)
        score += min(creator_affinity * 5, 20)  # Max 20 points
        
        # Recency boost (newer content gets more points)
        created_at = content.get('createdAt', '')
        if created_at:
            try:
                created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                hours_old = (datetime.now(created_dt.tzinfo) - created_dt).total_seconds() / 3600
                recency_score = max(0, 20 - hours_old / 2)  # Lose 0.5 points per hour
                score += recency_score
            except:
                pass
        
        # Current hour matching
        current_hour = datetime.now().hour
        if preferences.get('active_hours', {}).get(str(current_hour), 0) > 0:
            score += 5
        
        # Variety - penalize already seen content
        if content_id in preferences.get('content', {}):
            score -= 10
        
        # Random factor for discovery (5%)
        score += np.random.random() * 5
        
        return score
    
    def get_recommendations(self, user_id: str, limit: int = 20, content_type: str = "all") -> List[Dict]:
        """Get personalized recommendations for a user."""
        start = time.perf_counter()
        self.request_count += 1
        
        # Get user data
        behavior = self.db.get_user_behavior(user_id)
        following = self.db.get_user_follows(user_id)
        preferences = self.compute_user_preferences(behavior)
        
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
        
        # Score and rank
        scored = []
        for content in all_content:
            score = self.score_content(content, preferences, following)
            scored.append({
                'contentId': content.get('contentId', ''),
                'type': content.get('_type', 'unknown'),
                'score': round(score, 2),
                'creatorId': content.get('userId') or content.get('creatorId'),
            })
        
        # Sort by score and take top N
        scored.sort(key=lambda x: x['score'], reverse=True)
        recommendations = scored[:limit]
        
        latency = (time.perf_counter() - start) * 1000
        self.total_latency += latency
        
        return recommendations
    
    @property
    def stats(self) -> Dict:
        return {
            'total_requests': self.request_count,
            'avg_latency_ms': self.total_latency / max(1, self.request_count),
            'db_connected': self.db.connected
        }


# ============================================================================
# FastAPI Application
# ============================================================================

db: Optional[DynamoDBClient] = None
engine: Optional[RecommendationEngine] = None
start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global db, engine, start_time
    
    print("=" * 60)
    print("🧠 MindFlow Production Recommendation Server")
    print("=" * 60)
    
    # Initialize DynamoDB
    print(f"🔗 Connecting to DynamoDB ({ServerConfig.AWS_REGION})...")
    db = DynamoDBClient(ServerConfig.AWS_REGION)
    engine = RecommendationEngine(db)
    start_time = time.time()
    
    print(f"✅ Server ready!")
    print(f"🌐 http://{ServerConfig.HOST}:{ServerConfig.PORT}")
    print("-" * 60)
    
    yield
    
    print("👋 Shutting down...")


app = FastAPI(
    title="MindFlow Recommendation API",
    description="Production recommendation engine with DynamoDB integration",
    version="2.0.0",
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
    return HealthResponse(
        status="healthy",
        db_connected=db.connected if db else False,
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
    """Get recommendations for a user (GET method for compatibility)."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    start = time.perf_counter()
    recommendations = engine.get_recommendations(user_id, limit)
    latency = (time.perf_counter() - start) * 1000
    
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "algorithm": "mindflow-v2-dynamodb",
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
        algorithm="mindflow-v2-dynamodb",
        latency_ms=round(latency, 2)
    )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server_production:app",
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        workers=1,
        reload=False,
    )
