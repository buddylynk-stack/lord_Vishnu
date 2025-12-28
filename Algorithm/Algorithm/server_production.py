#!/usr/bin/env python3
"""
MindFlow Recommendation Server with DynamoDB Integration
Reads user behavior from DynamoDB and provides personalized recommendations
"""

import os
import json
import random
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import boto3
from boto3.dynamodb.conditions import Key
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="MindFlow Recommendation API",
    description="AI-powered content recommendations for Buddylynk",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DynamoDB
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
behavior_table = dynamodb.Table('Buddylynk_UserBehavior')
embeddings_table = dynamodb.Table('Buddylynk_UserEmbeddings')
content_table = dynamodb.Table('Buddylynk_ContentFeatures')

# Action weights for scoring
ACTION_WEIGHTS = {
    0: 1.0,   # VIEW
    1: 3.0,   # LIKE - strong positive
    2: 5.0,   # SHARE - very strong positive
    3: 4.0,   # COMMENT - high engagement
    4: 4.0,   # SAVE - strong intent
    5: -0.5,  # SKIP - mild negative
    6: -2.0,  # UNLIKE - negative
    7: -2.0   # UNSAVE - negative
}


class UserRequest(BaseModel):
    userId: str
    limit: int = 20
    excludeIds: List[str] = []


class ContentScoreRequest(BaseModel):
    userId: str
    contentIds: List[str]


class RecommendationResponse(BaseModel):
    userId: str
    recommendations: List[Dict]
    algorithm: str = "mindflow-v1"
    timestamp: str


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "mindflow", "version": "1.0.0"}


@app.get("/api/recommendations/{user_id}")
async def get_recommendations(user_id: str, limit: int = 20):
    """Get personalized content recommendations for a user"""
    try:
        # 1. Get user behavior history (last 30 days)
        cutoff = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
        
        response = behavior_table.query(
            KeyConditionExpression=Key('userId').eq(user_id) & Key('timestamp').gt(cutoff),
            ScanIndexForward=False,
            Limit=100
        )
        
        user_history = response.get('Items', [])
        
        # 2. Analyze user preferences from behavior
        content_scores = {}
        action_counts = {i: 0 for i in range(8)}
        
        for item in user_history:
            content_id = item.get('contentId')
            action_type = int(item.get('actionType', 0))
            weight = ACTION_WEIGHTS.get(action_type, 1.0)
            
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
            
            # Build affinity scores for content
            if content_id not in content_scores:
                content_scores[content_id] = 0
            content_scores[content_id] += weight
        
        # 3. Calculate user engagement profile
        total_actions = sum(action_counts.values()) or 1
        engagement_score = sum(
            action_counts.get(i, 0) * ACTION_WEIGHTS.get(i, 0) 
            for i in range(8)
        ) / total_actions
        
        # 4. Get recently engaged content as positive signals
        liked_content = [
            cid for cid, score in content_scores.items() 
            if score > 2.0  # High positive engagement
        ]
        
        # 5. Build recommendation list
        # For now, use engagement scoring + randomization
        # TODO: Integrate neural network model
        
        recommendations = []
        
        # Mix strategy:
        # 60% - Similar to liked content (exploit)
        # 30% - Trending/popular content
        # 10% - Random exploration
        
        # Mark viewed content to avoid repeats
        viewed_content = set(content_scores.keys())
        
        # Return recommendations with scores
        recommendation_list = [
            {
                "contentId": f"rec_{i}",  # Placeholder - integrate with actual content
                "score": round(random.uniform(0.5, 1.0), 3),
                "reason": random.choice([
                    "based_on_likes", 
                    "trending", 
                    "similar_users", 
                    "discover"
                ])
            }
            for i in range(min(limit, 20))
        ]
        
        return {
            "userId": user_id,
            "recommendations": recommendation_list,
            "userProfile": {
                "engagementScore": round(engagement_score, 2),
                "totalActions": total_actions,
                "likedContent": len(liked_content),
                "viewedContent": len(viewed_content)
            },
            "algorithm": "mindflow-v1",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/score")
async def score_content(request: ContentScoreRequest):
    """Score specific content for a user"""
    try:
        user_id = request.userId
        content_ids = request.contentIds
        
        # Get user behavior
        cutoff = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
        
        response = behavior_table.query(
            KeyConditionExpression=Key('userId').eq(user_id) & Key('timestamp').gt(cutoff),
            Limit=50
        )
        
        user_history = response.get('Items', [])
        
        # Calculate content affinity
        liked_patterns = set()
        for item in user_history:
            action_type = int(item.get('actionType', 0))
            if action_type in [1, 2, 3, 4]:  # LIKE, SHARE, COMMENT, SAVE
                liked_patterns.add(item.get('contentId', '')[:8])  # Pattern prefix
        
        # Score each content
        scores = {}
        for cid in content_ids:
            base_score = 0.5  # Neutral
            
            # Boost if matches liked patterns
            if cid[:8] in liked_patterns:
                base_score += 0.3
            
            # Add some randomness for exploration
            base_score += random.uniform(-0.1, 0.1)
            
            scores[cid] = round(min(1.0, max(0.0, base_score)), 3)
        
        return {
            "userId": user_id,
            "scores": scores,
            "algorithm": "mindflow-v1"
        }
        
    except Exception as e:
        print(f"Error scoring content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get user's engagement profile and preferences"""
    try:
        # Get behavior history
        cutoff = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
        
        response = behavior_table.query(
            KeyConditionExpression=Key('userId').eq(user_id) & Key('timestamp').gt(cutoff),
            Limit=100
        )
        
        items = response.get('Items', [])
        
        # Analyze behavior
        action_counts = {}
        content_engagement = {}
        time_patterns = {"hours": {}, "days": {}}
        
        for item in items:
            action = int(item.get('actionType', 0))
            content_id = item.get('contentId', '')
            hour = int(item.get('hour', 12))
            day = int(item.get('dayOfWeek', 0))
            
            action_counts[action] = action_counts.get(action, 0) + 1
            
            if content_id not in content_engagement:
                content_engagement[content_id] = 0
            content_engagement[content_id] += ACTION_WEIGHTS.get(action, 1.0)
            
            time_patterns["hours"][hour] = time_patterns["hours"].get(hour, 0) + 1
            time_patterns["days"][day] = time_patterns["days"].get(day, 0) + 1
        
        # Find peak activity times
        peak_hour = max(time_patterns["hours"].items(), key=lambda x: x[1])[0] if time_patterns["hours"] else 12
        peak_day = max(time_patterns["days"].items(), key=lambda x: x[1])[0] if time_patterns["days"] else 0
        
        # Calculate engagement score
        total_score = sum(
            action_counts.get(action, 0) * weight 
            for action, weight in ACTION_WEIGHTS.items()
        )
        
        return {
            "userId": user_id,
            "profile": {
                "totalActions": sum(action_counts.values()),
                "engagementScore": round(total_score / max(sum(action_counts.values()), 1), 2),
                "actionBreakdown": action_counts,
                "topContent": sorted(
                    content_engagement.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5],
                "peakActivityHour": peak_hour,
                "peakActivityDay": peak_day
            },
            "lastUpdated": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get overall algorithm stats"""
    return {
        "service": "mindflow",
        "version": "1.0.0",
        "tables": {
            "behavior": "Buddylynk_UserBehavior",
            "embeddings": "Buddylynk_UserEmbeddings",
            "content": "Buddylynk_ContentFeatures"
        },
        "actionWeights": ACTION_WEIGHTS,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
