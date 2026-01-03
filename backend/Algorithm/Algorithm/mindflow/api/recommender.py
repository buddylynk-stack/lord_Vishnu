"""
MindFlow Recommender API
High-level interface for getting recommendations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque


class UserState:
    """Tracks a user's recent interactions for context."""
    
    def __init__(self, user_id: int, max_history: int = 20):
        self.user_id = user_id
        self.max_history = max_history
        
        # Recent interaction history
        self.content_ids: deque = deque(maxlen=max_history)
        self.action_types: deque = deque(maxlen=max_history)
        self.hours: deque = deque(maxlen=max_history)
        self.days: deque = deque(maxlen=max_history)
    
    def add_interaction(
        self,
        content_id: int,
        action_type: int,
        hour: int,
        day: int,
    ):
        """Add a new interaction to history."""
        self.content_ids.append(content_id)
        self.action_types.append(action_type)
        self.hours.append(hour)
        self.days.append(day)
    
    def get_history(self, pad_length: int = 20) -> Dict[str, np.ndarray]:
        """Get history as numpy arrays, padded to specified length."""
        def pad_sequence(seq: deque, length: int) -> np.ndarray:
            arr = np.zeros(length, dtype=np.int64)
            seq_list = list(seq)
            if seq_list:
                start = max(0, length - len(seq_list))
                arr[start:] = seq_list[-(length - start):]
            return arr
        
        return {
            'content_ids': pad_sequence(self.content_ids, pad_length),
            'action_types': pad_sequence(self.action_types, pad_length),
            'hours': pad_sequence(self.hours, pad_length),
            'days': pad_sequence(self.days, pad_length),
        }
    
    def has_history(self) -> bool:
        """Check if user has any interaction history."""
        return len(self.content_ids) > 0


class Recommender:
    """
    High-level recommendation interface.
    
    Usage:
        recommender = Recommender("mindflow.onnx")
        
        # Record user actions
        recommender.record_action(user_id=1, content_id=100, action="like")
        
        # Get recommendations
        recs = recommender.get_recommendations(user_id=1, candidate_ids=[...])
    """
    
    # Action type mapping
    ACTION_MAP = {
        'view': 0,
        'like': 1,
        'share': 2,
        'comment': 3,
        'save': 4,
        'skip': 5,
        'hide': 6,
        'report': 7,
        'follow': 8,
        'unfollow': 9,
    }
    
    def __init__(
        self,
        model_path: str,
        sequence_length: int = 20,
        num_threads: int = 4,
    ):
        """
        Initialize recommender with ONNX model.
        
        Args:
            model_path: Path to .onnx model file
            sequence_length: Length of context sequence
            num_threads: CPU threads for inference
        """
        from ..onnx.onnx_inference import ONNXInferenceEngine
        
        self.engine = ONNXInferenceEngine(
            model_path,
            num_threads=num_threads,
        )
        self.sequence_length = sequence_length
        
        # User state cache
        self.user_states: Dict[int, UserState] = {}
    
    def _get_user_state(self, user_id: int) -> UserState:
        """Get or create user state."""
        if user_id not in self.user_states:
            self.user_states[user_id] = UserState(
                user_id, 
                max_history=self.sequence_length
            )
        return self.user_states[user_id]
    
    def record_action(
        self,
        user_id: int,
        content_id: int,
        action: str,
        hour: Optional[int] = None,
        day: Optional[int] = None,
    ):
        """
        Record a user action for context building.
        
        Args:
            user_id: User ID
            content_id: Content ID that was interacted with
            action: Action type ('like', 'share', 'view', etc.)
            hour: Hour of day (0-23), auto-detected if None
            day: Day of week (0-6), auto-detected if None
        """
        import datetime
        
        if hour is None:
            hour = datetime.datetime.now().hour
        if day is None:
            day = datetime.datetime.now().weekday()
        
        action_type = self.ACTION_MAP.get(action.lower(), 0)
        
        state = self._get_user_state(user_id)
        state.add_interaction(content_id, action_type, hour, day)
    
    def predict_engagement(
        self,
        user_id: int,
        content_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Predict engagement metrics for a user.
        
        Args:
            user_id: User ID
            content_id: Optional content ID to add temporarily
        
        Returns:
            Dictionary with engagement, click_prob, watch_time
        """
        state = self._get_user_state(user_id)
        
        if not state.has_history():
            # Cold start - return neutral predictions
            return {
                'engagement': 0.5,
                'click_prob': 0.5,
                'watch_time': 30.0,
            }
        
        history = state.get_history(self.sequence_length)
        
        return self.engine.predict_single(
            user_id=user_id,
            content_ids=history['content_ids'].tolist(),
            action_types=history['action_types'].tolist(),
            hours=history['hours'].tolist(),
            days=history['days'].tolist(),
        )
    
    def score_content(
        self,
        user_id: int,
        content_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """
        Score multiple content items for a user.
        
        Args:
            user_id: User ID
            content_ids: List of content IDs to score
        
        Returns:
            List of (content_id, score) tuples, sorted by score descending
        """
        state = self._get_user_state(user_id)
        history = state.get_history(self.sequence_length)
        
        scores = []
        for content_id in content_ids:
            # Simulate adding this content to history
            temp_content = history['content_ids'].copy()
            temp_content[-1] = content_id
            
            result = self.engine.predict_single(
                user_id=user_id,
                content_ids=temp_content.tolist(),
                action_types=history['action_types'].tolist(),
                hours=history['hours'].tolist(),
                days=history['days'].tolist(),
            )
            
            # Combine metrics into single score
            score = (
                result['engagement'] * 0.4 +
                result['click_prob'] * 0.4 +
                min(result['watch_time'] / 60.0, 1.0) * 0.2
            )
            scores.append((content_id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def get_recommendations(
        self,
        user_id: int,
        candidate_ids: List[int],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Get top-K recommendations from candidates.
        
        Args:
            user_id: User ID
            candidate_ids: Pool of candidate content IDs
            top_k: Number of recommendations to return
        
        Returns:
            List of recommendation dictionaries with content_id and score
        """
        scored = self.score_content(user_id, candidate_ids)
        
        recommendations = []
        for content_id, score in scored[:top_k]:
            recommendations.append({
                'content_id': content_id,
                'score': score,
                'rank': len(recommendations) + 1,
            })
        
        return recommendations
    
    def clear_user_history(self, user_id: int):
        """Clear a user's interaction history."""
        if user_id in self.user_states:
            del self.user_states[user_id]
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get statistics about a user's engagement."""
        state = self._get_user_state(user_id)
        
        engagement = self.predict_engagement(user_id)
        
        return {
            'user_id': user_id,
            'history_length': len(state.content_ids),
            'predicted_engagement': engagement['engagement'],
            'predicted_click_prob': engagement['click_prob'],
            'predicted_watch_time': engagement['watch_time'],
        }
