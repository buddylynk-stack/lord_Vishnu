"""
Synthetic Data Generator for MindFlow
Generates realistic user behavior patterns for training and testing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class UserProfile:
    """Simulated user profile with preferences."""
    user_id: int
    preferred_categories: List[int]
    activity_hours: List[int]  # Preferred hours of activity
    engagement_rate: float  # Base engagement probability
    session_length: int  # Average actions per session


@dataclass
class ContentItem:
    """Simulated content item."""
    content_id: int
    category: int
    quality_score: float
    virality_score: float


class SyntheticDataGenerator:
    """
    Generates synthetic user behavior data for training.
    
    Simulates realistic patterns:
    - Users have preferences for certain content categories
    - Engagement depends on content quality and user-content match
    - Time-of-day affects behavior
    - Users have different activity levels
    """
    
    def __init__(
        self,
        num_users: int = 1000,
        num_contents: int = 5000,
        num_categories: int = 20,
        num_action_types: int = 10,
        seed: int = 42,
    ):
        self.num_users = num_users
        self.num_contents = num_contents
        self.num_categories = num_categories
        self.num_action_types = num_action_types
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Action type weights (some actions are more common)
        self.action_weights = {
            0: 0.50,  # view
            1: 0.15,  # like
            2: 0.05,  # share
            3: 0.08,  # comment
            4: 0.07,  # save
            5: 0.08,  # skip
            6: 0.03,  # hide
            7: 0.01,  # report
            8: 0.02,  # follow
            9: 0.01,  # unfollow
        }
        
        # Generate users and contents
        self.users = self._generate_users()
        self.contents = self._generate_contents()
        
        # Category to content mapping for efficient lookup
        self.category_to_contents: Dict[int, List[int]] = {}
        for content in self.contents:
            if content.category not in self.category_to_contents:
                self.category_to_contents[content.category] = []
            self.category_to_contents[content.category].append(content.content_id)
    
    def _generate_users(self) -> List[UserProfile]:
        """Generate user profiles with diverse preferences."""
        users = []
        
        for user_id in range(1, self.num_users + 1):
            # Random number of preferred categories (1-5)
            num_preferred = np.random.randint(1, 6)
            preferred_categories = np.random.choice(
                self.num_categories, num_preferred, replace=False
            ).tolist()
            
            # Activity hours (model different user types)
            user_type = np.random.choice(['morning', 'evening', 'night', 'all_day'])
            if user_type == 'morning':
                activity_hours = list(range(6, 12))
            elif user_type == 'evening':
                activity_hours = list(range(17, 23))
            elif user_type == 'night':
                activity_hours = list(range(21, 24)) + list(range(0, 3))
            else:
                activity_hours = list(range(8, 23))
            
            # Base engagement rate (some users are more engaged)
            engagement_rate = np.random.beta(2, 5)  # Skewed towards lower engagement
            
            # Session length
            session_length = np.random.randint(5, 30)
            
            users.append(UserProfile(
                user_id=user_id,
                preferred_categories=preferred_categories,
                activity_hours=activity_hours,
                engagement_rate=engagement_rate,
                session_length=session_length,
            ))
        
        return users
    
    def _generate_contents(self) -> List[ContentItem]:
        """Generate content items with varying quality."""
        contents = []
        
        for content_id in range(1, self.num_contents + 1):
            category = np.random.randint(0, self.num_categories)
            quality_score = np.random.beta(2, 2)  # Bell-curved quality
            virality_score = np.random.beta(1, 5)  # Most content is not viral
            
            contents.append(ContentItem(
                content_id=content_id,
                category=category,
                quality_score=quality_score,
                virality_score=virality_score,
            ))
        
        return contents
    
    def _get_action_for_engagement(
        self,
        engagement_score: float,
        content: ContentItem,
    ) -> int:
        """Determine action based on engagement level."""
        # High engagement -> positive actions
        if engagement_score > 0.8:
            actions = [1, 2, 3, 4, 8]  # like, share, comment, save, follow
            weights = [0.4, 0.15, 0.2, 0.15, 0.1]
        elif engagement_score > 0.5:
            actions = [0, 1, 4]  # view, like, save
            weights = [0.5, 0.35, 0.15]
        elif engagement_score > 0.2:
            actions = [0, 5]  # view, skip
            weights = [0.7, 0.3]
        else:
            actions = [5, 6, 7]  # skip, hide, report
            weights = [0.7, 0.25, 0.05]
        
        return np.random.choice(actions, p=weights)
    
    def _calculate_engagement(
        self,
        user: UserProfile,
        content: ContentItem,
        hour: int,
    ) -> float:
        """Calculate engagement probability for user-content-time tuple."""
        # Base engagement from user profile
        engagement = user.engagement_rate
        
        # Boost if content matches user preferences
        if content.category in user.preferred_categories:
            engagement *= 1.5
        
        # Boost from content quality
        engagement *= (0.5 + content.quality_score)
        
        # Viral content has higher engagement
        engagement *= (1 + content.virality_score * 2)
        
        # Time-of-day effect
        if hour in user.activity_hours:
            engagement *= 1.2
        else:
            engagement *= 0.7
        
        # Add some noise
        engagement *= np.random.uniform(0.8, 1.2)
        
        return min(engagement, 1.0)
    
    def generate_user_session(
        self,
        user_id: int,
        session_length: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate a single user session with interactions."""
        user = self.users[user_id - 1]
        
        if session_length is None:
            session_length = user.session_length + np.random.randint(-5, 6)
            session_length = max(5, session_length)
        
        # Pick a starting hour
        hour = np.random.choice(user.activity_hours) if user.activity_hours else np.random.randint(0, 24)
        day = np.random.randint(0, 7)
        
        # Generate interactions
        content_ids = []
        action_types = []
        hours = []
        days = []
        engagement_scores = []
        click_labels = []
        watch_times = []
        
        for i in range(session_length):
            # Sample content (biased towards preferred categories)
            if np.random.random() < 0.7 and user.preferred_categories:
                # 70% chance to see preferred category
                category = np.random.choice(user.preferred_categories)
                available = self.category_to_contents.get(category, [])
                if available:
                    content_id = np.random.choice(available)
                else:
                    content_id = np.random.randint(1, self.num_contents + 1)
            else:
                content_id = np.random.randint(1, self.num_contents + 1)
            
            content = self.contents[content_id - 1]
            
            # Calculate engagement
            engagement = self._calculate_engagement(user, content, hour)
            
            # Determine action
            action = self._get_action_for_engagement(engagement, content)
            
            # Labels
            clicked = 1 if action in [1, 2, 3, 4, 8] else 0  # Positive actions
            watch_time = engagement * np.random.uniform(5, 60) if clicked else np.random.uniform(0, 5)
            
            content_ids.append(content_id)
            action_types.append(action)
            hours.append(hour)
            days.append(day)
            engagement_scores.append(engagement)
            click_labels.append(clicked)
            watch_times.append(watch_time)
            
            # Time passes
            if np.random.random() < 0.1:
                hour = (hour + 1) % 24
        
        return {
            'user_id': np.array([user_id]),
            'content_ids': np.array(content_ids),
            'action_types': np.array(action_types),
            'hours': np.array(hours),
            'days': np.array(days),
            'engagement_scores': np.array(engagement_scores, dtype=np.float32),
            'click_labels': np.array(click_labels, dtype=np.float32),
            'watch_times': np.array(watch_times, dtype=np.float32),
        }
    
    def generate_dataset(
        self,
        num_samples: int = 10000,
        sequence_length: int = 20,
    ) -> Dict[str, np.ndarray]:
        """Generate full training dataset."""
        all_user_ids = []
        all_content_ids = []
        all_action_types = []
        all_hours = []
        all_days = []
        all_engagement = []
        all_click = []
        all_watch_time = []
        
        samples_per_user = max(1, num_samples // self.num_users)
        
        for user in self.users:
            for _ in range(samples_per_user):
                session = self.generate_user_session(
                    user.user_id, 
                    session_length=sequence_length
                )
                
                all_user_ids.append(session['user_id'][0])
                all_content_ids.append(session['content_ids'][:sequence_length])
                all_action_types.append(session['action_types'][:sequence_length])
                all_hours.append(session['hours'][:sequence_length])
                all_days.append(session['days'][:sequence_length])
                all_engagement.append(session['engagement_scores'].mean())
                all_click.append(session['click_labels'].mean())
                all_watch_time.append(session['watch_times'].mean())
        
        # Convert to arrays
        return {
            'user_ids': np.array(all_user_ids),
            'content_ids': np.array(all_content_ids),
            'action_types': np.array(all_action_types),
            'hours': np.array(all_hours),
            'days': np.array(all_days),
            'engagement': np.array(all_engagement, dtype=np.float32),
            'click': np.array(all_click, dtype=np.float32),
            'watch_time': np.array(all_watch_time, dtype=np.float32),
        }
    
    def get_candidate_contents(
        self,
        user_id: int,
        num_candidates: int = 100,
    ) -> np.ndarray:
        """Get candidate content IDs for recommendation."""
        user = self.users[user_id - 1]
        
        candidates = []
        
        # Include some from preferred categories
        for cat in user.preferred_categories:
            cat_contents = self.category_to_contents.get(cat, [])
            n_from_cat = min(len(cat_contents), num_candidates // len(user.preferred_categories))
            candidates.extend(np.random.choice(cat_contents, n_from_cat, replace=False).tolist())
        
        # Fill rest randomly
        remaining = num_candidates - len(candidates)
        if remaining > 0:
            random_contents = np.random.choice(
                self.num_contents, remaining, replace=False
            ) + 1
            candidates.extend(random_contents.tolist())
        
        return np.array(candidates[:num_candidates])
