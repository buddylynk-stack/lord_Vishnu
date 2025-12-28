"""
PyTorch Dataset for MindFlow
Handles data loading and batching for training.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
import numpy as np


class UserBehaviorDataset(Dataset):
    """
    PyTorch Dataset for user behavior sequences.
    Each sample contains a user's interaction history and target labels.
    """
    
    def __init__(
        self,
        user_ids: np.ndarray,
        content_ids: np.ndarray,
        action_types: np.ndarray,
        hours: np.ndarray,
        days: np.ndarray,
        engagement_labels: np.ndarray,
        click_labels: np.ndarray,
        watch_time_labels: np.ndarray,
        sequence_length: int = 20,
    ):
        """
        Args:
            user_ids: (num_samples,)
            content_ids: (num_samples, seq_len)
            action_types: (num_samples, seq_len)
            hours: (num_samples, seq_len)
            days: (num_samples, seq_len)
            engagement_labels: (num_samples,)
            click_labels: (num_samples,)
            watch_time_labels: (num_samples,)
            sequence_length: Maximum sequence length
        """
        self.user_ids = user_ids
        self.content_ids = content_ids
        self.action_types = action_types
        self.hours = hours
        self.days = days
        self.engagement_labels = engagement_labels
        self.click_labels = click_labels
        self.watch_time_labels = watch_time_labels
        self.sequence_length = sequence_length
        
        self.num_samples = len(user_ids)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        # Get sequence (pad if necessary)
        content_seq = self.content_ids[idx]
        action_seq = self.action_types[idx]
        hour_seq = self.hours[idx]
        day_seq = self.days[idx]
        
        # Ensure correct length
        seq_len = len(content_seq)
        if seq_len < self.sequence_length:
            # Pad sequences
            pad_len = self.sequence_length - seq_len
            content_seq = np.pad(content_seq, (0, pad_len), mode='constant')
            action_seq = np.pad(action_seq, (0, pad_len), mode='constant')
            hour_seq = np.pad(hour_seq, (0, pad_len), mode='constant')
            day_seq = np.pad(day_seq, (0, pad_len), mode='constant')
            mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)])
        else:
            # Truncate
            content_seq = content_seq[:self.sequence_length]
            action_seq = action_seq[:self.sequence_length]
            hour_seq = hour_seq[:self.sequence_length]
            day_seq = day_seq[:self.sequence_length]
            mask = np.ones(self.sequence_length)
        
        return {
            'user_id': torch.tensor(self.user_ids[idx], dtype=torch.long),
            'content_ids': torch.tensor(content_seq, dtype=torch.long),
            'action_types': torch.tensor(action_seq, dtype=torch.long),
            'hours': torch.tensor(hour_seq, dtype=torch.long),
            'days': torch.tensor(day_seq, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'engagement': torch.tensor(self.engagement_labels[idx], dtype=torch.float),
            'click': torch.tensor(self.click_labels[idx], dtype=torch.float),
            'watch_time': torch.tensor(self.watch_time_labels[idx], dtype=torch.float),
        }
    
    @classmethod
    def from_generated_data(
        cls,
        data: Dict[str, np.ndarray],
        sequence_length: int = 20,
    ) -> 'UserBehaviorDataset':
        """Create dataset from generated data dictionary."""
        return cls(
            user_ids=data['user_ids'],
            content_ids=data['content_ids'],
            action_types=data['action_types'],
            hours=data['hours'],
            days=data['days'],
            engagement_labels=data['engagement'],
            click_labels=data['click'],
            watch_time_labels=data['watch_time'],
            sequence_length=sequence_length,
        )
    
    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple:
        """Split dataset into train and validation sets."""
        np.random.seed(seed)
        
        indices = np.random.permutation(self.num_samples)
        train_size = int(self.num_samples * train_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]
        
        train_dataset = UserBehaviorDataset(
            user_ids=self.user_ids[train_idx],
            content_ids=self.content_ids[train_idx],
            action_types=self.action_types[train_idx],
            hours=self.hours[train_idx],
            days=self.days[train_idx],
            engagement_labels=self.engagement_labels[train_idx],
            click_labels=self.click_labels[train_idx],
            watch_time_labels=self.watch_time_labels[train_idx],
            sequence_length=self.sequence_length,
        )
        
        val_dataset = UserBehaviorDataset(
            user_ids=self.user_ids[val_idx],
            content_ids=self.content_ids[val_idx],
            action_types=self.action_types[val_idx],
            hours=self.hours[val_idx],
            days=self.days[val_idx],
            engagement_labels=self.engagement_labels[val_idx],
            click_labels=self.click_labels[val_idx],
            watch_time_labels=self.watch_time_labels[val_idx],
            sequence_length=self.sequence_length,
        )
        
        return train_dataset, val_dataset


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Stacks individual samples into batched tensors.
    """
    return {
        'user_id': torch.stack([item['user_id'] for item in batch]),
        'content_ids': torch.stack([item['content_ids'] for item in batch]),
        'action_types': torch.stack([item['action_types'] for item in batch]),
        'hours': torch.stack([item['hours'] for item in batch]),
        'days': torch.stack([item['days'] for item in batch]),
        'mask': torch.stack([item['mask'] for item in batch]),
        'engagement': torch.stack([item['engagement'] for item in batch]),
        'click': torch.stack([item['click'] for item in batch]),
        'watch_time': torch.stack([item['watch_time'] for item in batch]),
    }
