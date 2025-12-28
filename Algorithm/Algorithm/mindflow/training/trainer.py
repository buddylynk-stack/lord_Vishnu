"""
MindFlow Trainer
CPU-optimized training loop with logging and checkpointing.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Optional, Callable
from tqdm import tqdm

from ..models.behavior_model import MindFlowModel
from ..utils import MetricsTracker, EarlyStopping, ensure_dir, save_json, format_time
from .dataset import UserBehaviorDataset, collate_fn


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for MindFlow training.
    Combines losses for engagement, click, and watch time prediction.
    """
    
    def __init__(
        self,
        engagement_weight: float = 1.0,
        click_weight: float = 1.0,
        watch_time_weight: float = 0.5,
    ):
        super().__init__()
        
        self.engagement_weight = engagement_weight
        self.click_weight = click_weight
        self.watch_time_weight = watch_time_weight
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-task loss.
        
        Args:
            predictions: Model predictions dict
            targets: Target labels dict
        
        Returns:
            Dictionary with total loss and individual losses
        """
        # Engagement loss (MSE)
        engagement_loss = self.mse_loss(
            predictions['engagement'],
            targets['engagement']
        )
        
        # Click loss (BCE)
        click_loss = self.bce_loss(
            predictions['click_prob'],
            targets['click']
        )
        
        # Watch time loss (MSE)
        watch_time_loss = self.mse_loss(
            predictions['watch_time'],
            targets['watch_time']
        )
        
        # Weighted total loss
        total_loss = (
            self.engagement_weight * engagement_loss +
            self.click_weight * click_loss +
            self.watch_time_weight * watch_time_loss
        )
        
        return {
            'total': total_loss,
            'engagement': engagement_loss,
            'click': click_loss,
            'watch_time': watch_time_loss,
        }


class Trainer:
    """
    Training manager for MindFlow model.
    Handles training loop, validation, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model: MindFlowModel,
        train_dataset: UserBehaviorDataset,
        val_dataset: Optional[UserBehaviorDataset] = None,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.max_grad_norm = max_grad_norm
        
        ensure_dir(checkpoint_dir)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # CPU training
            pin_memory=False,
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        # Loss function
        self.criterion = MultiTaskLoss()
        
        # Tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        self.history = {'train': [], 'val': []}
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}",
            leave=True,
        )
        
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(
                user_ids=batch['user_id'],
                content_ids=batch['content_ids'],
                action_types=batch['action_types'],
                hours=batch['hours'],
                days=batch['days'],
                mask=batch['mask'],
            )
            
            # Calculate loss
            targets = {
                'engagement': batch['engagement'],
                'click': batch['click'],
                'watch_time': batch['watch_time'],
            }
            losses = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            self.optimizer.step()
            
            # Track metrics
            self.train_metrics.update({
                'loss': losses['total'].item(),
                'engagement_loss': losses['engagement'].item(),
                'click_loss': losses['click'].item(),
                'watch_time_loss': losses['watch_time'].item(),
            })
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
            })
        
        return self.train_metrics.compute()
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.val_metrics.reset()
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            predictions = self.model(
                user_ids=batch['user_id'],
                content_ids=batch['content_ids'],
                action_types=batch['action_types'],
                hours=batch['hours'],
                days=batch['days'],
                mask=batch['mask'],
            )
            
            targets = {
                'engagement': batch['engagement'],
                'click': batch['click'],
                'watch_time': batch['watch_time'],
            }
            losses = self.criterion(predictions, targets)
            
            self.val_metrics.update({
                'loss': losses['total'].item(),
                'engagement_loss': losses['engagement'].item(),
                'click_loss': losses['click'].item(),
                'watch_time_loss': losses['watch_time'].item(),
            })
        
        return self.val_metrics.compute()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.model.config,
        }
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint: {path}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"⭐ Saved best model: {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"📂 Loaded checkpoint: {path}")
    
    def train(
        self,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_every: int = 5,
        callback: Optional[Callable] = None,
    ) -> Dict[str, list]:
        """
        Run full training loop.
        
        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_every: Save checkpoint every N epochs
            callback: Optional callback function(epoch, train_metrics, val_metrics)
        
        Returns:
            Training history
        """
        print(f"🚀 Starting training for {epochs} epochs")
        print(f"📊 Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        print(f"💻 Device: {self.device}")
        print("-" * 50)
        
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self._train_epoch(epoch)
            self.history['train'].append(train_metrics)
            
            # Validation
            val_metrics = self._validate()
            if val_metrics:
                self.history['val'].append(val_metrics)
            
            # Learning rate scheduling
            val_loss = val_metrics.get('loss', train_metrics['loss'])
            self.scheduler.step(val_loss)
            
            # Logging
            epoch_time = time.time() - epoch_start
            print(f"\n📈 Epoch {epoch + 1}/{epochs} ({format_time(epoch_time)})")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                print(f"   Val Loss:   {val_metrics['loss']:.4f}")
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"   🎉 New best validation loss!")
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'model_epoch_{epoch + 1}.pt', is_best)
            elif is_best:
                self.save_checkpoint('best_model.pt', is_best=True)
            
            # Callback
            if callback:
                callback(epoch, train_metrics, val_metrics)
            
            # Early stopping
            if early_stopping(val_loss):
                print(f"\n⏹️  Early stopping at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        print(f"\n✅ Training complete in {format_time(total_time)}")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final history
        save_json(self.history, os.path.join(self.checkpoint_dir, 'training_history.json'))
        
        return self.history
