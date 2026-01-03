#!/usr/bin/env python3
"""
MindFlow Production Training Script
Train the recommendation model for production deployment.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mindflow.models.behavior_model import MindFlowModel
from mindflow.training.data_generator import SyntheticDataGenerator
from mindflow.training.dataset import UserBehaviorDataset
from mindflow.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="🧠 Train MindFlow Recommendation Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                              # Train with defaults
  python train.py --epochs 100 --lr 0.0005     # Custom training
  python train.py --num-samples 100000         # Large scale training
        """
    )
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    
    # Data arguments
    parser.add_argument('--num-users', type=int, default=10000, help='Number of users')
    parser.add_argument('--num-contents', type=int, default=50000, help='Number of contents')
    parser.add_argument('--num-samples', type=int, default=100000, help='Training samples')
    parser.add_argument('--seq-length', type=int, default=20, help='Sequence length')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Attention layers')
    parser.add_argument('--embedding-dim', type=int, default=64, help='Embedding dimension')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')
    parser.add_argument('--save-every', type=int, default=10, help='Save every N epochs')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 MindFlow Production Training")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate training data
    print("\n📊 Generating training data...")
    generator = SyntheticDataGenerator(
        num_users=args.num_users,
        num_contents=args.num_contents,
    )
    
    data = generator.generate_dataset(
        num_samples=args.num_samples,
        sequence_length=args.seq_length,
    )
    
    print(f"   Users: {args.num_users:,}")
    print(f"   Contents: {args.num_contents:,}")
    print(f"   Samples: {len(data['user_ids']):,}")
    
    # Create dataset
    dataset = UserBehaviorDataset.from_generated_data(
        data, 
        sequence_length=args.seq_length
    )
    
    # Split into train/val
    train_dataset, val_dataset = dataset.split(train_ratio=0.8)
    print(f"   Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
    
    # Create model
    print("\n🔧 Creating model...")
    model = MindFlowModel(
        num_users=args.num_users + 1,
        num_contents=args.num_contents + 1,
        num_action_types=10,
        user_embedding_dim=args.embedding_dim,
        content_embedding_dim=args.embedding_dim,
        action_embedding_dim=32,
        time_embedding_dim=16,
        hidden_dim=args.hidden_dim,
        num_attention_heads=args.num_heads,
        num_attention_layers=args.num_layers,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    # Create trainer
    print("\n🚀 Starting training...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.output_dir,
        device='cpu',
    )
    
    # Train
    print("-" * 60)
    trainer.train(
        epochs=args.epochs,
        save_every=args.save_every,
    )
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"   Model saved to: {args.output_dir}/best_model.pt")
    print("\n📦 Export to ONNX:")
    print(f"   python export_onnx.py --checkpoint {args.output_dir}/best_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
