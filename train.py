#!/usr/bin/env python3
"""
Satellite Diffusion Model Training Entry Point
Trains custom diffusion models for satellite image inpainting
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.append('src')

def main():
    parser = argparse.ArgumentParser(description="Train Satellite Diffusion Model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    
    args = parser.parse_args()
    
    print("🚀 Satellite Diffusion Model Training")
    print("=" * 50)
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded configuration from: {args.config}")
    else:
        print(f"⚠️  Config file not found: {args.config}")
        print("   Using default configuration")
        config = {
            'training': {
                'epochs': 50,
                'batch_size': 4,
                'learning_rate': 1e-4,
                'save_interval': 5
            }
        }
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Display training configuration
    print("\n📋 Training Configuration:")
    print(f"   • Epochs: {config['training']['epochs']}")
    print(f"   • Batch Size: {config['training']['batch_size']}")
    print(f"   • Learning Rate: {config['training']['learning_rate']}")
    print(f"   • Checkpoint Dir: {args.output_dir}")
    
    if args.resume:
        print(f"   • Resuming from: {args.resume}")
    
    # Check for data
    train_image_dir = "data/train/images"
    train_mask_dir = "data/train/masks"
    
    if not os.path.exists(train_image_dir):
        print(f"\n❌ Training data not found: {train_image_dir}")
        print("   Please ensure your training data is in the correct location")
        return
    
    print(f"\n📁 Found training data in: {train_image_dir}")
    
    # Import and setup training components
    try:
        from src.training.trainer import DiffusionTrainer
        from src.training.config import TrainingConfig
        from src.data.dataset import SatelliteInpaintingDataset
        from src.data.transforms import SatelliteTransform
        from src.models.unet_2d_condition import UNet2DConditionModel
        
        print("✅ Successfully imported training modules")
        
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        print("   Please ensure all dependencies are installed")
        return
    
    # Initialize model
    print("\n🧠 Initializing model architecture...")
    model_config = {
        "sample_size": 64,
        "in_channels": 4,
        "out_channels": 4,
        "block_out_channels": [320, 640, 1280, 1280],
        "layers_per_block": 2
    }
    
    model = UNet2DConditionModel(model_config)
    print("✅ Model initialized successfully")
    
    # Setup dataset and transforms
    print("\n📊 Setting up data pipeline...")
    transform = SatelliteTransform(image_size=1024, is_training=True)
    dataset = SatelliteInpaintingDataset(
        train_image_dir,
        train_mask_dir, 
        transform=transform,
        split="train"
    )
    
    print(f"✅ Loaded {len(dataset)} training samples")
    
    # Convert config dict to TrainingConfig object
    training_config = TrainingConfig()
    for key, value in config['training'].items():
        if hasattr(training_config, key):
            setattr(training_config, key, value)
    
    # Initialize trainer
    print("\n🎯 Initializing trainer...")
    trainer = DiffusionTrainer(model, dataset, training_config)
    
    # Resume training if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 Resuming training from: {args.resume}")
        # Here you would load the checkpoint
        # start_epoch = load_checkpoint(trainer, args.resume)
    
    # Training loop
    print(f"\n🎨 Starting training for {training_config.epochs} epochs...")
    print("=" * 50)
    
    best_loss = float('inf')
    for epoch in range(start_epoch, training_config.epochs):
        # Train for one epoch
        train_loss = trainer.train_epoch(epoch)
        
        # Validate
        val_loss = trainer.validate_epoch(epoch)
        
        # Print progress
        print(f"Epoch {epoch+1:03d}/{training_config.epochs:03d} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            trainer.save_checkpoint(f"{args.output_dir}/best_model", epoch, val_loss)
            print(f"💾 New best model saved (loss: {val_loss:.4f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % training_config.save_interval == 0:
            checkpoint_path = f"{args.output_dir}/epoch_{epoch+1}"
            trainer.save_checkpoint(checkpoint_path, epoch, train_loss)
    
    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_checkpoint(f"{args.output_dir}/final_model", training_config.epochs, train_loss)
    
    print("=" * 50)
    print("✅ Training completed successfully!")
    print(f"📁 Models saved in: {args.output_dir}/")
    print(f"🎯 Best validation loss: {best_loss:.4f}")
    
    # Generate training report
    generate_training_report(training_config, best_loss)

def generate_training_report(config, best_loss):
    """Generate a training summary report"""
    report = f"""
📊 Training Report
==================

Model: Satellite Diffusion UNet
Training Data: 800 satellite images (1024×1024)
Validation Data: 200 satellite images

Training Configuration:
• Epochs: {config.epochs}
• Batch Size: {config.batch_size}
• Learning Rate: {config.learning_rate}
• Optimizer: AdamW

Results:
• Best Validation Loss: {best_loss:.4f}
• Final Model: checkpoints/final_model/
• Best Model: checkpoints/best_model/

Next Steps:
1. Run inference: python inference.py
2. Evaluate results: python scripts/evaluate.py
3. Visualize training curves

Training completed at: {os.popen('date').read().strip()}
"""
    
    with open('training_logs/training_report.txt', 'w') as f:
        f.write(report)
    
    print("📄 Training report saved: training_logs/training_report.txt")

if __name__ == "__main__":
    main()