"""
Training script with configuration support
Usage:
    python train_with_config.py --config GTX1070Config
    python train_with_config.py --config HighMemoryConfig
    python train_with_config.py  # uses default Config
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import random

from model import create_model
from dataset import create_dataloaders
from config import Config, GTX1070Config, HighMemoryConfig, CPUConfig, TestConfig


def levenshtein_distance(s1, s2):
    """
    Pure Python implementation of Levenshtein distance
    This replaces editdistance package
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


class CTCTrainer:
    """
    Trainer class for CTC-based text recognition
    """
    def __init__(self, model, train_loader, val_loader, test_loader, dataset,
                 device, save_dir='checkpoints', learning_rate=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dataset = dataset
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler (removed verbose parameter for PyTorch 2.0+)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_accuracy = 0.0
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, targets, target_lengths, _) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # CTC Loss expects: (T, N, C) where T=time_steps, N=batch, C=classes
            outputs = outputs.permute(1, 0, 2)  # (width, batch, classes)
            
            # Input lengths (width of the output sequence)
            input_lengths = torch.full(
                size=(outputs.size(1),),
                fill_value=outputs.size(0),
                dtype=torch.long,
                device=self.device
            )
            
            # Calculate loss
            loss = self.ctc_loss(outputs.log_softmax(2), targets, input_lengths, target_lengths)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        total_cer = 0  # Character Error Rate
        
        with torch.no_grad():
            for images, targets, target_lengths, texts in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # CTC Loss
                outputs_ctc = outputs.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(outputs_ctc.size(1),),
                    fill_value=outputs_ctc.size(0),
                    dtype=torch.long,
                    device=self.device
                )
                
                loss = self.ctc_loss(outputs_ctc.log_softmax(2), targets, input_lengths, target_lengths)
                total_loss += loss.item()
                
                # Decode predictions
                predictions = self.decode_predictions(outputs)
                
                # Calculate accuracy
                for pred, true_text in zip(predictions, texts):
                    if pred == true_text:
                        correct += 1
                    total += 1
                    
                    # Calculate CER using pure Python Levenshtein distance
                    cer = levenshtein_distance(pred, true_text) / max(len(true_text), 1)
                    total_cer += cer
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0
        avg_cer = total_cer / total if total > 0 else 0
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy, avg_cer
    
    def decode_predictions(self, outputs):
        """
        Decode model outputs using greedy decoding
        """
        # outputs: (batch, width, num_classes)
        predictions = []
        
        # Get argmax predictions
        _, preds = outputs.max(2)  # (batch, width)
        preds = preds.cpu().numpy()
        
        for pred in preds:
            # Remove consecutive duplicates and blanks
            decoded = []
            prev = -1
            for p in pred:
                if p != prev and p != 0:  # 0 is CTC blank
                    decoded.append(p)
                prev = p
            
            # Convert to text
            text = self.dataset.decode_text(decoded)
            predictions.append(text)
        
        return predictions
    
    def train(self, num_epochs):
        """
        Train the model for specified number of epochs
        """
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, accuracy, cer = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Accuracy: {accuracy*100:.2f}%")
            print(f"CER: {cer:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, 'best_loss.pth')
                print(f"✓ Saved best model (loss)")
            
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.save_checkpoint(epoch, 'best_accuracy.pth')
                print(f"✓ Saved best model (accuracy)")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')
        
        training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {training_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best accuracy: {self.best_accuracy*100:.2f}%")
        print(f"{'='*60}")
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'best_accuracy': self.best_accuracy,
        }
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_accuracy = checkpoint['best_accuracy']
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch']
    
    def test(self):
        """Test the model on test set"""
        print("\nTesting model...")
        self.model.eval()
        
        correct = 0
        total = 0
        total_cer = 0
        
        predictions_list = []
        ground_truth_list = []
        
        with torch.no_grad():
            for images, targets, target_lengths, texts in tqdm(self.test_loader, desc='Testing'):
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Decode predictions
                predictions = self.decode_predictions(outputs)
                
                # Calculate metrics
                for pred, true_text in zip(predictions, texts):
                    predictions_list.append(pred)
                    ground_truth_list.append(true_text)
                    
                    if pred == true_text:
                        correct += 1
                    total += 1
                    
                    cer = levenshtein_distance(pred, true_text) / max(len(true_text), 1)
                    total_cer += cer
        
        accuracy = correct / total if total > 0 else 0
        avg_cer = total_cer / total if total > 0 else 0
        
        print(f"\nTest Results:")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"CER: {avg_cer:.4f}")
        print(f"\nSample Predictions:")
        for i in range(min(10, len(predictions_list))):
            print(f"True: {ground_truth_list[i]}")
            print(f"Pred: {predictions_list[i]}")
            print()
        
        return accuracy, avg_cer


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_config(config_name):
    """Get configuration class by name"""
    configs = {
        'Config': Config,
        'GTX1070Config': GTX1070Config,
        'HighMemoryConfig': HighMemoryConfig,
        'CPUConfig': CPUConfig,
        'TestConfig': TestConfig
    }
    return configs.get(config_name, Config)


def main():
    parser = argparse.ArgumentParser(description='Train Prescription Recognition Model')
    parser.add_argument('--config', type=str, default='GTX1070Config',
                       help='Configuration to use (Config, GTX1070Config, HighMemoryConfig, CPUConfig, TestConfig)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--data_root', type=str, default=None,
                       help='Override dataset root path')
    
    args = parser.parse_args()
    
    # Get configuration
    ConfigClass = get_config(args.config)
    cfg = ConfigClass()
    
    # Override data root if provided
    if args.data_root:
        cfg.DATA_ROOT = args.data_root
    
    # Print configuration
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  Prescription Recognition Training                       ║")
    print("╚" + "═" * 58 + "╝")
    cfg.print_config()
    
    # Set random seed
    set_seed(cfg.SEED)
    print(f"Random seed set to {cfg.SEED}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create dataloaders
    print("\nLoading dataset...")
    try:
        train_loader, val_loader, test_loader, num_classes, train_dataset = create_dataloaders(
            root_dir=cfg.DATA_ROOT,
            batch_size=cfg.BATCH_SIZE,
            img_height=cfg.IMG_HEIGHT,
            img_width=cfg.IMG_WIDTH,
            num_workers=cfg.NUM_WORKERS
        )
        print(f"✓ Dataset loaded successfully")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Testing samples: {len(test_loader.dataset)}")
        print(f"  Number of classes: {num_classes}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("Please check:")
        print(f"  1. Dataset path: {cfg.DATA_ROOT}")
        print("  2. Dataset structure (see README.md)")
        print("  3. CSV file format")
        return
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        img_height=cfg.IMG_HEIGHT,
        img_channels=cfg.IMG_CHANNELS,
        num_classes=num_classes,
        hidden_size=cfg.HIDDEN_SIZE
    )
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (32-bit)")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = CTCTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset=train_dataset,
        device=device,
        save_dir=cfg.SAVE_DIR,
        learning_rate=cfg.LEARNING_RATE
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    try:
        trainer.train(num_epochs=cfg.NUM_EPOCHS)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Training interrupted by user")
        print("=" * 60)
        save_path = os.path.join(cfg.SAVE_DIR, 'interrupted_checkpoint.pth')
        trainer.save_checkpoint(len(trainer.train_losses), 'interrupted_checkpoint.pth')
        print(f"Checkpoint saved to {save_path}")
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test model
    print("\n" + "=" * 60)
    print("Testing model on test set...")
    print("=" * 60)
    try:
        trainer.test()
    except Exception as e:
        print(f"✗ Testing error: {e}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved in: {cfg.SAVE_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()