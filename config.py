"""
Configuration file for training parameters
Edit these values to customize your training
"""

class Config:
    # Dataset paths
    DATA_ROOT = "./data"  # Path to dataset root directory
    
    # Image parameters
    IMG_HEIGHT = 32       # Height to resize images to
    IMG_WIDTH = 128       # Width to resize images to
    IMG_CHANNELS = 1      # 1 for grayscale, 3 for RGB
    
    # Training parameters
    BATCH_SIZE = 32       # Batch size (reduce if GPU memory is limited)
    NUM_EPOCHS = 100      # Number of training epochs
    LEARNING_RATE = 0.001 # Initial learning rate
    
    # Model parameters
    HIDDEN_SIZE = 256     # LSTM hidden size
    NUM_WORKERS = 4       # Number of data loading workers (0 for Windows)
    
    # Optimization
    WEIGHT_DECAY = 1e-5   # L2 regularization
    GRADIENT_CLIP = 5.0   # Gradient clipping threshold
    
    # Learning rate scheduler
    LR_SCHEDULER = 'ReduceLROnPlateau'  # Options: 'ReduceLROnPlateau', 'CosineAnnealing'
    LR_PATIENCE = 5       # Patience for ReduceLROnPlateau
    LR_FACTOR = 0.5       # Factor to reduce LR
    
    # Checkpointing
    SAVE_DIR = 'checkpoints'
    SAVE_FREQUENCY = 10   # Save checkpoint every N epochs
    
    # Data augmentation (optional, to be implemented)
    USE_AUGMENTATION = False
    
    # Mixed precision training (for RTX cards)
    USE_AMP = False       # Automatic Mixed Precision
    
    # Miscellaneous
    SEED = 42             # Random seed for reproducibility
    PIN_MEMORY = True     # Pin memory for faster data transfer to GPU
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("\n" + "=" * 60)
        print("Configuration Parameters")
        print("=" * 60)
        
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key:20s}: {value}")
        
        print("=" * 60 + "\n")


# GTX 1070 optimized config (8GB VRAM)
class GTX1070Config(Config):
    BATCH_SIZE = 24
    NUM_WORKERS = 4
    HIDDEN_SIZE = 256
    USE_AMP = False  # GTX 1070 doesn't support tensor cores


# High memory GPU config (16GB+ VRAM)
class HighMemoryConfig(Config):
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    HIDDEN_SIZE = 512


# CPU training config (not recommended)
class CPUConfig(Config):
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    HIDDEN_SIZE = 128


# Quick test config (for debugging)
class TestConfig(Config):
    BATCH_SIZE = 4
    NUM_EPOCHS = 2
    NUM_WORKERS = 0
    SAVE_FREQUENCY = 1


if __name__ == "__main__":
    print("Available configurations:")
    print("1. Config (Default)")
    print("2. GTX1070Config (Optimized for GTX 1070)")
    print("3. HighMemoryConfig (For 16GB+ VRAM)")
    print("4. CPUConfig (CPU training, slow)")
    print("5. TestConfig (Quick test/debug)")
    
    print("\n" + "=" * 60)
    print("Current Default Configuration:")
    Config.print_config()
    
    print("\nGTX 1070 Optimized Configuration:")
    GTX1070Config.print_config()