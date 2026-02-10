"""
Quick Start Guide for Handwritten Prescription Recognition
============================================================

This script helps you get started with the project.
"""

import torch
import os


def check_environment():
    """Check if the environment is properly set up"""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    
    # Check Python version
    import sys
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # Check PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: Yes")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test CUDA
        try:
            x = torch.randn(100, 100).cuda()
            y = x @ x
            print(f"✓ CUDA test: Passed")
        except Exception as e:
            print(f"✗ CUDA test: Failed - {e}")
    else:
        print(f"✗ CUDA available: No")
        print(f"  Warning: Training will be slow on CPU")
    
    # Check required packages
    required_packages = [
        'cv2', 'numpy', 'pandas', 'PIL', 'tqdm', 'editdistance', 'matplotlib'
    ]
    
    print("\nPackage Check:")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Not installed")
    
    print("\n" + "=" * 60)


def check_dataset(data_root='./data'):
    """Check if dataset is properly organized"""
    print("\n" + "=" * 60)
    print("Dataset Check")
    print("=" * 60)
    
    if not os.path.exists(data_root):
        print(f"✗ Dataset directory not found: {data_root}")
        print(f"  Please create the directory and organize your data as described in README.md")
        return False
    
    required_structure = {
        'Training': ['training_word', 'training_labels.csv'],
        'Validation': ['validation_words', 'validation_labels.csv'],
        'Testing': ['testing_words', 'testing_labels.csv']
    }
    
    all_good = True
    for split, items in required_structure.items():
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            print(f"✗ Missing directory: {split_path}")
            all_good = False
            continue
        
        print(f"✓ {split} directory found")
        
        for item in items:
            item_path = os.path.join(split_path, item)
            if not os.path.exists(item_path):
                print(f"  ✗ Missing: {item}")
                all_good = False
            else:
                if item.endswith('.csv'):
                    # Check CSV file
                    import pandas as pd
                    try:
                        df = pd.read_csv(item_path)
                        print(f"  ✓ {item} ({len(df)} samples)")
                    except Exception as e:
                        print(f"  ✗ Error reading {item}: {e}")
                        all_good = False
                else:
                    # Check image directory
                    num_images = len([f for f in os.listdir(item_path) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    print(f"  ✓ {item} ({num_images} images)")
    
    print("=" * 60)
    return all_good


def test_model():
    """Test if model can be created and run"""
    print("\n" + "=" * 60)
    print("Model Test")
    print("=" * 60)
    
    try:
        from model import create_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model(img_height=32, img_channels=1, num_classes=80)
        model = model.to(device)
        
        # Test forward pass
        dummy_input = torch.randn(2, 1, 32, 128).to(device)
        output = model(dummy_input)
        
        print(f"✓ Model created successfully")
        print(f"  - Device: {device}")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 60)


def print_next_steps(dataset_ok, model_ok):
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    
    if not dataset_ok:
        print("""
1. Organize your dataset according to the structure:
   data/
   ├── Training/
   │   ├── training_word/
   │   └── training_labels.csv
   ├── Validation/
   │   ├── validation_words/
   │   └── validation_labels.csv
   └── Testing/
       ├── testing_words/
       └── testing_labels.csv

2. Make sure CSV files have at least two columns:
   - Column 1: Image filename
   - Column 2: Text label
""")
    
    if dataset_ok and model_ok:
        print("""
✓ Everything looks good! You can start training:

1. Start training:
   python train.py

2. Monitor training in the terminal
   - Training/validation loss
   - Accuracy and CER metrics
   - Checkpoints are saved in checkpoints/

3. After training, test your model:
   python inference.py --model checkpoints/best_accuracy.pth --image path/to/image.png

4. For batch inference:
   python inference.py --model checkpoints/best_accuracy.pth --image_dir path/to/images/

Tips:
- Training on GTX 1070 should take 5-10 seconds per epoch
- Reduce batch size if you get OOM errors
- Check README.md for detailed documentation
- Use utils.py to visualize training results
""")
    
    print("=" * 60)


def main():
    """Main function"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║  Handwritten Prescription Recognition - Quick Start     ║")
    print("╚" + "═" * 58 + "╝")
    
    # Run checks
    check_environment()
    dataset_ok = check_dataset()
    model_ok = test_model()
    
    # Print next steps
    print_next_steps(dataset_ok, model_ok)


if __name__ == "__main__":
    main()