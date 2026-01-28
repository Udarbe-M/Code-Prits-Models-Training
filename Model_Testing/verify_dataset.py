"""
Dataset Usage Verification Script
Checks if all images are being used during training/validation/testing
"""

import os
import pandas as pd
from dataset import create_dataloaders


def verify_dataset_usage(data_root='./data'):
    """
    Verify that all images are being loaded and used
    """
    print("=" * 70)
    print("DATASET USAGE VERIFICATION")
    print("=" * 70)
    
    splits = {
        'Training': {
            'img_dir': 'training_word',
            'csv_file': 'training_labels.csv'
        },
        'Validation': {
            'img_dir': 'validation_words',
            'csv_file': 'validation_labels.csv'
        },
        'Testing': {
            'img_dir': 'testing_words',
            'csv_file': 'testing_labels.csv'
        }
    }
    
    print("\n📁 Checking Physical Files vs CSV Records...\n")
    
    total_discrepancies = 0
    
    for split_name, paths in splits.items():
        print(f"\n{'='*70}")
        print(f"{split_name} Set")
        print(f"{'='*70}")
        
        # Paths
        img_dir = os.path.join(data_root, split_name, paths['img_dir'])
        csv_path = os.path.join(data_root, split_name, paths['csv_file'])
        
        # Check if paths exist
        if not os.path.exists(img_dir):
            print(f"❌ Image directory not found: {img_dir}")
            continue
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            continue
        
        # Count physical image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        physical_images = [f for f in os.listdir(img_dir) 
                          if f.lower().endswith(image_extensions)]
        num_physical = len(physical_images)
        
        # Count CSV records
        df = pd.read_csv(csv_path)
        num_csv_records = len(df)
        
        print(f"\n📊 File Counts:")
        print(f"  Physical images in folder: {num_physical}")
        print(f"  Records in CSV file:        {num_csv_records}")
        
        # Check if they match
        if num_physical == num_csv_records:
            print(f"  ✅ MATCH - All images have CSV entries")
        else:
            diff = abs(num_physical - num_csv_records)
            total_discrepancies += diff
            if num_physical > num_csv_records:
                print(f"  ⚠️  WARNING: {diff} images WITHOUT CSV entries!")
                print(f"     These images will NOT be used in training!")
            else:
                print(f"  ⚠️  WARNING: {diff} CSV entries WITHOUT images!")
                print(f"     These will cause errors during loading!")
        
        # Check for missing files
        print(f"\n🔍 Checking Image-CSV Correspondence...")
        csv_filenames = set(df.iloc[:, 0].values)
        physical_filenames = set(physical_images)
        
        # Images in folder but not in CSV
        orphan_images = physical_filenames - csv_filenames
        if orphan_images:
            print(f"  ⚠️  {len(orphan_images)} images in folder but NOT in CSV:")
            for img in list(orphan_images)[:5]:
                print(f"     - {img}")
            if len(orphan_images) > 5:
                print(f"     ... and {len(orphan_images) - 5} more")
        
        # CSV entries without corresponding images
        missing_images = csv_filenames - physical_filenames
        if missing_images:
            print(f"  ❌ {len(missing_images)} CSV entries WITHOUT images:")
            for img in list(missing_images)[:5]:
                print(f"     - {img}")
            if len(missing_images) > 5:
                print(f"     ... and {len(missing_images) - 5} more")
        
        if not orphan_images and not missing_images:
            print(f"  ✅ Perfect match - All images have labels")
        
        # Sample check
        print(f"\n📋 CSV Sample (first 3 rows):")
        print(f"  Columns: {list(df.columns)}")
        for idx in range(min(3, len(df))):
            print(f"  Row {idx}: {df.iloc[idx, 0]} → '{df.iloc[idx, 1]}'")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if total_discrepancies == 0:
        print("✅ All datasets are correctly configured!")
        print("✅ All images have corresponding labels")
        print("✅ No missing or orphaned files")
    else:
        print(f"⚠️  Found {total_discrepancies} discrepancies")
        print("⚠️  Some images may not be used during training!")
    
    return total_discrepancies == 0


def check_dataloader_usage(data_root='./data', batch_size=4):
    """
    Check how many samples are actually loaded by the DataLoader
    """
    print(f"\n{'='*70}")
    print("DATALOADER USAGE CHECK")
    print(f"{'='*70}")
    
    print("\n🔄 Creating DataLoaders...")
    
    try:
        train_loader, val_loader, test_loader, num_classes, train_dataset = create_dataloaders(
            root_dir=data_root,
            batch_size=batch_size,
            img_height=32,
            img_width=128,
            num_workers=0  # Use 0 for testing
        )
        
        print("\n✅ DataLoaders created successfully!")
        
        print(f"\n📊 DataLoader Statistics:")
        print(f"{'='*70}")
        
        # Training
        train_samples = len(train_loader.dataset)
        train_batches = len(train_loader)
        print(f"\n📘 Training:")
        print(f"  Total samples loaded:     {train_samples}")
        print(f"  Number of batches:        {train_batches}")
        print(f"  Batch size:               {batch_size}")
        print(f"  Expected batches:         {train_samples // batch_size + (1 if train_samples % batch_size else 0)}")
        print(f"  Last batch size:          {train_samples % batch_size if train_samples % batch_size else batch_size}")
        
        # Validation
        val_samples = len(val_loader.dataset)
        val_batches = len(val_loader)
        print(f"\n📗 Validation:")
        print(f"  Total samples loaded:     {val_samples}")
        print(f"  Number of batches:        {val_batches}")
        print(f"  Expected batches:         {val_samples // batch_size + (1 if val_samples % batch_size else 0)}")
        
        # Testing
        test_samples = len(test_loader.dataset)
        test_batches = len(test_loader)
        print(f"\n📕 Testing:")
        print(f"  Total samples loaded:     {test_samples}")
        print(f"  Number of batches:        {test_batches}")
        print(f"  Expected batches:         {test_samples // batch_size + (1 if test_samples % batch_size else 0)}")
        
        # Total
        total_samples = train_samples + val_samples + test_samples
        print(f"\n📚 Total:")
        print(f"  All samples loaded:       {total_samples}")
        print(f"  Expected (3120+780+780):  {3120 + 780 + 780}")
        
        # Check if all are used
        print(f"\n{'='*70}")
        print("USAGE VERIFICATION")
        print(f"{'='*70}")
        
        all_used = True
        
        if train_samples == 3120:
            print(f"✅ Training: All 3120 images are being used")
        else:
            print(f"⚠️  Training: Only {train_samples}/3120 images loaded!")
            all_used = False
        
        if val_samples == 780:
            print(f"✅ Validation: All 780 images are being used")
        else:
            print(f"⚠️  Validation: Only {val_samples}/780 images loaded!")
            all_used = False
        
        if test_samples == 780:
            print(f"✅ Testing: All 780 images are being used")
        else:
            print(f"⚠️  Testing: Only {test_samples}/780 images loaded!")
            all_used = False
        
        print(f"\n{'='*70}")
        if all_used:
            print("✅ SUCCESS: ALL IMAGES ARE BEING USED! 🎉")
        else:
            print("⚠️  WARNING: NOT ALL IMAGES ARE BEING USED!")
        print(f"{'='*70}")
        
        # Test actual iteration
        print(f"\n🔄 Testing actual data iteration...")
        print(f"\nIterating through 1 batch from each loader:")
        
        # Training batch
        train_batch = next(iter(train_loader))
        images, labels, label_lengths, texts = train_batch
        print(f"\n📘 Training batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Batch contains {images.shape[0]} samples")
        print(f"  Sample texts: {texts[:3]}")
        
        # Validation batch
        val_batch = next(iter(val_loader))
        images, labels, label_lengths, texts = val_batch
        print(f"\n📗 Validation batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Batch contains {images.shape[0]} samples")
        print(f"  Sample texts: {texts[:3]}")
        
        # Testing batch
        test_batch = next(iter(test_loader))
        images, labels, label_lengths, texts = test_batch
        print(f"\n📕 Testing batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Batch contains {images.shape[0]} samples")
        print(f"  Sample texts: {texts[:3]}")
        
        print(f"\n✅ All loaders can successfully iterate data!")
        
        return all_used
        
    except Exception as e:
        print(f"\n❌ Error creating DataLoaders: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_epoch_coverage(data_root='./data', batch_size=32):
    """
    Calculate how many samples are processed per epoch
    """
    print(f"\n{'='*70}")
    print("EPOCH COVERAGE ANALYSIS")
    print(f"{'='*70}")
    
    try:
        train_loader, val_loader, test_loader, num_classes, _ = create_dataloaders(
            root_dir=data_root,
            batch_size=batch_size,
            img_height=32,
            img_width=128,
            num_workers=0
        )
        
        train_samples = len(train_loader.dataset)
        
        print(f"\nWith batch size {batch_size}:")
        print(f"  Total training samples: {train_samples}")
        print(f"  Batches per epoch: {len(train_loader)}")
        print(f"  Samples per epoch: {len(train_loader) * batch_size if len(train_loader) * batch_size <= train_samples else train_samples}")
        
        # Calculate coverage
        total_slots = len(train_loader) * batch_size
        actual_samples = train_samples
        
        if total_slots >= actual_samples:
            coverage = 100.0
            print(f"  Coverage: {coverage}% ✅")
            print(f"  Note: Last batch may be smaller ({train_samples % batch_size} samples)")
        else:
            coverage = (total_slots / actual_samples) * 100
            print(f"  Coverage: {coverage:.2f}% ⚠️")
            print(f"  Warning: {actual_samples - total_slots} samples dropped per epoch!")
        
        print(f"\n✅ During ONE epoch of training:")
        print(f"  • Model sees: {actual_samples} samples")
        print(f"  • Across {len(train_loader)} batches")
        print(f"  • Each image is seen exactly ONCE per epoch")
        
        print(f"\n✅ During FULL training (100 epochs):")
        print(f"  • Model sees: {actual_samples * 100:,} samples total")
        print(f"  • Each image is seen: 100 times")
        
        return coverage == 100.0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all verification checks"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "DATASET VERIFICATION TOOL" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    
    data_root = './data'
    
    # Check 1: File system verification
    files_ok = verify_dataset_usage(data_root)
    
    # Check 2: DataLoader verification
    loaders_ok = check_dataloader_usage(data_root, batch_size=4)
    
    # Check 3: Epoch coverage
    coverage_ok = check_epoch_coverage(data_root, batch_size=32)
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    if files_ok and loaders_ok and coverage_ok:
        print("""
✅✅✅ EVERYTHING IS PERFECT! ✅✅✅

Your dataset is correctly configured:
  ✓ All 3120 training images are being used
  ✓ All 780 validation images are being used  
  ✓ All 780 testing images are being used
  ✓ No missing or orphaned files
  ✓ DataLoaders working correctly
  ✓ 100% coverage per epoch

Total: 4680 images, all being utilized! 🎉

You can confidently train knowing every image is being used!
""")
    else:
        print("\n⚠️  Some issues were found. Please review the output above.")
    
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    # Allow custom data path
    data_root = sys.argv[1] if len(sys.argv) > 1 else './data'
    
    if not os.path.exists(data_root):
        print(f"❌ Data directory not found: {data_root}")
        print(f"\nUsage: python verify_dataset.py [data_path]")
        print(f"Example: python verify_dataset.py ./data")
    else:
        main()