import os
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import torchvision.transforms as transforms


class PrescriptionDataset(Dataset):
    """
    Dataset loader for Doctor's Handwritten Prescription dataset
    """
    def __init__(self, root_dir, split='Training', img_height=32, img_width=128, 
                 max_label_length=25, transform=None):
        """
        Args:
            root_dir: Root directory containing Testing, Training, Validation folders
            split: One of 'Training', 'Testing', 'Validation'
            img_height: Target height for images
            img_width: Target width for images
            max_label_length: Maximum length of text labels
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.split = split
        self.img_height = img_height
        self.img_width = img_width
        self.max_label_length = max_label_length
        self.transform = transform
        
        # Set paths based on split
        if split == 'Training':
            self.img_dir = os.path.join(root_dir, 'Training', 'training_word')
            self.label_file = os.path.join(root_dir, 'Training', 'training_labels.csv')
        elif split == 'Testing':
            self.img_dir = os.path.join(root_dir, 'Testing', 'testing_words')
            self.label_file = os.path.join(root_dir, 'Testing', 'testing_labels.csv')
        elif split == 'Validation':
            self.img_dir = os.path.join(root_dir, 'Validation', 'validation_words')
            self.label_file = os.path.join(root_dir, 'Validation', 'validation_labels.csv')
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'Training', 'Testing', or 'Validation'")
        
        # Load labels
        self.df = pd.read_csv(self.label_file)
        print(f"Loaded {len(self.df)} samples from {split} split")
        
        # Build character set
        self.char_set = set()
        for label in self.df.iloc[:, 1]:  # Assuming second column contains labels
            self.char_set.update(label)
        
        # Create char to index mapping (0 reserved for CTC blank)
        self.chars = sorted(list(self.char_set))
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.chars)}
        self.idx_to_char[0] = ''  # CTC blank
        
        self.num_classes = len(self.chars) + 1  # +1 for CTC blank
        
        print(f"Character set size: {self.num_classes} (including CTC blank)")
        print(f"Characters: {''.join(self.chars)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image filename and label from CSV
        img_name = self.df.iloc[idx, 0]
        label = str(self.df.iloc[idx, 1])
        
        # Load image
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            # Try to load with PIL first
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            image = np.array(image)
        except:
            # Fallback to OpenCV
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not load image: {img_path}")
        
        # Resize image
        image = cv2.resize(image, (self.img_width, self.img_height))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor: H x W -> 1 x H x W
        image = torch.FloatTensor(image).unsqueeze(0)
        
        # Apply additional transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Encode label
        encoded_label = self.encode_text(label)
        label_length = len(encoded_label)
        
        return image, encoded_label, label_length, label
    
    def encode_text(self, text):
        """Encode text to indices"""
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                # Handle unknown characters (optional: skip or use special token)
                pass
        return encoded
    
    def decode_text(self, indices):
        """Decode indices to text"""
        decoded = []
        for idx in indices:
            if idx in self.idx_to_char and idx != 0:  # Skip CTC blank
                decoded.append(self.idx_to_char[idx])
        return ''.join(decoded)


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable length labels
    """
    images, labels, label_lengths, texts = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Concatenate all labels
    labels_concat = []
    for label in labels:
        labels_concat.extend(label)
    
    labels = torch.LongTensor(labels_concat)
    label_lengths = torch.LongTensor(label_lengths)
    
    return images, labels, label_lengths, texts


def create_dataloaders(root_dir, batch_size=32, img_height=32, img_width=128, 
                       num_workers=4, max_label_length=25):
    """
    Create DataLoaders for training, validation, and testing
    
    Args:
        root_dir: Root directory containing the dataset
        batch_size: Batch size for DataLoader
        img_height: Target height for images
        img_width: Target width for images
        num_workers: Number of worker processes for data loading
        max_label_length: Maximum length of text labels
    
    Returns:
        train_loader, val_loader, test_loader, charset_size
    """
    # Create datasets
    train_dataset = PrescriptionDataset(
        root_dir=root_dir,
        split='Training',
        img_height=img_height,
        img_width=img_width,
        max_label_length=max_label_length
    )
    
    val_dataset = PrescriptionDataset(
        root_dir=root_dir,
        split='Validation',
        img_height=img_height,
        img_width=img_width,
        max_label_length=max_label_length
    )
    
    test_dataset = PrescriptionDataset(
        root_dir=root_dir,
        split='Testing',
        img_height=img_height,
        img_width=img_width,
        max_label_length=max_label_length
    )
    
    # Merge character sets from all splits
    all_chars = set()
    all_chars.update(train_dataset.chars)
    all_chars.update(val_dataset.chars)
    all_chars.update(test_dataset.chars)
    
    # Update all datasets with merged character set
    chars = sorted(list(all_chars))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
    idx_to_char = {idx + 1: char for idx, char in enumerate(chars)}
    idx_to_char[0] = ''
    
    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset.chars = chars
        dataset.char_to_idx = char_to_idx
        dataset.idx_to_char = idx_to_char
        dataset.num_classes = len(chars) + 1
    
    print(f"\nFinal character set size: {len(chars) + 1} (including CTC blank)")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes, train_dataset


if __name__ == "__main__":
    # Test the dataset loader
    root_dir = "./data"  # Adjust this path
    
    try:
        train_loader, val_loader, test_loader, num_classes, train_dataset = create_dataloaders(
            root_dir=root_dir,
            batch_size=4,
            img_height=32,
            img_width=128,
            num_workers=0  # Use 0 for testing
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Testing batches: {len(test_loader)}")
        
        # Test one batch
        images, labels, label_lengths, texts = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"Images: {images.shape}")
        print(f"Labels: {labels.shape}")
        print(f"Label lengths: {label_lengths.shape}")
        print(f"Sample texts: {texts[:2]}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the dataset path is correct!")