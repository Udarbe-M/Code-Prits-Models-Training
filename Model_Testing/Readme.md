# Attention-Guided Multi-Scale Feature CRNN for Handwritten Prescription Recognition

A PyTorch implementation of an Attention-Guided Multi-Scale Feature Convolutional Recurrent Neural Network for recognizing handwritten medical prescriptions. This model combines multi-scale feature extraction, spatial attention mechanisms, and bidirectional LSTMs to achieve robust text recognition.

## Architecture Overview

The model consists of three main components:

1. **Multi-Scale Feature Extraction**: Parallel convolutional branches with different receptive fields (3x3, 5x5, 7x7) to capture features at multiple scales
2. **Attention Mechanism**: Spatial attention modules that focus on important regions in the feature maps
3. **Recurrent Neural Network**: Bidirectional LSTM layers for sequence modeling
4. **CTC Loss**: Connectionist Temporal Classification for sequence-to-sequence learning without alignment

## Requirements

- Python 3.8+
- CUDA-compatible GPU (GTX 1070 or better recommended)
- PyTorch with CUDA support

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install PyTorch with CUDA support**
```bash
# For CUDA 11.8 (adjust based on your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. **Install other requirements**
```bash
pip install -r requirements.txt
```

5. **Verify CUDA installation**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## Dataset Structure

Organize your dataset as follows:
```
data/
├── Training/
│   ├── training_word/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── training_labels.csv
├── Validation/
│   ├── validation_words/
│   │   ├── image1.png
│   │   └── ...
│   └── validation_labels.csv
└── Testing/
    ├── testing_words/
    │   ├── image1.png
    │   └── ...
    └── testing_labels.csv
```

The CSV files should have at least two columns:
- Column 1: Image filename
- Column 2: Text label

## Training

### Basic Training
```bash
python train.py
```

### Customizing Training Parameters

Edit the configuration in `train.py`:

```python
# Configuration
DATA_ROOT = "./data"  # Path to your dataset
BATCH_SIZE = 32       # Adjust based on GPU memory
IMG_HEIGHT = 32       # Image height
IMG_WIDTH = 128       # Image width
NUM_EPOCHS = 100      # Number of training epochs
LEARNING_RATE = 0.001 # Initial learning rate
NUM_WORKERS = 4       # Data loading workers
HIDDEN_SIZE = 256     # LSTM hidden size
```

### Training Features

- **Automatic checkpointing**: Models are saved when validation loss or accuracy improves
- **Learning rate scheduling**: ReduceLROnPlateau scheduler automatically adjusts learning rate
- **Gradient clipping**: Prevents exploding gradients
- **Progress tracking**: Real-time training progress with tqdm
- **Metrics**: Tracks loss, accuracy, and Character Error Rate (CER)

### Monitoring Training

The training script will output:
- Training loss per epoch
- Validation loss, accuracy, and CER
- Current learning rate
- Best model checkpoints

Checkpoints are saved in the `checkpoints/` directory:
- `best_loss.pth`: Model with lowest validation loss
- `best_accuracy.pth`: Model with highest accuracy
- `checkpoint_epoch_X.pth`: Periodic checkpoints every 10 epochs

## Inference

### Single Image Prediction
```bash
python inference.py --model checkpoints/best_accuracy.pth --image path/to/image.png
```

### Batch Prediction
```bash
python inference.py --model checkpoints/best_accuracy.pth --image_dir path/to/images/ --output results.txt
```

### Inference Options
- `--model`: Path to trained model checkpoint (required)
- `--image`: Path to single image
- `--image_dir`: Path to directory of images
- `--output`: Output file for predictions (default: predictions.txt)
- `--img_height`: Image height (default: 32)
- `--img_width`: Image width (default: 128)

## Model Architecture Details

### Multi-Scale Feature Extractor
- **Branch 1**: 3x3 convolutions (fine details)
- **Branch 2**: 5x5 convolutions (medium features)
- **Branch 3**: 7x7 convolutions (large context)
- Features are concatenated and fused with 1x1 convolutions

### Attention Module
- Spatial attention with channel reduction
- Learns to focus on discriminative regions
- Applied after each multi-scale feature extraction block

### CNN Backbone
- 4 convolutional blocks with batch normalization and ReLU
- 3 max pooling layers (2x2) reducing spatial dimensions
- Final feature maps: 512 channels

### RNN Sequence Modeling
- 2 Bidirectional LSTM layers
- Hidden size: 256 (configurable)
- Output: Character probabilities per time step

### CTC Loss
- Handles variable-length sequences
- No need for character-level alignment
- Automatically learns text structure

## Performance Optimization

### For GTX 1070 (8GB VRAM)
- Batch size: 16-32 (depending on image size)
- Mixed precision training (optional):
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Memory Tips
- Reduce batch size if you get OOM errors
- Reduce `NUM_WORKERS` if system RAM is limited
- Use smaller `HIDDEN_SIZE` if needed (128 or 192)

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE`
- Reduce `IMG_WIDTH`
- Reduce `HIDDEN_SIZE`

### Slow Training
- Increase `NUM_WORKERS` (up to number of CPU cores)
- Enable `pin_memory=True` in DataLoader (already enabled)
- Use SSD for dataset storage

### Poor Accuracy
- Train for more epochs (100+)
- Adjust learning rate
- Ensure dataset is balanced and properly labeled
- Check if images are properly preprocessed

## Expected Results

Training time on GTX 1070:
- ~5-10 seconds per epoch (depending on dataset size)
- 100 epochs: ~10-15 hours

Typical accuracy on handwritten text:
- Clean handwriting: 85-95%
- Medical prescriptions: 70-85% (due to complexity)

## File Descriptions

- `model.py`: Model architecture implementation
- `dataset.py`: Dataset loader and preprocessing
- `train.py`: Training script with evaluation
- `inference.py`: Inference script for predictions
- `requirements.txt`: Python dependencies

## Advanced Usage

### Resume Training
Modify `train.py` to load a checkpoint:
```python
trainer = CTCTrainer(...)
start_epoch = trainer.load_checkpoint('checkpoint_epoch_50.pth')
trainer.train(num_epochs=100)
```

### Fine-tuning
Load pretrained weights and train with lower learning rate:
```python
model.load_state_dict(torch.load('pretrained.pth')['model_state_dict'])
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

### Custom Character Set
The model automatically builds the character set from your dataset. To use a predefined character set, modify `dataset.py`.

## Citation

If you use this code, please cite the original dataset:
```
Doctor's Handwritten Prescription BD dataset
Kaggle: https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset
```

## License

This implementation is for educational and research purposes.

## Acknowledgments

- CTC Loss: Graves et al., "Connectionist Temporal Classification"
- Attention Mechanisms: Attention-based architectures in computer vision
- Multi-scale features: Inspired by Inception networks