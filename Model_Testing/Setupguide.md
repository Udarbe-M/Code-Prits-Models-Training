# Complete Setup Guide for Windows with VS Code and GTX 1070

## Prerequisites

- Windows 10/11
- NVIDIA GTX 1070 GPU
- 8GB+ RAM
- Python 3.8 or higher
- Visual Studio Code

## Step 1: Install NVIDIA Drivers and CUDA

### 1.1 Check Current Driver
```bash
nvidia-smi
```

If this command works, note your CUDA version. If not, proceed with driver installation.

### 1.2 Install NVIDIA Driver
1. Go to https://www.nvidia.com/Download/index.aspx
2. Select:
   - Product: GeForce
   - Series: GeForce 10 Series
   - Model: GeForce GTX 1070
   - OS: Windows 10/11
3. Download and install the driver
4. Restart your computer

### 1.3 Install CUDA Toolkit (11.8 recommended)
1. Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive
2. Select:
   - OS: Windows
   - Architecture: x86_64
   - Version: 10/11
   - Installer Type: exe (local)
3. Run installer and follow instructions
4. Restart computer

### 1.4 Verify Installation
```bash
nvidia-smi
nvcc --version
```

## Step 2: Install Python and VS Code

### 2.1 Install Python
1. Download Python 3.10 from: https://www.python.org/downloads/
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify installation:
```bash
python --version
pip --version
```

### 2.2 Install Visual Studio Code
1. Download from: https://code.visualstudio.com/
2. Install Python extension in VS Code:
   - Open VS Code
   - Click Extensions (Ctrl+Shift+X)
   - Search "Python"
   - Install the official Python extension by Microsoft

## Step 3: Set Up Project

### 3.1 Create Project Directory
```bash
mkdir prescription-recognition
cd prescription-recognition
```

### 3.2 Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows Command Prompt:
venv\Scripts\activate.bat

# On Windows PowerShell:
venv\Scripts\Activate.ps1

# On Git Bash:
source venv/Scripts/activate
```

### 3.3 Install PyTorch with CUDA
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3.4 Verify PyTorch CUDA Installation
```python
python
>>> import torch
>>> print(f"PyTorch version: {torch.__version__}")
>>> print(f"CUDA available: {torch.cuda.is_available()}")
>>> print(f"CUDA version: {torch.version.cuda}")
>>> print(f"GPU: {torch.cuda.get_device_name(0)}")
>>> exit()
```

You should see:
```
PyTorch version: 2.x.x+cu118
CUDA available: True
CUDA version: 11.8
GPU: GeForce GTX 1070
```

### 3.5 Install Other Dependencies
```bash
pip install opencv-python numpy pandas Pillow tqdm editdistance matplotlib scikit-learn
```

## Step 4: Download Dataset

### 4.1 Get Dataset from Kaggle
1. Go to: https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset
2. Click "Download" button
3. Extract the zip file

### 4.2 Organize Dataset
Create this structure in your project folder:
```
prescription-recognition/
├── data/
│   ├── Training/
│   │   ├── training_word/
│   │   │   ├── image1.png
│   │   │   ├── image2.png
│   │   │   └── ...
│   │   └── training_labels.csv
│   ├── Validation/
│   │   ├── validation_words/
│   │   └── validation_labels.csv
│   └── Testing/
│       ├── testing_words/
│       └── testing_labels.csv
├── venv/
├── model.py
├── dataset.py
├── train.py
└── ...
```

**IMPORTANT**: Make sure folder names match exactly:
- `training_word` (not training_words)
- `validation_words` (not validation_word)
- `testing_words` (not testing_word)

## Step 5: Add Project Files

Copy all the Python files into your project directory:
- model.py
- dataset.py
- train.py
- train_with_config.py
- inference.py
- config.py
- utils.py
- quickstart.py
- requirements.txt

## Step 6: Open in VS Code

### 6.1 Open Project
```bash
code .
```

Or:
1. Open VS Code
2. File → Open Folder
3. Select your project folder

### 6.2 Select Python Interpreter
1. Press Ctrl+Shift+P
2. Type "Python: Select Interpreter"
3. Choose the one from your venv folder

### 6.3 Configure VS Code Terminal
1. Open Terminal (Ctrl+`)
2. Make sure it's using your virtual environment
3. You should see `(venv)` at the start of the prompt

## Step 7: Test Setup

### 7.1 Run Quick Start Test
```bash
python quickstart.py
```

This will check:
- Python and package versions
- CUDA availability
- Dataset structure
- Model creation

### 7.2 Fix Common Issues

**Issue: "CUDA out of memory"**
```python
# Edit config.py
class GTX1070Config(Config):
    BATCH_SIZE = 16  # Reduce from 24
```

**Issue: "No module named 'cv2'"**
```bash
pip install opencv-python
```

**Issue: "No module named 'editdistance'"**
```bash
pip install editdistance
```

**Issue: NUM_WORKERS error on Windows**
```python
# Edit config.py
class GTX1070Config(Config):
    NUM_WORKERS = 0  # Change from 4
```

## Step 8: Start Training

### 8.1 Quick Test Run
```bash
# Test with 2 epochs
python train_with_config.py --config TestConfig
```

### 8.2 Full Training
```bash
# GTX 1070 optimized training
python train_with_config.py --config GTX1070Config
```

### 8.3 Monitor Training
Training will show:
```
Epoch 1/100
Training: 100%|████████| 125/125 [00:05<00:00, 25.00it/s, loss=3.456]
Validation: 100%|████████| 32/32 [00:01<00:00, 20.00it/s]

Train Loss: 3.4567
Val Loss: 3.1234
Accuracy: 15.67%
CER: 0.8234
Learning Rate: 0.001000
✓ Saved best model (loss)
```

## Step 9: Common VS Code Tips

### 9.1 Useful Shortcuts
- `Ctrl+` ` : Toggle terminal
- `Ctrl+Shift+P`: Command palette
- `F5`: Run with debugger
- `Ctrl+S`: Save file
- `Ctrl+/`: Comment/uncomment line

### 9.2 Debugging
1. Click on line number to set breakpoint
2. Press F5 to start debugging
3. Use Debug Console to inspect variables

### 9.3 Running Scripts
**Method 1: Terminal**
```bash
python train_with_config.py --config GTX1070Config
```

**Method 2: Right-click**
1. Right-click on Python file
2. Select "Run Python File in Terminal"

**Method 3: Debug Configuration**
Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Train Model",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/train_with_config.py",
            "args": ["--config", "GTX1070Config"],
            "console": "integratedTerminal"
        }
    ]
}
```

## Step 10: After Training

### 10.1 Check Results
Checkpoints are saved in `checkpoints/` folder:
- `best_loss.pth`: Model with lowest validation loss
- `best_accuracy.pth`: Model with highest accuracy

### 10.2 Test Model
```bash
python inference.py --model checkpoints/best_accuracy.pth --image test_image.png
```

### 10.3 Visualize Results
```python
from utils import load_and_visualize_checkpoint
load_and_visualize_checkpoint('checkpoints/best_accuracy.pth')
```

## Troubleshooting Guide

### GPU Not Being Used
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.current_device())  # Should be 0
```

If False:
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with CUDA
3. Restart VS Code

### Out of Memory Errors
```python
# Reduce batch size in config.py
BATCH_SIZE = 8  # or even 4
```

### Slow Training
1. Make sure CUDA is being used
2. Close other GPU-intensive applications
3. Increase NUM_WORKERS (only if you have good CPU)

### Dataset Not Found
```python
# Check your paths
import os
print(os.path.exists('./data'))
print(os.listdir('./data'))
```

### Module Import Errors
```bash
# Make sure virtual environment is activated
which python  # Should show path to venv
pip list  # Check installed packages
```

## Performance Expectations (GTX 1070)

- **Training Speed**: ~5-10 seconds per epoch
- **GPU Utilization**: 70-90%
- **Memory Usage**: 4-6 GB VRAM
- **100 Epochs**: ~10-15 hours
- **Expected Accuracy**: 70-85% (depending on data quality)

## Next Steps

1. ✓ Complete setup
2. ✓ Test with quickstart.py
3. ✓ Run short test training (2-5 epochs)
4. ✓ Monitor GPU usage with `nvidia-smi`
5. ✓ Start full training
6. Monitor training metrics
7. Evaluate on test set
8. Fine-tune hyperparameters if needed

## Getting Help

If you encounter issues:
1. Check error messages carefully
2. Run `quickstart.py` to diagnose
3. Verify dataset structure
4. Check CUDA installation
5. Review this guide step by step

Good luck with your training! 🚀