import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import os
from model import create_model


class PrescriptionRecognizer:
    """
    Inference class for handwritten prescription recognition
    """
    def __init__(self, model_path, img_height=32, img_width=128, device='cuda'):
        self.img_height = img_height
        self.img_width = img_width
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint to get character set
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # You need to save and load the character mapping
        # For now, we'll create a dummy one - you should save this during training
        print(f"Loading model from {model_path}")
        
        # Create model (you need to know num_classes)
        # This should be saved in the checkpoint
        self.model = create_model(
            img_height=img_height,
            img_channels=1,
            num_classes=checkpoint.get('num_classes', 80),  # Default to 80 if not saved
            hidden_size=256
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        """
        # Load image
        if isinstance(image_path, str):
            try:
                image = Image.open(image_path).convert('L')
                image = np.array(image)
            except:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            # If already numpy array
            image = image_path
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize
        image = cv2.resize(image, (self.img_width, self.img_height))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W
        
        return image
    
    def decode_prediction(self, output, char_to_idx=None):
        """
        Decode model output to text using greedy decoding
        """
        # output: (1, width, num_classes)
        _, preds = output.max(2)  # (1, width)
        preds = preds.squeeze(0).cpu().numpy()
        
        # Remove consecutive duplicates and blanks
        decoded = []
        prev = -1
        for p in preds:
            if p != prev and p != 0:  # 0 is CTC blank
                decoded.append(int(p))
            prev = p
        
        # If you have character mapping, convert to text
        # Otherwise return indices
        if char_to_idx is not None:
            idx_to_char = {v: k for k, v in char_to_idx.items()}
            text = ''.join([idx_to_char.get(idx, '?') for idx in decoded])
            return text
        else:
            return str(decoded)
    
    def predict(self, image_path, char_to_idx=None):
        """
        Predict text from image
        """
        # Preprocess
        image = self.preprocess_image(image_path)
        image = image.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image)
        
        # Decode
        text = self.decode_prediction(output, char_to_idx)
        
        return text
    
    def predict_batch(self, image_paths, char_to_idx=None):
        """
        Predict text from multiple images
        """
        results = []
        
        for img_path in image_paths:
            text = self.predict(img_path, char_to_idx)
            results.append({
                'image': img_path,
                'prediction': text
            })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Prescription Recognition Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--output', type=str, default='predictions.txt', help='Output file')
    parser.add_argument('--img_height', type=int, default=32, help='Image height')
    parser.add_argument('--img_width', type=int, default=128, help='Image width')
    
    args = parser.parse_args()
    
    # Create recognizer
    recognizer = PrescriptionRecognizer(
        model_path=args.model,
        img_height=args.img_height,
        img_width=args.img_width
    )
    
    # Predict
    if args.image:
        # Single image
        text = recognizer.predict(args.image)
        print(f"Prediction: {text}")
        
        # Save to file
        with open(args.output, 'w') as f:
            f.write(f"{args.image}: {text}\n")
    
    elif args.image_dir:
        # Directory of images
        image_files = [
            os.path.join(args.image_dir, f) 
            for f in os.listdir(args.image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        results = recognizer.predict_batch(image_files)
        
        # Print and save results
        with open(args.output, 'w') as f:
            for result in results:
                print(f"{result['image']}: {result['prediction']}")
                f.write(f"{result['image']}: {result['prediction']}\n")
    
    else:
        print("Please provide either --image or --image_dir")


if __name__ == "__main__":
    main()