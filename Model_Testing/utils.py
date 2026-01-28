import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from model import create_model
import os


def plot_training_history(train_losses, val_losses, val_accuracies, save_path='training_history.png'):
    """
    Plot training history including loss and accuracy curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, [acc * 100 for acc in val_accuracies], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def visualize_predictions(model, dataset, device, num_samples=10, save_dir='predictions_viz'):
    """
    Visualize model predictions on sample images
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Get random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx, sample_idx in enumerate(indices):
        image, _, _, true_text = dataset[sample_idx]
        
        # Predict
        with torch.no_grad():
            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)
        
        # Decode prediction
        _, preds = output.max(2)
        preds = preds.squeeze(0).cpu().numpy()
        
        # Remove consecutive duplicates and blanks
        decoded = []
        prev = -1
        for p in preds:
            if p != prev and p != 0:
                decoded.append(int(p))
            prev = p
        
        pred_text = dataset.decode_text(decoded)
        
        # Visualize
        img_array = image.squeeze().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.imshow(img_array, cmap='gray')
        ax.axis('off')
        
        title = f"True: {true_text}\nPred: {pred_text}"
        if true_text == pred_text:
            color = 'green'
            title += " ✓"
        else:
            color = 'red'
            title += " ✗"
        
        ax.set_title(title, fontsize=12, fontweight='bold', color=color)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_{idx+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Prediction visualizations saved to {save_dir}/")


def visualize_attention_maps(model, image, device, save_path='attention_maps.png'):
    """
    Visualize attention maps from the model
    Note: This requires modifying the model to return attention weights
    """
    model.eval()
    
    with torch.no_grad():
        image_input = image.unsqueeze(0).to(device)
        
        # You would need to modify the model to return attention maps
        # This is a placeholder for the concept
        
        # Forward pass
        output = model(image_input)
    
    print("Note: Attention visualization requires model modification to return attention weights")


def analyze_errors(predictions, ground_truths, save_path='error_analysis.txt'):
    """
    Analyze and categorize prediction errors
    """
    errors = []
    error_types = {
        'substitution': 0,
        'insertion': 0,
        'deletion': 0,
        'correct': 0
    }
    
    for pred, true in zip(predictions, ground_truths):
        if pred == true:
            error_types['correct'] += 1
        else:
            errors.append((pred, true))
            
            # Simple error categorization
            if len(pred) > len(true):
                error_types['insertion'] += 1
            elif len(pred) < len(true):
                error_types['deletion'] += 1
            else:
                error_types['substitution'] += 1
    
    # Save analysis
    with open(save_path, 'w') as f:
        f.write("Error Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total samples: {len(predictions)}\n")
        f.write(f"Correct: {error_types['correct']} ({error_types['correct']/len(predictions)*100:.2f}%)\n")
        f.write(f"Errors: {len(errors)} ({len(errors)/len(predictions)*100:.2f}%)\n\n")
        
        f.write("Error Breakdown:\n")
        f.write(f"Substitutions: {error_types['substitution']}\n")
        f.write(f"Insertions: {error_types['insertion']}\n")
        f.write(f"Deletions: {error_types['deletion']}\n\n")
        
        f.write("Sample Errors:\n")
        f.write("-" * 50 + "\n")
        for i, (pred, true) in enumerate(errors[:20], 1):
            f.write(f"{i}. True: '{true}' | Pred: '{pred}'\n")
    
    print(f"Error analysis saved to {save_path}")
    return error_types


def create_confusion_matrix(predictions, ground_truths, char_set):
    """
    Create character-level confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Flatten character sequences
    true_chars = []
    pred_chars = []
    
    for pred, true in zip(predictions, ground_truths):
        min_len = min(len(pred), len(true))
        true_chars.extend(list(true[:min_len]))
        pred_chars.extend(list(pred[:min_len]))
    
    # Get unique characters
    unique_chars = sorted(set(true_chars + pred_chars))
    
    if len(unique_chars) > 50:
        print("Too many characters for confusion matrix visualization")
        return
    
    # Create confusion matrix
    cm = confusion_matrix(true_chars, pred_chars, labels=unique_chars)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_chars, yticklabels=unique_chars)
    plt.title('Character-Level Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to confusion_matrix.png")
    plt.close()


def load_and_visualize_checkpoint(checkpoint_path, save_dir='visualization'):
    """
    Load checkpoint and create visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Plot training history
    if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
        plot_training_history(
            checkpoint['train_losses'],
            checkpoint['val_losses'],
            checkpoint['val_accuracies'],
            save_path=os.path.join(save_dir, 'training_history.png')
        )
    
    print(f"\nCheckpoint Information:")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best Validation Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    print(f"Best Accuracy: {checkpoint.get('best_accuracy', 'N/A')*100:.2f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualization utilities')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--visualize_predictions', action='store_true', 
                       help='Visualize model predictions')
    parser.add_argument('--data_root', type=str, default='./data', 
                       help='Root directory of dataset')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        load_and_visualize_checkpoint(args.checkpoint)
        
        if args.visualize_predictions:
            print("\nTo visualize predictions, please run this as a module with dataset loaded")
            print("Example usage in a script:")
            print("from utils import visualize_predictions")
            print("visualize_predictions(model, dataset, device)")