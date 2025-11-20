import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
from tqdm import tqdm

# Import your model and config from the original script
# Ensure QAViTV2_EXTREME.py is in the same directory
try:
    from QAViTV2_EXTREME import QAViT, QAViTConfig, TrainingConfig
except ImportError:
    print("Error: Could not import from QAViTV2_EXTREME.py. Make sure the file exists in the same directory.")
    exit(1)

# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT_PATH = "./checkpoints/best_model.pth"  # Path to your best checkpoint
DATA_ROOT = "./data"
BATCH_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-100 Constants for un-normalization
MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1).to(DEVICE)
STD = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1).to(DEVICE)

def load_model(checkpoint_path):
    """Loads the model architecture and weights from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Load Config
    # If config is saved in checkpoint, use it. Otherwise use default.
    if 'model_config' in checkpoint:
        print("✓ Loaded model configuration from checkpoint")
        config = checkpoint['model_config']
    else:
        print("⚠ Config not found in checkpoint, using default configuration")
        config = QAViTConfig()

    model = QAViT(config).to(DEVICE)
    
    # Load State Dict
    state_dict = checkpoint['model_state_dict']
    
    # Handle torch.compile prefix ('_orig_mod.')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    
    print(f"✓ Model loaded successfully (Epoch {checkpoint.get('epoch', 'Unknown')})")
    print(f"  Validation Accuracy at save: {checkpoint.get('val_acc', 'Unknown')}%")
    
    return model

def get_test_loader():
    """Returns the CIFAR-100 Test Loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    test_dataset = datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=transform)
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return loader, test_dataset.classes

def unnormalize(img_tensor):
    """Reverts normalization for visualization."""
    img_tensor = img_tensor * STD + MEAN
    img_tensor = torch.clamp(img_tensor, 0, 1)
    return img_tensor.permute(1, 2, 0).cpu().numpy()

def evaluate_model(model, loader, classes):
    """Comprehensive evaluation loop."""
    all_preds = []
    all_targets = []
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    print("\nRunning Inference on Test Set...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            
            # Top-1
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Accuracy Calculation
            total += targets.size(0)
            top1_correct += pred.eq(targets).sum().item()
            
            # Top-5
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

    top1_acc = 100. * top1_correct / total
    top5_acc = 100. * top5_correct / total
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS")
    print(f"{'='*50}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"{'='*50}")
    
    return np.array(all_preds), np.array(all_targets)

def plot_confusion_matrix(preds, targets, classes):
    """Plots a confusion matrix (simplified for 100 classes)."""
    cm = confusion_matrix(targets, preds)
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, cmap='viridis', square=True, cbar=False)
    plt.title('Confusion Matrix (Heatmap only for readability)', fontsize=20)
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('True', fontsize=15)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\n✓ Saved 'confusion_matrix.png'")

def analyze_class_performance(preds, targets, classes):
    """Prints best and worst performing classes."""
    report = classification_report(targets, preds, target_names=classes, output_dict=True)
    
    class_accs = []
    for name in classes:
        if name in report:
            class_accs.append((name, report[name]['precision']))
    
    class_accs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Top 10 Best Classes':<30} {'Precision':<10}")
    print("-" * 40)
    for name, score in class_accs[:10]:
        print(f"{name:<30} {score:.2%}")
        
    print(f"\n{'Top 10 Worst Classes':<30} {'Precision':<10}")
    print("-" * 40)
    for name, score in class_accs[-10:]:
        print(f"{name:<30} {score:.2%}")

def visualize_predictions(model, loader, classes, num_images=10):
    """Visualizes a grid of predictions vs ground truth."""
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = outputs.max(1)
    
    # Select random indices
    indices = np.random.choice(len(inputs), num_images, replace=False)
    
    fig = plt.figure(figsize=(15, 6))
    
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(2, 5, i + 1)
        img = unnormalize(inputs[idx])
        ax.imshow(img)
        
        pred_label = classes[preds[idx].item()]
        true_label = classes[targets[idx].item()]
        
        color = 'green' if pred_label == true_label else 'red'
        
        ax.set_title(f"P: {pred_label}\nT: {true_label}", color=color, fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('predictions.png')
    print("✓ Saved 'predictions.png'")

def main():
    # 1. Load Data
    loader, classes = get_test_loader()
    
    # 2. Load Model
    model = load_model(CHECKPOINT_PATH)
    
    # 3. Evaluate
    preds, targets = evaluate_model(model, loader, classes)
    
    # 4. Analyze
    analyze_class_performance(preds, targets, classes)
    
    # 5. Visualize
    visualize_predictions(model, loader, classes)
    plot_confusion_matrix(preds, targets, classes)

if __name__ == "__main__":
    main()