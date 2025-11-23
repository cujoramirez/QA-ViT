import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from dataclasses import dataclass

# Try to import the HQA model and configs
try:
    from HQAViT_CIFAR100 import HQAViT, HQAViTConfig, TrainingConfig
except ImportError:
    print("Error: Could not import from HQAViT_CIFAR100.py. Make sure the file exists in the same directory.")
    raise


# Placeholder for finetune's FineTuneConfig so torch.load can unpickle checkpoints
# that serialized an instance of that class from the finetune script's __main__.
@dataclass
class FineTuneConfig:
    data_root: str = './data'

# ============================================================================
# Configuration
# ============================================================================
CLEAR_CHECK = None
CHECKPOINT_PATH = os.path.join("checkpoints_finetuned", "best_finetuned.pth")
# Prefer data_root from TrainingConfig (pretraining), but allow overriding from checkpoint
try:
    DATA_ROOT = TrainingConfig().data_root
except Exception:
    DATA_ROOT = "./data"
BATCH_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def load_model(checkpoint_path: str = CHECKPOINT_PATH, device: torch.device = DEVICE):
    """Loads the HQAViT model and weights (if checkpoint exists).
    If checkpoint not found, returns a freshly initialized model on device.
    Handles common state-dict prefixes (e.g., `_orig_mod.` from torch.compile).
    """
    # Load checkpoint early so we can build model from saved config if present
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Returning uninitialized model.")
        config = HQAViTConfig()
        model = HQAViT(config).to(device)
        model.eval()
        return model

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # If checkpoint contains a saved model_config, prefer it
    model_cfg = checkpoint.get('model_config', None)
    if model_cfg is None:
        config = HQAViTConfig()
    else:
        # model_cfg may be an instance of HQAViTConfig or a dict
        try:
            if isinstance(model_cfg, dict):
                config = HQAViTConfig(**model_cfg)
            else:
                config = model_cfg
        except Exception:
            config = HQAViTConfig()

    model = HQAViT(config).to(device)
    model.eval()

    # Support common key names
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, nn.Module):
        # whole model saved
        return checkpoint.to(device)
    else:
        # Unknown checkpoint format
        state_dict = None

    if state_dict is None:
        raise RuntimeError('Unable to find a state dict in checkpoint.')

    # Fix keys from torch.compile or DataParallel
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state[k[10:]] = v
        elif k.startswith('module.'):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=False)

    # If checkpoint contains training/config info, update DATA_ROOT if available
    global DATA_ROOT
    chk_cfg = checkpoint.get('config', None)
    if chk_cfg is not None:
        try:
            # config could be dataclass instance or dict
            if hasattr(chk_cfg, 'data_root'):
                DATA_ROOT = chk_cfg.data_root
            elif isinstance(chk_cfg, dict) and 'data_root' in chk_cfg:
                DATA_ROOT = chk_cfg['data_root']
        except Exception:
            pass

    print(f"Model loaded (epoch={checkpoint.get('epoch', 'unknown')}, val_acc={checkpoint.get('val_acc', 'unknown')})")
    return model


def get_cifar100_loader(data_root: str = DATA_ROOT, batch_size: int = BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader, test_dataset.classes


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device = DEVICE):
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Top-1
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            total += targets.size(0)
            top1_correct += pred.eq(targets).sum().item()

            # Top-5
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

    top1_acc = 100. * top1_correct / total
    top5_acc = 100. * top5_correct / total

    print(f"\nTest Results: Top-1: {top1_acc:.2f}%, Top-5: {top5_acc:.2f}%")
    return np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(preds, targets, classes, out_path='confusion_matrix_hqa.png'):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(18, 18))
    plt.imshow(cm, interpolation='nearest', cmap='viridis')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def analyze_class_performance(preds, targets, classes):
    report = classification_report(targets, preds, target_names=classes, output_dict=True)
    class_accs = []
    for name in classes:
        if name in report:
            class_accs.append((name, report[name].get('precision', 0.0)))
    class_accs.sort(key=lambda x: x[1], reverse=True)

    print('\nTop 10 Best Classes:')
    for name, score in class_accs[:10]:
        print(f"  {name:30} {score:.2%}")

    print('\nTop 10 Worst Classes:')
    for name, score in class_accs[-10:]:
        print(f"  {name:30} {score:.2%}")


def visualize_predictions(model: nn.Module, loader: DataLoader, classes, num_images: int = 10, out_path='predictions_hqa.png'):
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = outputs.max(1)

    indices = np.random.choice(len(inputs), min(num_images, len(inputs)), replace=False)
    fig = plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(2, 5, i + 1)
        img = inputs[idx].cpu() * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        pred_label = classes[preds[idx].item()]
        true_label = classes[targets[idx].item()]
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f"P: {pred_label}\nT: {true_label}", color=color, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved predictions grid to {out_path}")


# ---------------------------
# Grad-CAM / Heatmap helpers
# ---------------------------
def generate_gradcam_for_image(model: nn.Module, input_tensor: torch.Tensor, target_class: int = None,
                                layer_module=None, device: torch.device = DEVICE):
    """Generate a Grad-CAM heatmap for a single input image.

    - `layer_module` defaults to the patch embedding conv layer (`model.patch_embed.proj`).
    - Returns: (heatmap_np, overlay_np) where both are HxW x 3 arrays in [0,1].
    """
    model = model.to(device)
    model.eval()

    if layer_module is None:
        # default to patch projection conv (spatial map aligned to patches)
        layer_module = getattr(model, 'patch_embed', None)
        if layer_module is not None and hasattr(layer_module, 'proj'):
            layer_module = layer_module.proj
        else:
            raise RuntimeError('Unable to locate default conv layer for Grad-CAM on the model')

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        # Save activation and register hook to capture gradients on backward
        activations['value'] = out

        def save_grad(grad):
            gradients['value'] = grad

        out.register_hook(save_grad)

    handle = layer_module.register_forward_hook(forward_hook)

    # forward
    input_batch = input_tensor.unsqueeze(0).to(device)
    outputs = model(input_batch)

    if target_class is None:
        pred = outputs.argmax(dim=1).item()
    else:
        pred = int(target_class)

    # backward on the predicted class score
    model.zero_grad()
    score = outputs[0, pred]
    score.backward(retain_graph=False)

    # get activation and gradient
    handle.remove()

    if 'value' not in activations or 'value' not in gradients:
        raise RuntimeError('Failed to capture activations or gradients for Grad-CAM')

    act = activations['value'].detach()  # [1, C, h, w]
    grad = gradients['value'].detach()   # [1, C, h, w]

    # compute channel-wise weights
    weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # [1,1,h,w]

    # Normalize cam
    cam = cam.squeeze(0).squeeze(0)  # [h, w]
    cam = cam.cpu()
    if cam.max() == cam.min():
        cam_norm = torch.zeros_like(cam)
    else:
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min())

    # Upsample to input image size
    H_in = input_tensor.shape[1]
    W_in = input_tensor.shape[2]
    cam_tensor = cam_norm.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    cam_up = F.interpolate(cam_tensor, size=(H_in, W_in), mode='bilinear', align_corners=False)
    cam_up = cam_up.squeeze().numpy()

    # Map to color using colormap
    cmap = cm.get_cmap('jet')
    cam_rgb = cmap(cam_up)[:, :, :3]

    # Un-normalize image for overlay
    mean = np.array(MEAN).reshape(1, 1, 3)
    std = np.array(STD).reshape(1, 1, 3)
    img_np = input_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)

    # Overlay
    alpha = 0.5
    overlay = (1.0 - alpha) * img_np + alpha * cam_rgb
    overlay = np.clip(overlay, 0, 1)

    return cam_rgb, overlay


def gradcam_on_loader(model: nn.Module, loader: DataLoader, classes, out_dir: str = 'gradcam',
                      num_images: int = 8, device: torch.device = DEVICE):
    os.makedirs(out_dir, exist_ok=True)

    # take first batch and run Grad-CAM on a subset
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = outputs.max(1)

    # choose indices: mix of correct/incorrect randomly
    indices = np.random.choice(len(inputs), min(num_images, len(inputs)), replace=False)

    for idx in indices:
        inp = inputs[idx].cpu()
        true = targets[idx].item()
        pred = preds[idx].item()

        try:
            heat, overlay = generate_gradcam_for_image(model, inp, target_class=pred, layer_module=None, device=device)
        except Exception as e:
            print(f"Grad-CAM failed for idx {idx}: {e}")
            continue

        # save heatmap and overlay
        heat_path = os.path.join(out_dir, f'gradcam_{idx}_pred{pred}_true{true}_heat.png')
        overlay_path = os.path.join(out_dir, f'gradcam_{idx}_pred{pred}_true{true}_overlay.png')

        plt.imsave(heat_path, heat)
        plt.imsave(overlay_path, overlay)
        print(f"Saved Grad-CAM images for idx {idx} -> {overlay_path}")


def smoke_test_forward_pass(device: torch.device = DEVICE):
    """Quick smoke test: instantiate model and run a forward pass with dummy input."""
    cfg = HQAViTConfig()
    model = HQAViT(cfg).to(device)
    model.eval()

    with torch.no_grad():
        dummy = torch.randn(2, cfg.in_channels, cfg.img_size, cfg.img_size, device=device)
        out = model(dummy)

    assert out.shape[0] == 2 and out.shape[1] == cfg.num_classes, f"Unexpected output shape: {out.shape}"
    print(f"Smoke test passed — output shape: {out.shape}")


def main(run_eval: bool = True, run_smoke: bool = True, run_gradcam: bool = True):
    if run_smoke:
        smoke_test_forward_pass(DEVICE)

    if not run_eval:
        return

    loader, classes = get_cifar100_loader(DATA_ROOT, BATCH_SIZE)
    model = load_model(CHECKPOINT_PATH, DEVICE)
    preds, targets = evaluate_model(model, loader, DEVICE)
    analyze_class_performance(preds, targets, classes)
    visualize_predictions(model, loader, classes)
    plot_confusion_matrix(preds, targets, classes)

    if run_gradcam:
        gradcam_on_loader(model, loader, classes, out_dir='gradcam_hqa', num_images=8, device=DEVICE)


if __name__ == "__main__":
    main()
