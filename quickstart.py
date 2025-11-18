"""
QAViT Quick Start Example
Demonstrates basic model creation, training, and inference
"""

import torch
from QAViT import qavit_tiny, qavit_small, create_model_and_dataloaders, build_dataloader, DATASET_CONFIGS

def example_model_creation():
    """Example 1: Creating QAViT models"""
    print("\n" + "="*80)
    print("Example 1: Model Creation")
    print("="*80)
    
    # Create Tiny model
    model_tiny = qavit_tiny(num_classes=100)
    print(f"\n[OK] QAViT-Tiny created")
    print(f"   Parameters: {sum(p.numel() for p in model_tiny.parameters()):,}")
    
    # Create Small model
    model_small = qavit_small(num_classes=100)
    print(f"\n[OK] QAViT-Small created")
    print(f"   Parameters: {sum(p.numel() for p in model_small.parameters()):,}")


def example_forward_pass():
    """Example 2: Forward pass"""
    print("\n" + "="*80)
    print("Example 2: Forward Pass")
    print("="*80)
    
    model = qavit_tiny(num_classes=100)
    model.eval()
    
    # Create dummy input
    x = torch.randn(2, 3, 224, 224)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"[OK] Forward pass successful!")


def example_cifar100_dataloader():
    """Example 3: CIFAR-100 dataloader"""
    print("\n" + "="*80)
    print("Example 3: CIFAR-100 DataLoader")
    print("="*80)
    
    # Create dataloaders
    _, train_loader, val_loader, _ = create_model_and_dataloaders(
        model_size='tiny',
        dataset_name='cifar100'
    )
    
    print(f"\n[OK] DataLoaders created")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch example:")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")


def example_simple_training():
    """Example 4: Simple training loop"""
    print("\n" + "="*80)
    print("Example 4: Simple Training Loop (3 steps, CPU for compatibility)")
    print("="*80)
    print("\nNote: For actual training on RTX 3060, use:")
    print("  python train.py --model tiny --batch-size 128 --amp --device cuda")
    
    # Setup - use CPU for compatibility (GPU would be much faster)
    device = torch.device('cpu')
    print(f"\nUsing device: {device}")
    
    # Model
    model = qavit_tiny(num_classes=100).to(device)
    model.train()
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Dataloader (small batch for CPU example)
    from QAViT import DatasetConfig
    config = DatasetConfig(
        dataset_name='cifar100',
        data_root='./data',
        num_classes=100,
        img_size=224,
        batch_size=4,  # Small batch for CPU demo
        num_workers=0  # No multiprocessing for quick example
    )
    train_loader = build_dataloader(config, is_train=True)
    
    # Training loop (3 steps for quick demo)
    print("\nTraining for 3 steps (demo only)...")
    for step, (images, labels) in enumerate(train_loader):
        if step >= 3:
            break
            
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = correct / labels.size(0) * 100
        
        print(f"   Step {step+1}/3: Loss={loss.item():.4f}, Acc={accuracy:.2f}%")
    
    print("\n[OK] Training loop completed!")
    print("\nFor full training on your RTX 3060 (6GB):")
    print("  - Tiny model: batch_size=128, ~3.5GB VRAM, ~25sec/epoch")
    print("  - Small model: batch_size=64, ~5GB VRAM, ~45sec/epoch")


def example_inference():
    """Example 5: Inference on single image"""
    print("\n" + "="*80)
    print("Example 5: Single Image Inference")
    print("="*80)
    
    # Create model
    model = qavit_tiny(num_classes=100)
    model.eval()
    
    # Dummy image (in practice, load from PIL)
    image = torch.randn(1, 3, 224, 224)
    print(f"\nInput image shape: {image.shape}")
    
    # Inference
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        top5_probs, top5_indices = torch.topk(probs, 5)
    
    print(f"\nTop-5 predictions:")
    for i in range(5):
        print(f"   {i+1}. Class {top5_indices[0][i].item()}: {top5_probs[0][i].item()*100:.2f}%")
    
    print("\n[OK] Inference completed!")


def example_save_load_checkpoint():
    """Example 6: Save and load checkpoint"""
    print("\n" + "="*80)
    print("Example 6: Save & Load Checkpoint")
    print("="*80)
    
    # Create model
    model = qavit_tiny(num_classes=100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Save checkpoint
    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc1': 75.5
    }
    
    torch.save(checkpoint, 'example_checkpoint.pth')
    print("\n[OK] Checkpoint saved: example_checkpoint.pth")
    
    # Load checkpoint
    model_new = qavit_tiny(num_classes=100)
    checkpoint_loaded = torch.load('example_checkpoint.pth')
    model_new.load_state_dict(checkpoint_loaded['model_state_dict'])
    
    print(f"[OK] Checkpoint loaded")
    print(f"   Epoch: {checkpoint_loaded['epoch']}")
    print(f"   Best Acc@1: {checkpoint_loaded['best_acc1']:.2f}%")
    
    # Cleanup
    import os
    os.remove('example_checkpoint.pth')


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("                   QAViT Quick Start Examples")
    print("="*80)
    
    try:
        example_model_creation()
        example_forward_pass()
        example_cifar100_dataloader()
        example_simple_training()
        example_inference()
        example_save_load_checkpoint()
        
        print("\n" + "="*80)
        print("                   All Examples Completed! [OK][OK]")
        print("="*80)
        print("\nNext steps:")
        print("1. Run full validation: python test_qavit.py")
        print("2. Start training: python train.py --model tiny --dataset cifar100")
        print("3. Evaluate model: python evaluate.py --model tiny --checkpoint <path>")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n[OK] Error: {e}")
        print("\nMake sure you have:")
        print("1. Installed PyTorch: pip install torch torchvision")
        print("2. CIFAR-100 data will download automatically on first run")


if __name__ == '__main__':
    main()

