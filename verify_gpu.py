"""
GPU Setup Verification for RTX 3060 Laptop (6GB VRAM)
Tests CUDA availability, VRAM capacity, and optimal batch sizes
"""

# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import torch
import sys


def check_cuda_setup():
    """Verify CUDA is available and working"""
    print("="*80)
    print("CUDA Setup Verification")
    print("="*80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA is not available!")
        print("\nPossible solutions:")
        print("1. Install CUDA-enabled PyTorch:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print("2. Update GPU drivers from NVIDIA")
        print("3. Check if GPU is enabled in Device Manager")
        return False
    
    print("\n[OK] CUDA is available")
    
    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[OK] GPU: {gpu_name}")
    
    # VRAM info
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[OK] Total VRAM: {total_vram:.2f} GB")
    
    # Current VRAM usage
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"[OK] Currently allocated: {allocated:.2f} GB")
    print(f"[OK] Currently reserved: {reserved:.2f} GB")
    
    # PyTorch version
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] CUDA version: {torch.version.cuda}")
    
    return True


def test_batch_sizes():
    """Test different batch sizes to find optimal configuration"""
    print("\n" + "="*80)
    print("Testing Batch Sizes for QAViT-Tiny")
    print("="*80)
    
    from QAViT import qavit_tiny
    
    batch_sizes = [256, 192, 128, 96, 64, 48, 32]
    device = torch.device('cuda')
    
    successful_configs = []
    
    for batch_size in batch_sizes:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Create model
            model = qavit_tiny(num_classes=100).to(device)
            model.train()
            
            # Test input
            x = torch.randn(batch_size, 3, 224, 224, device=device)
            
            # Forward pass
            with torch.cuda.amp.autocast():
                output = model(x)
            
            # Backward pass
            loss = output.sum()
            loss.backward()
            
            # Get memory usage
            max_memory = torch.cuda.max_memory_allocated(0) / 1024**3
            
            print(f"\n[OK] Batch size {batch_size:3d}: {max_memory:.2f} GB VRAM")
            successful_configs.append((batch_size, max_memory))
            
            # Clean up
            del model, x, output, loss
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[FAIL] Batch size {batch_size:3d}: Out of memory")
            else:
                print(f"\n[ERROR] Batch size {batch_size:3d}: {str(e)[:50]}...")
            torch.cuda.empty_cache()
    
    # Recommend optimal batch size
    if successful_configs:
        print("\n" + "="*80)
        print("Recommended Configuration")
        print("="*80)
        
        # Find largest safe batch size (< 5.5GB to leave headroom)
        safe_configs = [(bs, mem) for bs, mem in successful_configs if mem < 5.5]
        
        if safe_configs:
            optimal_bs, optimal_mem = max(safe_configs, key=lambda x: x[0])
            print(f"\n[RECOMMENDED] Batch size: {optimal_bs}")
            print(f"              VRAM usage: {optimal_mem:.2f} GB")
            print(f"              Headroom: {6 - optimal_mem:.2f} GB")
            
            print("\nTraining command:")
            print(f"python train.py --model tiny --batch-size {optimal_bs} --amp --device cuda")
        else:
            # All configs use too much memory
            smallest_bs, smallest_mem = min(successful_configs, key=lambda x: x[1])
            print(f"\n[WARNING] All batch sizes use significant VRAM")
            print(f"[RECOMMENDED] Batch size: {smallest_bs} (uses {smallest_mem:.2f} GB)")
            print("              Close background applications before training")


def test_flashattention():
    """Check if FlashAttention2 is available"""
    print("\n" + "="*80)
    print("FlashAttention2 Check")
    print("="*80)
    
    try:
        from flash_attn import flash_attn_func
        print("\n[OK] FlashAttention2 is installed")
        print("[OK] Will use FlashAttention2 for faster training (~1.5-2x speedup)")
        print("\nNote: FlashAttention requires FP16/BF16 (use --amp flag)")
        return True
    except ImportError:
        print("\n[INFO] FlashAttention2 not installed (optional)")
        print("[INFO] Will use PyTorch SDPA (slower but works)")
        print("\nTo install FlashAttention2:")
        print("pip install flash-attn --no-build-isolation")
        return False


def quick_training_test():
    """Run a quick training test (3 steps)"""
    print("\n" + "="*80)
    print("Quick Training Test (3 steps)")
    print("="*80)
    
    from QAViT import qavit_tiny, create_model_and_dataloaders
    import time
    
    device = torch.device('cuda')
    
    try:
        # Create model
        print("\nCreating QAViT-Tiny model...")
        model = qavit_tiny(num_classes=100).to(device)
        model.train()
        
        # Create tiny dataloader
        print("Loading CIFAR-100 (may download on first run)...")
        from QAViT import DatasetConfig, build_dataloader
        config = DatasetConfig(
            dataset_name='cifar100',
            data_root='./data',
            num_classes=100,
            img_size=224,
            batch_size=32,  # Small batch for test
            num_workers=2
        )
        train_loader = build_dataloader(config, is_train=True)
        
        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        
        # Train for 3 steps
        print("\nRunning 3 training steps...")
        times = []
        
        for step, (images, labels) in enumerate(train_loader):
            if step >= 3:
                break
            
            start_time = time.time()
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            step_time = time.time() - start_time
            times.append(step_time)
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(labels).sum().item() / labels.size(0) * 100
            
            # VRAM usage
            vram_used = torch.cuda.max_memory_allocated(0) / 1024**3
            
            print(f"  Step {step+1}/3: Loss={loss.item():.4f}, "
                  f"Acc={accuracy:.1f}%, Time={step_time:.3f}s, "
                  f"VRAM={vram_used:.2f}GB")
        
        avg_time = sum(times) / len(times)
        throughput = 32 / avg_time  # images per second
        
        print(f"\n[OK] Quick test completed!")
        print(f"     Average time per step: {avg_time:.3f}s")
        print(f"     Throughput: {throughput:.1f} images/sec")
        print(f"     Peak VRAM: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
        
        # Estimate full training time
        batches_per_epoch = len(train_loader)
        estimated_epoch_time = avg_time * batches_per_epoch
        estimated_300_epochs = (estimated_epoch_time * 300) / 3600
        
        print(f"\n[ESTIMATE] Full training (300 epochs):")
        print(f"           Time per epoch: ~{estimated_epoch_time:.1f}s")
        print(f"           Total time: ~{estimated_300_epochs:.1f} hours")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("QAViT GPU Setup Verification for RTX 3060 (6GB VRAM)")
    print("="*80 + "\n")
    
    # 1. Check CUDA
    if not check_cuda_setup():
        print("\n[FAILED] Please fix CUDA setup before proceeding")
        sys.exit(1)
    
    # 2. Check FlashAttention
    test_flashattention()
    
    # 3. Test batch sizes
    try:
        test_batch_sizes()
    except Exception as e:
        print(f"\n[ERROR] Batch size test failed: {e}")
    
    # 4. Quick training test
    try:
        success = quick_training_test()
        if not success:
            print("\n[WARNING] Quick training test failed, but validation may still work")
    except Exception as e:
        print(f"\n[ERROR] Training test failed: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("Setup Verification Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run validation: python test_qavit.py")
    print("2. Start training: python train.py --model tiny --batch-size 128 --amp")
    print("3. Monitor with TensorBoard: tensorboard --logdir ./logs")
    print("\nSee train_configs.md for detailed RTX 3060 optimization tips!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
