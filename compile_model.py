"""
QAViT Model Compilation & Specifications
Shows model architecture, parameters, and compiles for optimization
"""

# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import torch
from QAViT import qavit_tiny, qavit_small


def compile_and_show_specs(model_size='tiny'):
    """Compile model and show detailed specifications"""
    
    print("="*80)
    print(f"QAViT-{model_size.upper()} Model Compilation & Specifications")
    print("="*80)
    
    # Create model
    if model_size == 'tiny':
        model = qavit_tiny(num_classes=100)
    else:
        model = qavit_small(num_classes=100)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nDevice: {device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'Parameter Statistics':-^80}")
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Non-trainable:         {total_params - trainable_params:,}")
    print(f"Memory (FP32):         {total_params * 4 / 1024**2:.2f} MB")
    print(f"Memory (FP16/AMP):     {total_params * 2 / 1024**2:.2f} MB")
    
    # Layer breakdown
    print(f"\n{'Layer Breakdown':-^80}")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name:20s}: {num_params:>12,} parameters")
    
    # Architecture details
    print(f"\n{'Architecture Details':-^80}")
    config = model.config if hasattr(model, 'config') else None
    if config:
        print(f"Embedding Dimension (d):     {config.embed_dim}")
        print(f"Compressed Dimension (d'):   {config.embed_dim // config.compress_ratio}")
        print(f"Bottleneck Dimension (r):    {config.embed_dim // config.bottleneck_ratio}")
        print(f"Number of Heads:             {config.num_heads}")
        print(f"Transformer Depth:           {config.depth}")
        print(f"MLP Ratio:                   {config.mlp_ratio}")
        print(f"Global Bank Size:            {config.global_bank_size}")
        print(f"Window Size (SWA):           {config.window_size}")
        print(f"Channel Groups (CGA):        {config.num_channel_groups}")
    
    # Test forward pass
    print(f"\n{'Forward Pass Test':-^80}")
    model.eval()
    batch_sizes = [1, 8, 32, 128] if device.type == 'cuda' else [1, 8]
    
    for bs in batch_sizes:
        try:
            x = torch.randn(bs, 3, 224, 224, device=device)
            
            with torch.no_grad():
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record()
                    output = model(x)
                    end.record()
                    
                    torch.cuda.synchronize()
                    elapsed = start.elapsed_time(end) / 1000  # ms to seconds
                else:
                    import time
                    start = time.time()
                    output = model(x)
                    elapsed = time.time() - start
            
            throughput = bs / elapsed
            
            if device.type == 'cuda':
                vram = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Batch {bs:3d}: {output.shape} | {elapsed*1000:6.2f}ms | {throughput:6.1f} img/s | {vram:.2f}GB VRAM")
                torch.cuda.reset_peak_memory_stats()
            else:
                print(f"Batch {bs:3d}: {output.shape} | {elapsed*1000:6.2f}ms | {throughput:6.1f} img/s")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch {bs:3d}: Out of memory")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                print(f"Batch {bs:3d}: Error - {str(e)[:50]}")
    
    # Compile model (PyTorch 2.0+)
    print(f"\n{'Model Compilation (torch.compile)':-^80}")
    
    try:
        # Check PyTorch version
        pytorch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        
        if pytorch_version >= (2, 0):
            print("Compiling model with torch.compile()...")
            print("Mode: 'default' (balanced speed/memory)")
            
            # Compile the model
            compiled_model = torch.compile(model, mode='default')
            
            print("[OK] Model compiled successfully!")
            print("\nCompilation benefits:")
            print("  - Fused operations for faster execution")
            print("  - Reduced memory overhead")
            print("  - Optimized for your hardware")
            print("  - Expected speedup: 1.2-2x depending on GPU")
            
            # Test compiled model
            print("\nTesting compiled model...")
            x = torch.randn(32, 3, 224, 224, device=device)
            
            with torch.no_grad():
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record()
                    output = compiled_model(x)
                    end.record()
                    
                    torch.cuda.synchronize()
                    elapsed = start.elapsed_time(end) / 1000
                else:
                    import time
                    start = time.time()
                    output = compiled_model(x)
                    elapsed = time.time() - start
            
            throughput = 32 / elapsed
            print(f"[OK] Compiled forward pass: {elapsed*1000:.2f}ms | {throughput:.1f} img/s")
            
            return compiled_model
            
        else:
            print(f"[INFO] PyTorch version {torch.__version__} detected")
            print("[INFO] torch.compile() requires PyTorch 2.0+")
            print("[INFO] Model will run in eager mode (still fast!)")
            return model
            
    except Exception as e:
        print(f"[WARNING] Compilation failed: {e}")
        print("[INFO] Model will run in eager mode")
        return model


def main():
    import sys
    
    model_size = sys.argv[1] if len(sys.argv) > 1 else 'tiny'
    
    if model_size not in ['tiny', 'small']:
        print("Usage: python compile_model.py [tiny|small]")
        print("Defaulting to 'tiny'")
        model_size = 'tiny'
    
    compiled_model = compile_and_show_specs(model_size)
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"\nModel: QAViT-{model_size.upper()}")
    print(f"Status: Ready for training")
    print(f"Compilation: {'Enabled' if hasattr(compiled_model, '_orig_mod') else 'Eager mode'}")
    
    print("\nNext steps:")
    print("1. Run validation: python test_qavit.py")
    print("2. Start training: python train.py --model tiny --batch-size 128 --amp")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
