# QAViT Training Configurations for RTX 3060 Laptop (6GB VRAM)

**Hardware**: Ryzen 7 6800H, RTX 3060 Mobile 6GB, 16GB DDR5 RAM

## Recommended Training Configurations

### QAViT-Tiny (2.47M parameters) - RECOMMENDED

**Full Training (300 epochs):**
```bash
python train.py \
    --model tiny \
    --dataset cifar100 \
    --epochs 300 \
    --batch-size 128 \
    --lr 1e-3 \
    --weight-decay 0.05 \
    --warmup-epochs 20 \
    --amp \
    --num-workers 4 \
    --device cuda
```

**Quick Training (100 epochs for testing):**
```bash
python train.py \
    --model tiny \
    --dataset cifar100 \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-3 \
    --weight-decay 0.05 \
    --warmup-epochs 10 \
    --amp \
    --num-workers 4
```

**Expected Performance:**
- VRAM Usage: ~3-4 GB with AMP
- Training Speed: ~8-10 batches/sec
- Time per epoch: ~25-30 seconds
- Total time (300 epochs): ~2.5-3 hours
- Expected accuracy: 75-80% top-1 on CIFAR-100

---

### QAViT-Small (5.29M parameters) - POSSIBLE

**Conservative Settings:**
```bash
python train.py \
    --model small \
    --dataset cifar100 \
    --epochs 300 \
    --batch-size 64 \
    --lr 8e-4 \
    --weight-decay 0.05 \
    --warmup-epochs 20 \
    --amp \
    --num-workers 4 \
    --grad-clip 1.0
```

**Expected Performance:**
- VRAM Usage: ~5-5.5 GB with AMP (close to limit!)
- Training Speed: ~5-6 batches/sec
- Time per epoch: ~40-50 seconds
- Total time (300 epochs): ~4-5 hours
- Expected accuracy: 78-82% top-1 on CIFAR-100

**⚠️ Warning**: Small model may be tight on 6GB VRAM. If you get OOM errors:
- Reduce batch size to 48 or 32
- Close all other applications
- Monitor GPU with: `nvidia-smi -l 1`

---

## Batch Size Guidelines

| Model | Batch Size | VRAM Usage | Speed | Notes |
|-------|------------|------------|-------|-------|
| Tiny | 256 | ~5.5 GB | Fast | May OOM occasionally |
| Tiny | 128 | ~3.5 GB | ✅ RECOMMENDED | Safe, good speed |
| Tiny | 64 | ~2 GB | Slower | Very safe, use for debugging |
| Small | 64 | ~5 GB | ✅ RECOMMENDED | Close to limit |
| Small | 48 | ~4 GB | Safer | Use if 64 OOMs |
| Small | 32 | ~3 GB | Safe | Slower but stable |

## Memory Optimization Tips

### 1. Enable Mixed Precision (AMP)
Always use `--amp` flag - reduces VRAM by ~40% with minimal accuracy loss.

### 2. Reduce Gradient Accumulation (if needed)
If batch size is too small, simulate larger batches:
```bash
# Effective batch size = 32 * 4 = 128
python train.py --batch-size 32 --gradient-accumulation-steps 4
```

### 3. Monitor VRAM Usage
```bash
# In separate terminal
watch -n 1 nvidia-smi

# Or Windows PowerShell
while($true) { nvidia-smi; sleep 1; clear }
```

### 4. Close Background Apps
- Close browser tabs
- Close Discord, Spotify, etc.
- Disable Windows hardware acceleration in browsers

### 5. Set PyTorch Memory Management
Add to train.py at top:
```python
import torch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  # Speed optimization
```

---

## Quick Start Guide

### Step 1: Verify Setup
```bash
# Test that CUDA works
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Run validation suite
python test_qavit.py
```

### Step 2: Quick Training Test (10 epochs)
```bash
python train.py \
    --model tiny \
    --dataset cifar100 \
    --epochs 10 \
    --batch-size 128 \
    --amp \
    --save-freq 5
```

This will:
- Download CIFAR-100 (if not already)
- Train for 10 epochs (~5 minutes)
- Save checkpoints to verify everything works
- Use ~3.5GB VRAM

### Step 3: Full Training
Once verified, start full training:
```bash
python train.py \
    --model tiny \
    --dataset cifar100 \
    --epochs 300 \
    --batch-size 128 \
    --amp
```

### Step 4: Monitor Progress
```bash
# In separate terminal
tensorboard --logdir ./logs
```
Open browser to http://localhost:6006

---

## Expected Timeline

### QAViT-Tiny on CIFAR-100 (300 epochs)
- **Setup**: 5 minutes (first run, downloads dataset)
- **Training**: ~2.5-3 hours
- **Total**: ~3 hours

### Milestones:
- Epoch 50: ~60-65% accuracy
- Epoch 100: ~70-73% accuracy
- Epoch 200: ~75-77% accuracy
- Epoch 300: ~78-80% accuracy (expected final)

### QAViT-Small on CIFAR-100 (300 epochs)
- **Training**: ~4-5 hours
- **Expected final**: ~80-82% accuracy

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Solution 1: Reduce batch size
python train.py --batch-size 64  # or 48, 32

# Solution 2: Clear cache and retry
python -c "import torch; torch.cuda.empty_cache()"

# Solution 3: Reduce model size
python train.py --model tiny --batch-size 128
```

### "Training is slow"
```bash
# Check GPU utilization
nvidia-smi

# If GPU usage is low:
# - Increase num_workers: --num-workers 6
# - Ensure data is on SSD, not HDD
# - Check CPU isn't bottlenecked
```

### "Accuracy not improving"
```bash
# Check learning rate schedule
tensorboard --logdir ./logs

# Try different learning rates
python train.py --lr 5e-4  # or 2e-3
```

### "FlashAttention errors"
```bash
# Use FP16 for FlashAttention
python train.py --amp

# Or FlashAttention will auto-fallback to SDPA (slower but works)
```

---

## Performance Benchmarks (Estimated)

### Your Hardware (RTX 3060 Mobile 6GB)
| Model | Batch Size | Time/Epoch | Images/Sec | VRAM |
|-------|------------|------------|------------|------|
| Tiny | 128 | ~25s | ~2000 | 3.5GB |
| Tiny | 256 | ~18s | ~2800 | 5.5GB |
| Small | 64 | ~45s | ~1100 | 5GB |
| Small | 48 | ~55s | ~900 | 4GB |

### Comparison: Desktop RTX 3090 (24GB)
| Model | Batch Size | Time/Epoch | Speedup |
|-------|------------|------------|---------|
| Tiny | 512 | ~12s | 2.1× |
| Small | 256 | ~25s | 1.8× |

**Your 3060 is perfectly adequate for this research!** The smaller VRAM just means slightly smaller batch sizes, but training time is still very reasonable for a CVPR paper.

---

## Optimal Configuration Summary

**For fastest iteration & development:**
```bash
# Use Tiny model, batch size 128, AMP enabled
python train.py --model tiny --batch-size 128 --amp --epochs 300
```

**For best accuracy (if VRAM allows):**
```bash
# Use Small model, batch size 64, AMP enabled
python train.py --model small --batch-size 64 --amp --epochs 300
```

**For experimentation:**
```bash
# Quick 50-epoch runs to test hyperparameters
python train.py --model tiny --batch-size 128 --epochs 50 --amp
```

---

## Next Steps

1. ✅ Run validation: `python test_qavit.py`
2. ✅ Test 10-epoch training: `python train.py --epochs 10 --batch-size 128 --amp`
3. ✅ Start full training: `python train.py --epochs 300 --batch-size 128 --amp`
4. 📊 Monitor with TensorBoard: `tensorboard --logdir ./logs`
5. 🎯 Expected result: ~78-80% CIFAR-100 accuracy for CVPR submission

Good luck with your paper! 🚀
