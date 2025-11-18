"""
QAViT Evaluation Script
Evaluate a trained QAViT model on test/validation data
"""

import torch
import torch.nn as nn
import argparse
import time
from tqdm import tqdm

from QAViT import (
    qavit_tiny, qavit_small,
    build_dataloader, DATASET_CONFIGS
)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on dataset"""
    model.eval()
    
    total_loss = 0.0
    total_correct_1 = 0
    total_correct_5 = 0
    total_samples = 0
    
    inference_times = []
    
    with torch.no_grad():
        for images, target in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Measure inference time
            start = time.time()
            output = model(images)
            inference_times.append(time.time() - start)
            
            # Compute loss
            loss = criterion(output, target)
            total_loss += loss.item() * images.size(0)
            
            # Compute accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            total_correct_1 += acc1[0].item() * images.size(0) / 100.0
            total_correct_5 += acc5[0].item() * images.size(0) / 100.0
            total_samples += images.size(0)
    
    avg_loss = total_loss / total_samples
    avg_acc1 = (total_correct_1 / total_samples) * 100
    avg_acc5 = (total_correct_5 / total_samples) * 100
    avg_time = sum(inference_times) / len(inference_times)
    throughput = data_loader.batch_size / avg_time
    
    return {
        'loss': avg_loss,
        'acc@1': avg_acc1,
        'acc@5': avg_acc5,
        'avg_inference_time': avg_time,
        'throughput': throughput,
        'total_samples': total_samples
    }


def main():
    parser = argparse.ArgumentParser(description='QAViT Evaluation')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='tiny', 
                        choices=['tiny', 'small'],
                        help='Model size: tiny or small')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'imagenet'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Path to dataset')
    
    # Evaluation settings
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    # Create model
    print(f"{'='*80}")
    print(f"Loading QAViT-{args.model.upper()} model...")
    print(f"{'='*80}\n")
    
    if args.model == 'tiny':
        model = qavit_tiny(num_classes=100 if args.dataset == 'cifar100' else 1000)
    else:
        model = qavit_small(num_classes=100 if args.dataset == 'cifar100' else 1000)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"Checkpoint from epoch: {checkpoint['epoch']}")
        if 'best_acc1' in checkpoint:
            print(f"Checkpoint best accuracy: {checkpoint['best_acc1']:.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params:,}\n")
    
    # Create dataloader
    print(f"Loading {args.dataset.upper()} dataset...")
    import copy
    dataset_config = copy.copy(DATASET_CONFIGS[args.dataset])
    dataset_config.data_root = args.data_path
    dataset_config.batch_size = args.batch_size
    dataset_config.num_workers = args.num_workers
    val_loader = build_dataloader(dataset_config, is_train=False)
    
    print(f"Evaluation batches: {len(val_loader)}\n")
    
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Evaluate
    print(f"{'='*80}")
    print(f"Starting evaluation...")
    print(f"{'='*80}\n")
    
    results = evaluate(model, val_loader, criterion, device)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Dataset:              {args.dataset.upper()}")
    print(f"Model:                QAViT-{args.model.upper()}")
    print(f"Total Samples:        {results['total_samples']:,}")
    print(f"{'─'*80}")
    print(f"Loss:                 {results['loss']:.4f}")
    print(f"Top-1 Accuracy:       {results['acc@1']:.2f}%")
    print(f"Top-5 Accuracy:       {results['acc@5']:.2f}%")
    print(f"{'─'*80}")
    print(f"Avg Inference Time:   {results['avg_inference_time']*1000:.2f} ms/batch")
    print(f"Throughput:           {results['throughput']:.1f} images/sec")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
