import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Set the style for professional looking plots
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (18, 10)
plt.rcParams['lines.linewidth'] = 1.5

def parse_log(file_path):
    """Parses the HQA-ViT training log format."""
    
    data = []
    
    # Regex patterns based on your log format
    epoch_pattern = re.compile(r"EPOCH\s+(\d+)/450\s+SUMMARY")
    loss_pattern = re.compile(r"Loss\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)")
    acc_pattern = re.compile(r"Accuracy \(%\)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)")
    lr_pattern = re.compile(r"Learning Rate\s+([\d\.]+)")
    grad_pattern = re.compile(r"Gradient Norm\s+([\d\.]+)")
    ema_dist_pattern = re.compile(r"EMA Param Distance\s+([\d\.]+)")

    current_epoch = None
    current_data = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please save your log text to this file.")
        return pd.DataFrame()

    for line in lines:
        # Detect Epoch Start
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            # Save previous epoch data if exists
            if current_epoch is not None and current_data:
                current_data['Epoch'] = int(current_epoch)
                data.append(current_data)
            
            current_epoch = epoch_match.group(1)
            current_data = {}
            continue

        # Parse Metrics inside the summary block
        if current_epoch:
            loss_match = loss_pattern.search(line)
            if loss_match:
                current_data['Train Loss'] = float(loss_match.group(1))
                current_data['Val Loss'] = float(loss_match.group(2))
                current_data['EMA Val Loss'] = float(loss_match.group(3))
                continue

            acc_match = acc_pattern.search(line)
            if acc_match:
                current_data['Train Acc'] = float(acc_match.group(1))
                current_data['Val Acc'] = float(acc_match.group(2))
                current_data['EMA Val Acc'] = float(acc_match.group(3))
                continue

            lr_match = lr_pattern.search(line)
            if lr_match:
                current_data['Learning Rate'] = float(lr_match.group(1))
                continue

            grad_match = grad_pattern.search(line)
            if grad_match:
                current_data['Gradient Norm'] = float(grad_match.group(1))
                continue
                
            ema_match = ema_dist_pattern.search(line)
            if ema_match:
                current_data['EMA Param Dist'] = float(ema_match.group(1))
                continue

    # Append the last epoch
    if current_epoch is not None and current_data:
        current_data['Epoch'] = int(current_epoch)
        data.append(current_data)

    return pd.DataFrame(data)

def plot_training_results(df):
    if df.empty:
        return

    # Create a 2x3 grid layout
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # 1. Accuracy Plot (Spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df['Epoch'], df['Train Acc'], label='Train Acc', color='gray', alpha=0.4)
    ax1.plot(df['Epoch'], df['Val Acc'], label='Val Acc', color='#1f77b4')
    ax1.plot(df['Epoch'], df['EMA Val Acc'], label='EMA Val Acc', color='#ff7f0e', linestyle='--')
    
    # Highlight Best Accuracy
    best_ema_epoch = df.loc[df['EMA Val Acc'].idxmax()]
    ax1.scatter(best_ema_epoch['Epoch'], best_ema_epoch['EMA Val Acc'], color='red', zorder=5)
    ax1.annotate(f"Best EMA: {best_ema_epoch['EMA Val Acc']}%", 
                 (best_ema_epoch['Epoch'], best_ema_epoch['EMA Val Acc']),
                 xytext=(10, -20), textcoords='offset points', color='red', fontweight='bold')
    
    ax1.set_title('Accuracy Progression', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend(loc='lower right')

    # 2. Loss Plot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(df['Epoch'], df['Train Loss'], label='Train', color='gray', alpha=0.5)
    ax2.plot(df['Epoch'], df['Val Loss'], label='Val', color='#d62728')
    ax2.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # 3. Learning Rate
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['Epoch'], df['Learning Rate'], color='#9467bd')
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_ylabel('LR')
    ax3.set_xlabel('Epoch')

    # 4. Gradient Norm
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['Epoch'], df['Gradient Norm'], color='#2ca02c', linewidth=1)
    ax4.set_title('Gradient Norm (Stability)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Norm')
    ax4.set_xlabel('Epoch')

    # 5. EMA Parameter Distance
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(df['Epoch'], df['EMA Param Dist'], color='#8c564b')
    ax5.set_title('EMA Parameter Distance', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Distance')
    ax5.set_xlabel('Epoch')

    # Final Layout Adjustments
    plt.suptitle(f"HQA-ViT Training Log Analysis (Total Epochs: {len(df)})", fontsize=18, y=1.05)
    
    print(f"Plot generated. Best EMA Accuracy: {best_ema_epoch['EMA Val Acc']}% at Epoch {int(best_ema_epoch['Epoch'])}")
    plt.show()

if __name__ == "__main__":
    # 1. Create a dummy file for demonstration if you run this directly
    #    In practice, replace 'training_log.txt' with your actual log file path.
    # Default to the attached log file; allow overriding via CLI argument
    default_log = 'log hqavit450.txt'
    log_filename = sys.argv[1] if len(sys.argv) > 1 else default_log

    # Check if file exists, if not, print helpful message
    try:
        df = parse_log(log_filename)
        if not df.empty:
            plot_training_results(df)
        else:
            print(f"No data found in log file '{log_filename}'. Please ensure it contains the training log output.")
    except FileNotFoundError:
        print(f"Error: File '{log_filename}' not found. Please provide a training log file as the first argument.")
    except Exception as e:
        print(f"An error occurred while parsing '{log_filename}': {e}")