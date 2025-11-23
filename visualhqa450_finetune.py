import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12

def parse_finetune_log(file_path):
    """Parses the HQA-ViT Fine-Tuning log format."""
    
    data = []
    
    # Regex patterns
    epoch_header_pattern = re.compile(r"EPOCH\s+(\d+)/50\s+SUMMARY")
    # Pattern to catch the LR from the progress lines (e.g., "LR: 0.0000025")
    lr_line_pattern = re.compile(r"LR:\s+([\d\.e-]+)")
    
    # Summary block patterns
    # Loss: Train, Val, EMA
    loss_pattern = re.compile(r"Loss\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)")
    # Acc: Train, Val, EMA, TTA, TTA+EMA
    acc_pattern = re.compile(r"Accuracy \(%\)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)")
    
    current_lr = 0.0
    current_epoch = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return pd.DataFrame()

    # Temporary storage for the current epoch's data
    epoch_data = {}

    for line in lines:
        # 1. Track Learning Rate from progress lines
        lr_match = lr_line_pattern.search(line)
        if lr_match:
            current_lr = float(lr_match.group(1))

        # 2. Detect Epoch Summary Start
        epoch_match = epoch_header_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epoch_data = {'Epoch': current_epoch, 'LR': current_lr}
            continue

        # 3. Parse Metrics inside summary
        if current_epoch is not None:
            # Parse Loss (3 columns)
            loss_match = loss_pattern.search(line)
            if loss_match:
                epoch_data['Train Loss'] = float(loss_match.group(1))
                epoch_data['Val Loss'] = float(loss_match.group(2))
                epoch_data['EMA Loss'] = float(loss_match.group(3))
                continue

            # Parse Accuracy (5 columns)
            acc_match = acc_pattern.search(line)
            if acc_match:
                epoch_data['Train Acc'] = float(acc_match.group(1))
                epoch_data['Val Acc'] = float(acc_match.group(2))
                epoch_data['EMA Acc'] = float(acc_match.group(3))
                epoch_data['TTA Acc'] = float(acc_match.group(4))
                epoch_data['TTA+EMA Acc'] = float(acc_match.group(5))
                
                # Once we have accuracy, the block is mostly done, append to list
                data.append(epoch_data)
                current_epoch = None # Reset
                continue

    return pd.DataFrame(data)

def plot_finetuning_results(df):
    if df.empty:
        print("No data to plot.")
        return

    # Create a 2x2 grid
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    
    # --- Plot 1: Accuracy Comparison (The most important one) ---
    ax1 = axes[0, 0]
    # Plot lines
    ax1.plot(df['Epoch'], df['Val Acc'], label='Val Acc', color='#1f77b4', linestyle=':', alpha=0.7)
    ax1.plot(df['Epoch'], df['EMA Acc'], label='EMA Acc', color='#ff7f0e', linestyle='--')
    ax1.plot(df['Epoch'], df['TTA Acc'], label='TTA Acc', color='#2ca02c', alpha=0.6)
    ax1.plot(df['Epoch'], df['TTA+EMA Acc'], label='TTA+EMA (Best)', color='#d62728', linewidth=2.5)
    
    # Annotate the absolute best accuracy
    best_row = df.loc[df['TTA+EMA Acc'].idxmax()]
    ax1.scatter(best_row['Epoch'], best_row['TTA+EMA Acc'], color='black', s=50, zorder=5)
    ax1.annotate(f"Peak: {best_row['TTA+EMA Acc']}%", 
                 (best_row['Epoch'], best_row['TTA+EMA Acc']),
                 xytext=(0, 10), textcoords='offset points', ha='center', fontweight='bold')

    ax1.set_title('Fine-Tuning Accuracy Trajectories', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # --- Plot 2: Loss Convergence ---
    ax2 = axes[0, 1]
    ax2.plot(df['Epoch'], df['Train Loss'], label='Train Loss', color='gray', alpha=0.5)
    ax2.plot(df['Epoch'], df['Val Loss'], label='Val Loss', color='#1f77b4')
    ax2.plot(df['Epoch'], df['EMA Loss'], label='EMA Val Loss', color='#ff7f0e', linestyle='--')
    
    ax2.set_title('Loss Convergence (Regularization Check)', fontweight='bold')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    # --- Plot 3: Learning Rate Schedule ---
    ax3 = axes[1, 0]
    ax3.plot(df['Epoch'], df['LR'], color='#9467bd')
    ax3.set_title('Learning Rate Decay', fontweight='bold')
    ax3.set_ylabel('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) # Scientific notation for small LR

    # --- Plot 4: The "TTA Boost" Analysis ---
    # Calculate how much TTA adds over standard EMA
    ax4 = axes[1, 1]
    tta_gain = df['TTA+EMA Acc'] - df['EMA Acc']
    
    # Color bars based on positive/negative gain (though likely all positive)
    colors = ['#2ca02c' if x >= 0 else '#d62728' for x in tta_gain]
    
    ax4.bar(df['Epoch'], tta_gain, color=colors, alpha=0.7)
    ax4.axhline(0, color='black', linewidth=0.8)
    ax4.set_title('Performance Gain from TTA (TTA+EMA vs EMA)', fontweight='bold')
    ax4.set_ylabel('Accuracy Gain (%)')
    ax4.set_xlabel('Epoch')

    # Add text summary
    plt.suptitle(f"HQA-ViT CIFAR-100 Fine-Tuning Analysis\nPretrained: 72.65% -> Final Best: {df['TTA+EMA Acc'].max()}%", fontsize=18, y=1.05)

    print(f"Plot generated. Max TTA+EMA Accuracy: {df['TTA+EMA Acc'].max()}% at Epoch {int(best_row['Epoch'])}")
    plt.show()

if __name__ == "__main__":
    # 1. Save your log content to 'finetune_log.txt'
    log_filename = 'log hqavit. finetunetxt.txt'
    
    df = parse_finetune_log(log_filename)
    
    if not df.empty:
        # Optional: Print first few rows to verify parsing
        print(df.head())
        plot_finetuning_results(df)
    else:
        print("Could not parse data. Please ensure 'finetune_log.txt' exists and contains the log output.")