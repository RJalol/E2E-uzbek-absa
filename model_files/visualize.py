import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


# Usage: python model_files/visualize.py -log_dir snapshot/aspect-generators

def plot_training_history(csv_path, save_dir):
    """Draws Training vs Validation curves, automatically detecting metric types."""
    if not os.path.exists(csv_path):
        print(f"Log file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 1. Plot Loss (Common to all models)
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    print(f"[-] Saved loss_curve.png to {save_dir}")
    plt.close()

    # 2. Check for Multitask Columns (Tagging & Category Accuracies)
    if 'train_tag_acc' in df.columns and 'train_cat_acc' in df.columns:
        print("[-] Detected Multi-Task Log. Generating split accuracy plots...")

        # Plot Tagging Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_tag_acc'], label='Train Tag Acc', marker='o', color='purple')
        plt.plot(df['epoch'], df['val_tag_acc'], label='Val Tag Acc', marker='o', color='orange')
        plt.title('Aspect Term Extraction (Tagging) Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'tag_accuracy_curve.png'))
        print(f"[-] Saved tag_accuracy_curve.png to {save_dir}")
        plt.close()

        # Plot Category Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_cat_acc'], label='Train Cat Acc', marker='o', color='green')
        plt.plot(df['epoch'], df['val_cat_acc'], label='Val Cat Acc', marker='o', color='red')
        plt.title('Aspect Category Detection Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'cat_accuracy_curve.png'))
        print(f"[-] Saved cat_accuracy_curve.png to {save_dir}")
        plt.close()

    # 3. Check for Standard Columns (GCAE/Simple Accuracy)
    elif 'train_acc' in df.columns:
        print("[-] Detected Single-Task Log. Generating standard accuracy plot...")
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy', marker='o')
        plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='o')
        plt.title('Training vs Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
        print(f"[-] Saved accuracy_curve.png to {save_dir}")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log_dir', type=str, required=True,
                        help='Path to the snapshot directory containing training_log.csv')
    args = parser.parse_args()

    csv_path = os.path.join(args.log_dir, 'training_log.csv')
    plot_training_history(csv_path, args.log_dir)