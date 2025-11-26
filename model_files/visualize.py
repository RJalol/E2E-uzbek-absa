import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import argparse


# Usage: python model_files/visualize.py -log_dir snapshot/YYYY-MM-DD...

def plot_training_history(csv_path, save_dir):
    """Draws Training vs Validation Loss and Accuracy curves."""
    if not os.path.exists(csv_path):
        print(f"Log file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    print(f"Saved loss_curve.png to {save_dir}")
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='o')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    print(f"Saved accuracy_curve.png to {save_dir}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_dir):
    """Draws a professional Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    print(f"Saved confusion_matrix.png to {save_dir}")
    plt.close()


def visualize_embeddings(model_path, vocab_path, save_dir, max_words=1000):
    """Visualizes Word Embeddings using t-SNE."""
    # This requires loading the model structure.
    # Since we are doing this offline, we act as if we have the embedding weights.
    # For now, this is a placeholder to show how you WOULD do it if you loaded the model object.
    print("Embedding visualization requires the full model object loaded in memory.")
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log_dir', type=str, required=True,
                        help='Path to the snapshot directory containing training_log.csv')
    args = parser.parse_args()

    csv_path = os.path.join(args.log_dir, 'training_log.csv')
    plot_training_history(csv_path, args.log_dir)