import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import new modules
from model_ate import BiLSTM_ATE
from dataset_ate import ATEDataset, collate_fn
from w2v import load_fasttext_embedding  # Assuming w2v.py is available


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # 1. Define Tag Set (Must match your data labels)
    # 0 is reserved for padding in collate_fn
    tag2idx = {'<pad>': 0, 'O': 1, 'B-ASP': 2, 'I-ASP': 3}

    # 2. Load Vocab & Embeddings
    print("[-] Loading Embeddings...")
    # Simple vocab builder fallback
    word2idx = {'<pad>': 0, '<unk>': 1}
    vectors = []
    # Add dummy vectors for pad/unk
    vectors.append(np.zeros(300))
    vectors.append(np.random.uniform(-0.1, 0.1, 300))

    try:
        # Try to load real embeddings if file exists
        with open(args.embed_file, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.strip().split()
                w = parts[0]
                v = np.array([float(x) for x in parts[1:]])
                word2idx[w] = len(word2idx)
                vectors.append(v)
        vectors = np.array(vectors)
    except:
        print("[!] Warning: Embedding file not found or error. Using random initialization.")
        # Create random embeddings for a small vocab just to allow running
        vectors = np.random.uniform(-0.1, 0.1, (len(word2idx) + 1000, 300))

    # 3. Load Data
    train_dataset = ATEDataset(args.train_path, word2idx, tag2idx)
    test_dataset = ATEDataset(args.test_path, word2idx, tag2idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Initialize Model
    print(f"[-] Initializing Model on {device}...")
    model_args = argparse.Namespace(
        vocab_size=len(word2idx) + 1000,  # Buffer for unknown
        embed_dim=300,
        hidden_dim=args.hidden_dim,
        num_tags=len(tag2idx),
        dropout=args.dropout,
        embedding=torch.tensor(vectors).float()
    )
    model = BiLSTM_ATE(model_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 5. Training Loop
    best_loss = float('inf')
    save_dir = "snapshot/ate_model"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            ids, tags, masks = [x.to(device) for x in batch]

            optimizer.zero_grad()
            emissions = model(ids, masks)
            loss = model.loss(emissions, tags, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        # Validation (using Tests set for now)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                ids, tags, masks = [x.to(device) for x in batch]
                emissions = model(ids, masks)
                loss = model.loss(emissions, tags, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch {epoch}: Train Loss {avg_loss:.4f} | Val Loss {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_ate.pt"))
            print("  >>> Saved Best Model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/absa/train.json')
    parser.add_argument('--test_path', type=str, default='data/absa/test.json')
    parser.add_argument('--embed_file', type=str, default='embedding/cc.uz.300.vec')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train(args)