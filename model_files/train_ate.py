import os
import argparse
import time
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import modules
from model_ate import BiLSTM_ATE
from dataset_ate import ATEDataset, collate_fn


def calculate_accuracy(emissions, tags, mask, model):
    """Decodes tags and calculates token-level accuracy ignoring padding."""
    # decode returns list of list of tag indices
    pred_tags = model.crf.decode(emissions, mask)

    correct = 0
    total = 0

    # Flatten everything to compare
    # tags is (Batch, Seq), mask is (Batch, Seq)
    tags = tags.cpu().numpy()
    mask = mask.cpu().numpy()

    for i, seq_preds in enumerate(pred_tags):
        # Get true tags for this sequence, ignoring padding based on mask
        seq_len = int(mask[i].sum())
        true_seq = tags[i][:seq_len]
        pred_seq = seq_preds[:seq_len]

        # Compare
        for p, t in zip(pred_seq, true_seq):
            if p == t:
                correct += 1
            total += 1

    return correct, total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # 1. Setup Directories & Log
    save_dir = "snapshot/ate"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_path = os.path.join(save_dir, 'training_log_with_FastText.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # We use standard headers so visualize.py picks them up as "Single Task"
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'time'])

    # 2. Config & Data
    tag2idx = {'<pad>': 0, 'O': 1, 'B-ASP': 2, 'I-ASP': 3}

    print("[-] Loading Embeddings...")
    word2idx = {'<pad>': 0, '<unk>': 1}
    vectors = [np.zeros(300), np.random.uniform(-0.1, 0.1, 300)]

    try:
        with open(args.embed_file, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    w = parts[0]
                    v = np.array([float(x) for x in parts[1:]])
                    word2idx[w] = len(word2idx)
                    vectors.append(v)
        vectors = np.array(vectors)
    except:
        print("[!] Embeddings not found, using random.")
        vectors = np.random.uniform(-0.1, 0.1, (len(word2idx) + 1000, 300))

    train_dataset = ATEDataset(args.train_path, word2idx, tag2idx)
    test_dataset = ATEDataset(args.test_path, word2idx, tag2idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. Model
    print(f"[-] Initializing Model on {device}...")
    model_args = argparse.Namespace(
        vocab_size=len(word2idx) + 1000,
        embed_dim=300,
        hidden_dim=args.hidden_dim,
        num_tags=len(tag2idx),
        dropout=args.dropout,
        embedding=torch.tensor(vectors).float()
    )
    model = BiLSTM_ATE(model_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')

    # 4. Training Loop
    print(f"[-] Starting Training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        # --- TRAIN ---
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            ids, tags, masks = [x.to(device) for x in batch]

            optimizer.zero_grad()
            emissions = model(ids, masks)
            loss = model.loss(emissions, tags, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate Acc
            c, t = calculate_accuracy(emissions, tags, masks, model)
            train_correct += c
            train_total += t

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total if train_total > 0 else 0

        # --- VAL ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in test_loader:
                ids, tags, masks = [x.to(device) for x in batch]
                emissions = model(ids, masks)
                loss = model.loss(emissions, tags, masks)
                val_loss += loss.item()

                c, t = calculate_accuracy(emissions, tags, masks, model)
                val_correct += c
                val_total += t

        avg_val_loss = val_loss / len(test_loader)
        avg_val_acc = val_correct / val_total if val_total > 0 else 0
        epoch_time = time.time() - start_time

        # Log to Console
        print(f"    Train Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.2%}")
        print(f"    Val   Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.2%}")

        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc, f"{epoch_time:.2f}"])

        # # Save Best
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     save_path = os.path.join(save_dir, "best_ate.pt")
        #     torch.save(model.state_dict(), save_path)
        #     print(f"    >>> ⭐ New Best Model Saved! (Loss: {best_val_loss:.4f})")

        # Initialize these before the loop starts
        best_val_loss = float('inf')
        best_val_acc = 0.0

        # ... inside loop ...

        # Logic: Save if Loss is better OR Accuracy is better
        is_best = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            is_best = True

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            is_best = True

        if is_best:
            save_path = os.path.join(save_dir, "best_ate.pt")
            torch.save(model.state_dict(), save_path)
            print(f"    >>> ⭐ New Best Model Saved! (Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.2%})")

        print("-" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/absa/train.json')
    parser.add_argument('--test_path', type=str, default='data/absa/test.json')
    parser.add_argument('--embed_file', type=str, default='embedding/cc.uz.300.vec')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train(args)