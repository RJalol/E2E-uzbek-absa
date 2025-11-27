import torch
import argparse
import numpy as np
import os
import sys

# Add model_files to path
sys.path.append(os.path.join(os.getcwd(), 'model_files'))

from model_files.model_ate import BiLSTM_ATE

# --- Configuration ---
TAG2IDX = {'<pad>': 0, 'O': 1, 'B-ASP': 2, 'I-ASP': 3}
IDX2TAG = {v: k for k, v in TAG2IDX.items()}


def load_vocab(path):
    """Rebuilds the vocabulary from the embedding file exactly like training."""
    vocab = {'<pad>': 0, '<unk>': 1}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0:
                    w = parts[0]
                    if w not in vocab: vocab[w] = len(vocab)
    except Exception as e:
        print(f"[!] Error loading embeddings: {e}")
        # Fallback for testing without embeddings (if random init was used)
        pass
    return vocab


def load_model(model_path, vocab_size, device):
    # 1. Load the checkpoint first to inspect shapes
    state_dict = torch.load(model_path, map_location=device)

    # 2. Automatically detect correct embedding sizes from the saved weights
    saved_vocab_size = state_dict['embedding.weight'].shape[0]
    print(f"[-] Detected saved vocab size: {saved_vocab_size}")

    # 3. Initialize model with the DETECTED size
    args = argparse.Namespace(
        vocab_size=saved_vocab_size,  # Use exact size from file
        embed_dim=300,
        hidden_dim=256,
        num_tags=len(TAG2IDX),
        dropout=0.0,
        embedding=None
    )

    model = BiLSTM_ATE(args)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(sentence, model, word2idx, device):
    # 1. Tokenize (Simple split to match dataset)
    tokens = sentence.split()  # Use split() to preserve mapping, or specialized tokenizer if used

    # 2. Convert to IDs
    input_ids = [word2idx.get(t.lower(), word2idx['<unk>']) for t in tokens]
    tensor_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 3. Create Mask
    mask = torch.ones_like(tensor_ids, dtype=torch.bool).to(device)

    # 4. Predict
    with torch.no_grad():
        # Returns list of lists of tag indices
        pred_tag_indices = model.predict(tensor_ids, mask)

    pred_tags = [IDX2TAG[i] for i in pred_tag_indices[0]]

    # 5. Extract Aspects
    aspects = []
    current_aspect = []

    for token, tag in zip(tokens, pred_tags):
        if tag == 'B-ASP':
            if current_aspect: aspects.append(" ".join(current_aspect))
            current_aspect = [token]
        elif tag == 'I-ASP':
            current_aspect.append(token)
        else:
            if current_aspect: aspects.append(" ".join(current_aspect))
            current_aspect = []
    if current_aspect: aspects.append(" ".join(current_aspect))

    return aspects, list(zip(tokens, pred_tags))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', type=str, required=True, help="Sentence to analyze")
    parser.add_argument('--checkpoint', type=str, default="snapshot/ate_model/best_ate.pt")
    parser.add_argument('--embed_file', type=str, default="embedding/cc.uz.300.vec")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[-] Loading Vocab...")
    word2idx = load_vocab(args.embed_file)

    print("[-] Loading Model...")
    model = load_model(args.checkpoint, len(word2idx), device)

    print("[-] Running Prediction...")
    aspects, tagged_tokens = predict(args.sentence, model, word2idx, device)

    print("\n" + "=" * 40)
    print(f"INPUT:   {args.sentence}")
    print(f"ASPECTS: {aspects}")
    print("-" * 40)
    print("DETAILS:")
    for t, l in tagged_tokens:
        print(f"  {t:15} : {l}")
    print("=" * 40 + "\n")