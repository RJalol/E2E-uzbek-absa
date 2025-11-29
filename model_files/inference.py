import torch
import torch.nn.functional as F
import argparse
import os
import sys
import numpy as np

# Add the project root to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from torchtext.legacy import data
from model_files import mydatasets, w2v
from model_files.cnn_gate_aspect_model import CNN_Gate_Aspect_Text
from model_files.getsemeval import get_semeval

# ==========================================
#  CONFIGURATION
# ==========================================
DEFAULT_SNAPSHOT = "snapshot/acsa/best_model.pt"


# ==========================================

def load_vocab_and_model(args):
    print(f"--- Loading System ---")

    print("1. Rebuilding Vocabulary...")
    text_field = data.Field(lower=True, tokenize='moses')
    as_field = data.Field(sequential=False)
    sm_field = data.Field(sequential=False)

    train_data_raw, test_data_raw = get_semeval(None, None)

    if not train_data_raw:
        print("CRITICAL ERROR: No training data found.")
        sys.exit(1)

    train_ds, test_ds, _ = mydatasets.SemEval.splits_train_test(
        text_field, as_field, sm_field,
        train_data_raw, test_data_raw
    )

    text_field.build_vocab(train_ds, test_ds)
    as_field.build_vocab(train_ds, test_ds)
    sm_field.build_vocab(train_ds, test_ds)

    print("2. Initializing Model Architecture...")
    args.embed_num = len(text_field.vocab)
    args.class_num = len(sm_field.vocab) - 1
    args.aspect_num = len(as_field.vocab)
    args.aspect_embed_dim = args.embed_dim
    args.model = "CNN_Gate_Aspect"

    args.embedding = torch.randn(args.embed_num, args.embed_dim)
    args.aspect_embedding = torch.randn(args.aspect_num, args.embed_dim)

    model = CNN_Gate_Aspect_Text(args)

    snapshot_path = args.snapshot if args.snapshot else DEFAULT_SNAPSHOT

    if not os.path.exists(snapshot_path):
        print(f"\n[ERROR] Model file not found: {snapshot_path}")
        sys.exit(1)

    print(f"3. Loading Weights from: {snapshot_path}")
    model.load_state_dict(torch.load(snapshot_path, map_location=lambda storage, loc: storage))

    if args.cuda:
        model = model.cuda()

    model.eval()
    return model, text_field, as_field, sm_field


def predict_sentence(model, sentence, aspect, text_field, as_field, sm_field, args):
    # 1. Preprocess
    tokens = text_field.preprocess(sentence)

    # --- FIX: AUTO-PADDING FOR SHORT SENTENCES ---
    # The model crashes if sentence length < largest kernel size (5)
    max_kernel = max(args.kernel_sizes)
    if len(tokens) < max_kernel:
        pad_len = max_kernel - len(tokens)
        # Use the pad token defined in the field (usually '<pad>')
        tokens.extend([text_field.pad_token] * pad_len)
    # ---------------------------------------------

    # 2. Check Aspect
    if aspect not in as_field.vocab.stoi:
        print(f"[WARNING] Aspect '{aspect}' is unknown (not in training data).")
        print(f"Known aspects: {list(as_field.vocab.stoi.keys())}")

    # 3. Convert to Tensor
    text_tensor = text_field.process([tokens])
    aspect_tensor = as_field.process([aspect])

    if args.cuda:
        text_tensor = text_tensor.cuda()
        aspect_tensor = aspect_tensor.cuda()

    # 4. Predict
    with torch.no_grad():
        text_tensor.t_()
        if not args.aspect_phrase:
            aspect_tensor.unsqueeze_(0)
        aspect_tensor.t_()

        logit, _, _ = model(text_tensor, aspect_tensor)
        probs = F.softmax(logit, dim=1)
        pred_idx = torch.max(probs, 1)[1].item()

    # Offset by 1 because 0 is usually <unk> or padding in SM vocab
    sentiment_label = sm_field.vocab.itos[pred_idx + 1]
    confidence = probs[0][pred_idx].item()

    return sentiment_label, confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-snapshot', type=str, default=None)
    parser.add_argument('-sentence', type=str, required=True)
    parser.add_argument('-aspect', type=str, required=True)
    parser.add_argument('-embed_dim', type=int, default=300)
    parser.add_argument('-kernel_num', type=int, default=100)
    parser.add_argument('-kernel_sizes', type=str, default='3,4,5')
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-aspect_phrase', action='store_true', default=False)
    parser.add_argument('-no-cuda', action='store_true', default=False)

    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

    model, text_field, as_field, sm_field = load_vocab_and_model(args)

    print("\n" + "=" * 40)
    print(f"INPUT:  {args.sentence}")
    print(f"ASPECT: {args.aspect}")
    print("-" * 40)

    try:
        label, conf = predict_sentence(model, args.sentence, args.aspect, text_field, as_field, sm_field, args)
        color = "\033[92m" if label == 'positive' else "\033[91m" if label == 'negative' else "\033[93m"
        reset = "\033[0m"
        print(f"RESULT: {color}{label.upper()}{reset} ({conf * 100:.2f}%)")
    except Exception as e:
        print(f"Error: {e}")
    print("=" * 40 + "\n")