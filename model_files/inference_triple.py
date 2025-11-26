import torch
import torch.nn.functional as F
import argparse
import os
import sys
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from torchtext.legacy import data
from model_files import mydatasets
from model_files.cnn_gate_aspect_model import CNN_Gate_Aspect_Text
from model_files.cnn_gate_aspect_model_atsa import CNN_Gate_Aspect_Text as CNN_Gate_Aspect_Text_ATSA
from model_files.getsemeval import get_semeval, get_semeval_target

# ==========================================
#  CONFIGURATION: UPDATE THESE PATHS!
# ==========================================
# 1. ACSA (Category) Model Snapshot
ACSA_SNAPSHOT = "snapshot/asca-with-fastText/best_model.pt"

# 2. ATSA (Term) Model Snapshot (TRAIN THIS FIRST!)
ATSA_SNAPSHOT = "snapshot/atsa-with-fastText/best_model.pt"


# ==========================================

class ABSAPredictor:
    def __init__(self, task_type, snapshot_path, args):
        self.task_type = task_type
        self.snapshot_path = snapshot_path
        self.args = args
        self.model = None
        self.fields = {}
        self.loaded = False

        if os.path.exists(self.snapshot_path):
            self._load_system()
            self.loaded = True
        else:
            print(f"[{task_type.upper()}] Skipping: Model not found at {self.snapshot_path}")

    def _load_system(self):
        print(f"[{self.task_type.upper()}] Loading System...")

        text_field = data.Field(lower=True, tokenize='moses')
        as_field = data.Field(sequential=False)
        sm_field = data.Field(sequential=False)

        # Load Data to build vocab
        if self.task_type == 'acsa':
            train_raw, test_raw = get_semeval(None, None)
        else:
            train_raw, test_raw = get_semeval_target(None, None)

        train_ds, test_ds, _ = mydatasets.SemEval.splits_train_test(
            text_field, as_field, sm_field, train_raw, test_raw
        )

        text_field.build_vocab(train_ds, test_ds)
        as_field.build_vocab(train_ds, test_ds)
        sm_field.build_vocab(train_ds, test_ds)

        self.fields = {'text': text_field, 'aspect': as_field, 'sentiment': sm_field}

        # Init Model
        self.args.embed_num = len(text_field.vocab)
        self.args.class_num = len(sm_field.vocab) - 1
        self.args.aspect_num = len(as_field.vocab)
        self.args.aspect_embed_dim = self.args.embed_dim

        # Initialize dummy embeddings to satisfy constructor
        self.args.embedding = torch.randn(self.args.embed_num, self.args.embed_dim)
        self.args.aspect_embedding = torch.randn(self.args.aspect_num, self.args.aspect_embed_dim)

        if self.task_type == 'acsa':
            self.model = CNN_Gate_Aspect_Text(self.args)
        else:
            self.model = CNN_Gate_Aspect_Text_ATSA(self.args)

        self.model.load_state_dict(torch.load(self.snapshot_path, map_location=lambda s, l: s))

        if self.args.cuda:
            self.model = self.model.cuda()
        self.model.eval()

    def predict(self, sentence, aspect_input):
        if not self.loaded:
            return "MODEL_NOT_LOADED", 0.0

        text_field = self.fields['text']
        as_field = self.fields['aspect']
        sm_field = self.fields['sentiment']

        # Preprocess
        tokens = text_field.preprocess(sentence)
        max_kernel = max(self.args.kernel_sizes)
        if len(tokens) < max_kernel:
            tokens.extend([text_field.pad_token] * (max_kernel - len(tokens)))

        if aspect_input not in as_field.vocab.stoi:
            # print(f"[{self.task_type.upper()}] Warning: '{aspect_input}' unknown aspect.")
            pass

        text_tensor = text_field.process([tokens])
        aspect_tensor = as_field.process([aspect_input])

        if self.args.cuda:
            text_tensor = text_tensor.cuda()
            aspect_tensor = aspect_tensor.cuda()

        with torch.no_grad():
            text_tensor.t_()
            if not self.args.aspect_phrase:
                aspect_tensor.unsqueeze_(0)
            aspect_tensor.t_()

            logit, _, _ = self.model(text_tensor, aspect_tensor)
            probs = F.softmax(logit, dim=1)
            pred_idx = torch.max(probs, 1)[1].item()

        # vocab includes <unk> at 0
        return sm_field.vocab.itos[pred_idx + 1], probs[0][pred_idx].item()


def style(p):
    c = "\033[92m" if p == 'positive' else "\033[91m" if p == 'negative' else "\033[93m"
    return f"{c}{p.upper()}\033[0m"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sentence', type=str, required=True)
    parser.add_argument('-category', type=str, required=True, help='Comma separated categories e.g. "food,service"')
    parser.add_argument('-term', type=str, required=True, help='Comma separated terms e.g. "osh,waiter"')

    # Defaults
    parser.add_argument('-acsa_path', type=str, default=ACSA_SNAPSHOT)
    parser.add_argument('-atsa_path', type=str, default=ATSA_SNAPSHOT)
    parser.add_argument('-embed_dim', type=int, default=300)
    parser.add_argument('-kernel_num', type=int, default=100)
    parser.add_argument('-kernel_sizes', type=str, default='3,4,5')
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-aspect_phrase', action='store_true', default=False)
    parser.add_argument('-no-cuda', action='store_true', default=False)

    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

    print("\n" + "=" * 60)
    print(f"TRIPLE SENTIMENT ANALYSIS (MULTI-TARGET)")
    print(f"Input: \"{args.sentence}\"")
    print("=" * 60)

    # Initialize Predictors once
    acsa = ABSAPredictor('acsa', args.acsa_path, args)
    atsa = ABSAPredictor('atsa', args.atsa_path, args)

    print("-" * 60)
    print(f"{'TYPE':<10} | {'TARGET':<20} | {'PREDICTION':<20} | {'CONF'}")
    print("-" * 60)

    # Process Categories
    categories = [c.strip() for c in args.category.split(',')]
    for cat in categories:
        res, conf = acsa.predict(args.sentence, cat)
        print(f"{'Category':<10} | {cat:<20} | {style(res):<29} | {conf * 100:.1f}%")

    print("-" * 60)

    # Process Terms
    terms = [t.strip() for t in args.term.split(',')]
    for term in terms:
        res, conf = atsa.predict(args.sentence, term)
        print(f"{'Term':<10} | {term:<20} | {style(res):<29} | {conf * 100:.1f}%")

    print("=" * 60 + "\n")