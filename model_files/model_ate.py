import torch
import torch.nn as nn
from crf import CRF


class BiLSTM_ATE(nn.Module):
    def __init__(self, args):
        super(BiLSTM_ATE, self).__init__()

        self.embedding = nn.Embedding(args.vocab_size, args.embed_dim)
        if args.embedding is not None:
            self.embedding.weight = nn.Parameter(args.embedding, requires_grad=True)

        self.lstm = nn.LSTM(args.embed_dim, args.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(args.dropout)

        # Map LSTM output to Tag Space
        self.hidden2tag = nn.Linear(args.hidden_dim, args.num_tags)

        # CRF Layer
        self.crf = CRF(args.num_tags, batch_first=True)

    def forward(self, input_ids, mask=None):
        embeds = self.embedding(input_ids)
        embeds = self.dropout(embeds)

        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)

        return emissions

    def loss(self, emissions, tags, mask):
        # CRF forward returns log-likelihood. We want -log_likelihood for loss.
        log_likelihood = self.crf(emissions, tags, mask=mask, reduction='mean')
        return -log_likelihood

    def predict(self, input_ids, mask=None):
        emissions = self.forward(input_ids, mask)
        return self.crf.decode(emissions, mask)