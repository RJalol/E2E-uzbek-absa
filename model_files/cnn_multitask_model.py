import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Gate_MultiTask(nn.Module):
    def __init__(self, args):
        super(CNN_Gate_MultiTask, self).__init__()
        self.args = args

        # 1. Embeddings
        V = args.embed_num
        D = args.embed_dim
        self.embed = nn.Embedding(V, D)
        self.embed.weight = nn.Parameter(args.embedding, requires_grad=True)

        # Aspect Embedding (Used for Sentiment Classification Head)
        A = args.aspect_num
        self.aspect_embed = nn.Embedding(A, args.aspect_embed_dim)
        self.aspect_embed.weight = nn.Parameter(args.aspect_embedding, requires_grad=True)

        # 2. Convolutional Layers (Shared Feature Extractor)
        Co = args.kernel_num
        Ks = args.kernel_sizes

        # Tanh-Gated CNN Units
        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])

        # Aspect-Gating Convolution (Specific to GCAE Sentiment logic)
        self.convs3 = nn.ModuleList(
            [nn.Conv1d(D, Co, K, padding=K - 2) for K in Ks])  # Padding to keep length same for gating

        # 3. Prediction Heads

        # Head A: Sentiment Classification (The original GCAE part)
        self.fc1 = nn.Linear(len(Ks) * Co, Co)
        self.fc_aspect = nn.Linear(args.aspect_embed_dim, Co)
        self.fc_sentiment = nn.Linear(Co, args.class_num)  # Sentiment classes

        # Head B: Aspect Category Detection (Multi-label or Single-label)
        # Predicts which category (food, service, etc.) is present in the sentence
        self.fc_category = nn.Linear(len(Ks) * Co, A)

        # Head C: Aspect Term Extraction (Sequence Labeling)
        # Predicts B-Asp, I-Asp, O for each word.
        # We need an output of size (Batch, Seq_Len, Num_Tags)
        # Assuming 3 tags: B, I, O. You might need to define this in args.
        self.num_tags = 3
        self.fc_term = nn.Linear(len(Ks) * Co, self.num_tags)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, feature, aspect):
        # feature: (Batch, Seq_Len)
        # aspect: (Batch,) or (Batch, 1)

        x = self.embed(feature)  # (N, L, D)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (N, D, L)

        aspect_v = self.aspect_embed(aspect)  # (N, Aspect_Dim)
        if aspect_v.dim() == 3:  # Handle phrase case
            aspect_v = F.avg_pool2d(aspect_v, (aspect_v.shape[1], 1)).squeeze(1)

        # --- Shared Convolutional Encoder ---
        # A: Tanh Filter
        x_tanh = [F.tanh(conv(x)) for conv in self.convs1]  # List of (N, Co, L_out)

        # B: Gating Filter (Sigmoid) - Original GCAE uses aspect here
        # To allow multi-tasking, we can use a generic gating or the aspect-aware gating.
        # For the Sentiment Head, we definitely need aspect.
        # For Term/Category heads, we shouldn't rely on knowing the aspect beforehand (that's cheating for extraction).

        # Let's compute a "General Sentence Representation" first using Max Pooling
        pooled_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_tanh]
        sentence_rep = torch.cat(pooled_x, 1)  # (N, len(Ks)*Co)

        # --- HEAD 1: Sentiment Classification (GCAE Logic) ---
        # This head uses the specific aspect embedding to gate the features
        aa = self.fc_aspect(aspect_v)  # (N, Co)

        # We need to apply this 'aa' gate to the convolutional outputs
        # This is complex to combine with standard pooling.
        # Simplified GCAE: Gate the pooled vector? Or Gate the conv maps?
        # Standard GCAE gates the conv maps.

        x_gate = [F.relu(conv(x) + aa.unsqueeze(2)) for conv in
                  self.convs2]  # Slicing/Padding might be needed for shapes to match
        # Note: In original GCAE code, they do: (conv1(x) * sigmoid(conv2(x) + aspect))
        # Let's stick to the implementation logic from your cnn_gate_aspect_model.py roughly

        # Recalculating GCAE-style features for Sentiment only
        x_logit = [F.relu(conv(x) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x_logit_pooled = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_logit]
        x_logit_cat = torch.cat(x_logit_pooled, 1)

        sentiment_logit = self.fc_sentiment(x_logit_cat)

        # --- HEAD 2: Aspect Category Classification ---
        # Uses the generic sentence representation (ungated by specific aspect)
        category_logit = self.fc_category(sentence_rep)

        # --- HEAD 3: Aspect Term Extraction (Token-level) ---
        # This requires preserving sequence length.
        # We can concatenate the conv outputs (upsampled to original length) or use the raw embedding?
        # For simplicity in this CNN architecture, we can try to map the conv features back.
        # Ideally, for ATE, an LSTM is better. But sticking to CNN:
        # We will project the generic sentence_rep back to sequence length? No, that loses local info.
        # We will use the output of the conv layers before pooling.

        # Concatenate feature maps along channel dimension?
        # Shapes are (N, Co, L_out). L_out varies by kernel size.
        # We need to pad/trim to match original Seq_Len L.
        # This is tricky with basic CNNs.
        # ALTERNATIVE: Use the embeddings + a small LSTM for the Term Head.

        # Let's simplify: Return Sentiment and Category logits only for this step.
        # ATE with CNN without same-padding is hard.

        return sentiment_logit, category_logit, None  # Term logit placeholder