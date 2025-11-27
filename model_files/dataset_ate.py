import json
import torch
from torch.utils.data import Dataset
import numpy as np


class ATEDataset(Dataset):
    def __init__(self, file_path, word2idx, tag2idx):
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        for item in raw_data:
            tokens = item['tokens']
            labels = item['labels']

            # Convert words to IDs
            # Lowercase for consistency
            token_ids = [self.word2idx.get(t.lower(), self.word2idx['<unk>']) for t in tokens]

            # Convert labels to IDs
            # Ensure label exists, else default to 'O'
            label_ids = [self.tag2idx.get(l, self.tag2idx['O']) for l in labels]

            self.data.append({
                'ids': token_ids,
                'tags': label_ids
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    # Dynamic padding to max length in this batch
    max_len = max(len(x['ids']) for x in batch)

    batch_ids = []
    batch_tags = []
    masks = []

    for x in batch:
        length = len(x['ids'])
        # Pad IDs with 0 (<pad>)
        ids = x['ids'] + [0] * (max_len - length)
        # Pad Tags with 0 (<pad>)
        tags = x['tags'] + [0] * (max_len - length)
        # Mask: 1 for real tokens, 0 for pads
        mask = [1] * length + [0] * (max_len - length)

        batch_ids.append(ids)
        batch_tags.append(tags)
        masks.append(mask)

    return (torch.tensor(batch_ids, dtype=torch.long),
            torch.tensor(batch_tags, dtype=torch.long),
            torch.tensor(masks, dtype=torch.bool))