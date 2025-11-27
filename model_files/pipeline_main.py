import torch
import numpy as np
from multitask_model import BiLSTM_Multitask
from cnn_gate_aspect_model import CNN_Gate_Aspect_Text


# ... load vocabularies ...

def e2e_inference(sentence, extractor_model, sentiment_model, word2idx):
    # 1. Preprocess
    tokens = sentence.lower().split()
    input_ids = [word2idx.get(t, 0) for t in tokens]
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    # 2. Extract Aspects & Categories
    extractor_model.eval()
    with torch.no_grad():
        tags, cats = extractor_model.predict(input_tensor)

    # Parse Tags to Terms
    extracted_terms = []
    current_term = []
    tag_seq = tags[0]  # Batch size 1

    for token, tag_id in zip(tokens, tag_seq):
        # Assuming tag mapping: 0: O, 1: B-ASP, 2: I-ASP
        if tag_id == 1:  # B-ASP
            if current_term: extracted_terms.append(" ".join(current_term))
            current_term = [token]
        elif tag_id == 2:  # I-ASP
            current_term.append(token)
        else:
            if current_term: extracted_terms.append(" ".join(current_term))
            current_term = []
    if current_term: extracted_terms.append(" ".join(current_term))

    # Parse Categories
    extracted_cats = []
    # logic to map True/False back to category names

    # 3. Predict Sentiment for each
    results = []
    sentiment_model.eval()

    for term in extracted_terms:
        # Prepare input for GCAE (Aspect Term Mode)
        # ... logic to run sentiment model ...
        pred = "Positive"  # Placeholder for actual GCAE call
        results.append({'target': term, 'sentiment': pred})

    return results