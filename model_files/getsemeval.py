import json
import os
from collections import defaultdict

# =========================================================
# FIX: Use absolute paths relative to this script file
# This ensures it works regardless of where you run python from
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets .../E2E-uzbek-absa/model_files
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Gets .../E2E-uzbek-absa

# Define data directories
ACSA_DIR = os.path.join(PROJECT_ROOT, 'data', 'acsa')
ASTA_DIR = os.path.join(PROJECT_ROOT, 'data', 'atsa')


def load_and_group_json(file_path):
    """
    Loads an unrolled JSON file (list of dicts with 'sentence', 'aspect', 'sentiment')
    and groups it by sentence to reconstruct the format run.py expects.
    """
    # Normalize path to fix any mixed slashes (Windows/Linux compatibility)
    file_path = os.path.normpath(file_path)

    if not os.path.exists(file_path):
        print(f"[ERROR] Data file not found at: {file_path}")
        return []

    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Group by sentence
    grouped_data = defaultdict(list)
    for entry in raw_data:
        sentence = entry.get('sentence', '').strip()
        sentiment = entry.get('sentiment')

        # Try to find the aspect term key (handles both ACSA and ATSA formats)
        aspect = entry.get('aspect')
        if aspect is None:
            aspect = entry.get('term', entry.get('target', 'NULL'))

        if sentence:
            grouped_data[sentence].append((aspect, sentiment))

    # Reconstruct dataset list
    dataset = []
    for sentence, asp_sent_list in grouped_data.items():
        dataset.append({
            "sentence": sentence,
            "aspect_sentiment": asp_sent_list
        })

    return dataset


def get_semeval(years, aspects, rest_lap='r', use_attribute=False, dedup=False):
    """
    Loads ACSA (Aspect Category) data from your 'data/acsa' folder.
    """
    train_path = os.path.join(ACSA_DIR, 'acsa_train.json')
    test_path = os.path.join(ACSA_DIR, 'acsa_test.json')

    train_data = load_and_group_json(train_path)
    test_data = load_and_group_json(test_path)

    print(f"# ACSA Train loaded: {len(train_data)} sentences")
    print(f"# ACSA Test loaded: {len(test_data)} sentences")

    return train_data, test_data


def get_semeval_target(years, rest_lap='rest', dedup=False):
    """
    Loads ATSA (Aspect Term) data from your 'data/atsa' folder.
    """
    train_path = os.path.join(ASTA_DIR, 'asta_train.json')
    test_path = os.path.join(ASTA_DIR, 'asta_test.json')

    train_data = load_and_group_json(train_path)
    test_data = load_and_group_json(test_path)

    print(f"# ATSA Train loaded: {len(train_data)} sentences")
    print(f"# ATSA Test loaded: {len(test_data)} sentences")

    return train_data, test_data


def read_yelp(N):
    """Placeholder to satisfy imports in run.py"""
    return [], []


if __name__ == '__main__':
    # Test block to verify paths
    print(f"Base Dir: {BASE_DIR}")
    print(f"Project Root: {PROJECT_ROOT}")
    print("Testing ACSA loader...")
    t, v = get_semeval(None, None)