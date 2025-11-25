import json
import os
from collections import defaultdict

# Define paths relative to this script (assuming script is in model_files/ and data is in ../data/)
# We use the folder names you provided: 'asca' and 'asta'
ACSA_DIR = '../data/asca'
ASTA_DIR = '../data/asta'  # User folder name 'asta' (likely meant 'atsa')


def load_and_group_json(file_path):
    """
    Loads an unrolled JSON file (list of dicts with 'sentence', 'aspect', 'sentiment')
    and groups it by sentence to reconstruct the format run.py expects.

    Returns:
        List of dicts: [{'sentence': text, 'aspect_sentiment': [(aspect, sentiment), ...]}, ...]
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Group by sentence
    grouped_data = defaultdict(list)
    for entry in raw_data:
        # Support both 'aspect' (ACSA) and 'term'/'target' (ATSA) keys if they vary
        # Assuming your json keys are 'sentence', 'aspect', 'sentiment' based on standard output
        sentence = entry.get('sentence', '').strip()
        sentiment = entry.get('sentiment')

        # Try to find the aspect term key
        aspect = entry.get('aspect')
        if aspect is None:
            aspect = entry.get('term', entry.get('target', 'NULL'))

        if sentence:
            grouped_data[sentence].append((aspect, sentiment))

    # reconstruct dataset list
    dataset = []
    for sentence, asp_sent_list in grouped_data.items():
        dataset.append({
            "sentence": sentence,
            "aspect_sentiment": asp_sent_list
        })

    return dataset


def get_semeval(years, aspects, rest_lap='r', use_attribute=False, dedup=False):
    """
    Loads ACSA (Aspect Category) data from your 'asca' folder.
    Arguments (years, etc.) are ignored to use your static files.
    """
    train_path = os.path.join(ACSA_DIR, 'acsa_train.json')
    test_path = os.path.join(ACSA_DIR, 'acsa_test.json')

    # We load the full train/test files.
    # run.py will generate the 'hard' (mixed) subsets automatically from these.
    train_data = load_and_group_json(train_path)
    test_data = load_and_group_json(test_path)

    print(f"# ACSA Train loaded: {len(train_data)} sentences")
    print(f"# ACSA Test loaded: {len(test_data)} sentences")

    return train_data, test_data


def get_semeval_target(years, rest_lap='rest', dedup=False):
    """
    Loads ATSA (Aspect Term) data from your 'asta' folder.
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
    # Simple test block
    print("Testing ACSA loader...")
    try:
        t, v = get_semeval(None, None)
        if t:
            print("First ACSA example:", t[0])
    except Exception as e:
        print(e)

    print("\nTesting ATSA loader...")
    try:
        t, v = get_semeval_target(None, None)
        if t:
            print("First ATSA example:", t[0])
    except Exception as e:
        print(e)