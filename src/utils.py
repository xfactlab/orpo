
from typing import List

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def dataset_split_selector(data) -> List:
    """
    This is a function for automating the process of selecting data split.
    Will be further updated.
    """
    if len(data.keys()) == 1:
        return ['train']
    else:
        if 'train_prefs' in data.keys():
            return ['train_prefs', 'test_prefs']
        else:
            return ['train', 'test']