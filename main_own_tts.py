import json
from difflib import SequenceMatcher
from process import extract_speech_text_and_timestamps, extract_facial_expressions

import os

BASE_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output/")
USE_BASE_OUTPUT_PATH = True
_GET_OUTPUT_PATH = lambda x: os.path.join(BASE_OUTPUT_PATH, x) if USE_BASE_OUTPUT_PATH else x

EXPRESSIONS_TO_REMOVE = [
    "EYE_BLINK_LEFT", "EYE_BLINK_RIGHT",
    "EYE_SQUINT_LEFT", "EYE_SQUINT_RIGHT",
]

def align_words_lcs(original_words, new_words):
    """
    Aligns words using the longest common subsequence (LCS), similar to Git diff.

    original_words: List of words from original audio transcript.
    new_words: List of words from generated TTS transcript.

    Returns: List of (original index, new index) alignments.
    """
    matcher = SequenceMatcher(None, original_words, new_words)
    mapping = []

    for tag, orig_i1, orig_i2, new_i1, new_i2 in matcher.get_opcodes():
        if tag == 'equal':
            # Words match exactly, 1:1 mapping
            for oi, ni in zip(range(orig_i1, orig_i2), range(new_i1, new_i2)):
                mapping.append((oi, ni))

        elif tag == 'replace' or tag == 'delete':
            # Words in original but not in new → map proportionally
            for oi in range(orig_i1, orig_i2):
                closest_new = new_i1 if new_i1 < len(new_words) else new_i1 - 1
                mapping.append((oi, closest_new))

        elif tag == 'insert':
            # Words in new but not in original → map to previous original word
            for ni in range(new_i1, new_i2):
                closest_orig = orig_i1 - 1 if orig_i1 > 0 else 0
                mapping.append((closest_orig, ni))

    return mapping



