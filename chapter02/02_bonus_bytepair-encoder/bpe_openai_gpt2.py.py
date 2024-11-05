import os
import json
import regex as re
import requests
from tqdm import tqdm
from functools import lru_cache


@lru_cache()
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("i"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encode:
    def __init__(self, encode, bpe_merge, errors='replace'):



def get_encode(model_name, models_dir):
    with open(os.path.join(models_dir, model_name,
            'encoder.join'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(models_dir, model_name,
        'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str
                  in bpe_data.split('\n')[1:-1]]
    return Encode(encoder=encoder, bpe_merges=bpe_merges)

