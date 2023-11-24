"""
A simple BPE tokenizer implementation
"""
import json
from typing import Any
import regex as re # regex is cooler than re
import warnings
import os
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

# a hacky custom warning formatter to avoid full path being shown
def custom_formatwarning(msg, category, filename, lineno, line=None):
    filename = os.path.basename(filename)
    # Format the warning message
    return f'{filename}:{lineno}: {category.__name__}: {msg}\n'

warnings.formatwarning = custom_formatwarning

def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    Reference: HF's GPT-2 tokenizer
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class BPE:
    def __init__(self, vocab_file: str):
        self.token_to_id = {} # vocab. Called `encoder` in `GPT2Tokenizer`
        self.id_to_token = {} # called `decoder` in `GPT2Tokenizer`
        self.merges = []
        self.bpe_ranks = dict()
        self.byte_encoder = bytes_to_unicode() # maps bytes to unicode strings
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.load_vocab(vocab_file)
    
    def load_vocab(self, vocab_file: str):
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
        self.token_to_id = vocab_data["vocab"]
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.merges = vocab_data["merges"]
        for i, merge in enumerate(self.merges):
            pair = tuple(merge.split()) # merge is repr as "a b". Works because we split on whitespace in pre-tok step
            # i is the index of the merge in the merges list, also the rank. lower rank means merge happens earlier
            self.bpe_ranks[pair] = i 
        
    def __call__(self, word: str, dont_byte_encode: bool = False) -> Any:
        if " " in word and not dont_byte_encode:
            warnings.warn("Word contains whitespaces. Encoding to unicode strings...")
            word = "".join([self.byte_encoder[b] for b in word.encode("utf-8")])
        pairs = get_pairs(word) # "obobc" -> set([("o", "b"), ("b", "o"), ("b", "c")])
        while True:
            # get pair of chars/tokens with lowest rank and merge
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i) # find index of occurence of `first` in word[i:]
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        return word

    def __repr__(self) -> str:
        return f"BPE(vocab_size={len(self.token_to_id)})"
    
    def add_token(self, token: str):
        if token in self.token_to_id:
            raise ValueError(f"Token {token} already in vocabulary.")
        self.token_to_id[token] = len(self.token_to_id)
        self.id_to_token[len(self.id_to_token)] = token
    

if __name__ == "__main__":

    bpe = BPE("vocab.json")
    from transformers import AutoTokenizer
    gpt2 = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    gpt2.encode(" worda")
    import pdb; pdb.set_trace()