"""
A simple BPE tokenizer implementation
"""
import json
from typing import Any
import regex as re # regex is cooler than re

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
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []
        self.bpe_ranks = dict()
        # Regex for pre-tokenization - breaking up a piece of text into words, splitting at whitespaces, contractions, etc. Borrowed from GPT-2
        self.pattern_for_splitting = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
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
        
    def __call__(self, text: str):
        # pre-tokenization
        words = self.pattern_for_splitting.findall(text)
        all_tokens = []
        import pdb; pdb.set_trace()
        for word in words:
            word_tokens = self.segment_word(word)
            all_tokens.extend(word_tokens)
        return all_tokens
    
    def segment_word(self, word: str):
        pairs = get_pairs(word) # "obobc" -> set([("o", "b"), ("b", "o"), ("b", "c")])
        while True:
            # get pair with lowest rank and merge
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
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
    