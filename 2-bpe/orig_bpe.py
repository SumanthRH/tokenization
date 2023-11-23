"""
Minimal implementation of BPE (Byte Pair Encoding)
Simple extension of the original code in "Neural Machine Translation of Rare Words with Subword Units"
"""
import re
from collections import defaultdict

EOW_TOKEN = '</w>'
def get_initial_words(filename):
    segmented_word_to_freq = defaultdict(int) # {"w o r d </w>": 1, ...}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split() # words are split on whitespace; this is our pre-tokenization step
            for word in words:
                # Separate each character with spaces, to show tokens for clarity. "word" -> "w o r d </w>"
                segmented_word = ' '.join(list(word)) + " " +  EOW_TOKEN
                segmented_word_to_freq[segmented_word] += 1
    return segmented_word_to_freq

def get_tokens(word_to_freq):
    tokens = defaultdict(int)
    for word in word_to_freq.keys():
        for token in word.split():
            if token not in tokens:
                tokens[token] = len(tokens)
    return tokens

def get_stats(word_to_freq: dict):
    pairs = defaultdict(int)
    for word, freq in word_to_freq.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq 
    return pairs

def merge_word_splits(pair, v_in: dict):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

word_to_freq = get_initial_words('ex_corpus.txt')
vocab = get_tokens(word_to_freq)
print("Initial vocab: ", vocab)
print("##################")
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(word_to_freq)
    best_pair = max(pairs, key=pairs.get)
    word_to_freq = merge_word_splits(best_pair, word_to_freq)
    new_token = ''.join(best_pair)
    vocab[new_token] = len(vocab)
    print(f"Iteration {i+1}")
    print("Best pair: ", best_pair)
    print("New token: ", new_token)
    print("All words: ", list(word_to_freq.keys()))
    print("##################")

print("Final vocab: ", vocab)