import re
from collections import defaultdict

EOW_TOKEN = '</w>'
def get_segmented_text(filename):
    """Extracts unique words and performs simple character-level tokenization from a given file."""
    segmented_word_to_freq = defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                # Separate each character by space and add end of word token
                segmented_word = ' '.join(list(word)) + " " + EOW_TOKEN
                segmented_word_to_freq[segmented_word] += 1
    return segmented_word_to_freq

def merge_pair_and_update_vocab(pair, word_to_freq: dict, token_vocab: dict):
    """Merges the most frequent pair in vocabulary. Updates token_vocab in place"""
    out_word_to_freq = {}
    bigram = re.escape(' '.join(pair))
    # pattern object: negative lookbehind i.e bigram is preceeeded
    # only by a whitespace and negative lookahead.
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in word_to_freq:
        # Merge the pair if it's in the word
        new_word = pattern.sub(''.join(pair), word)
        out_word_to_freq[new_word] = word_to_freq[word]
        if new_word != word:
            token_vocab[''.join(pair)] += 1
    return out_word_to_freq

def get_token_pairs(vocab: dict):
    """Get the counts of pairs of symbols in the vocabulary."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            if EOW_TOKEN in [symbols[i], symbols[i+1]]:
                continue
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def get_token_vocabulary(word_to_freq):
    """Returns the token vocabulary based on the current state of the vocabulary."""
    tokens = defaultdict(int)
    for word, freq in word_to_freq.items():
        for token in word.split():
            tokens[token] += freq
    return tokens

# Get the vocabulary from a file
word_to_freq = get_segmented_text('ex_corpus.txt')

# Number of merges to perform
num_merges = 10

for i in range(num_merges):
    # Count frequency of pairs of tokens
    pairs = get_token_pairs(word_to_freq)
    tokens = get_token_vocabulary(word_to_freq)
    if not pairs:
        break

    # Get the most frequent pair
    most_frequent_pair = max(pairs, key=pairs.get)
    # Merge the most frequent pair in the vocabulary
    word_to_freq = merge_pair_and_update_vocab(most_frequent_pair, word_to_freq=word_to_freq, token_vocab=tokens)

    # Optional: Print the iteration, the most frequent pair, and the number of unique tokens
    print(f'Iter: {i}')
    print("Segmented words: ", list(word_to_freq.keys()))
    print("Tokens: ", tokens)
    print(f'Best pair: {most_frequent_pair}')
    print('Number of tokens: {}'.format(len(tokens)))
    print('==========')

# Final tokens and freqs
sorted_tokens = sorted(tokens.items(), key=lambda item: item[1], reverse=True)
# Calculate the maximum length of the token strings for alignment
max_token_length = max(len(token) for token, _ in sorted_tokens)

print("Final Token Vocabulary:")
print(f"{'Token':<{max_token_length}} Frequency")
for token, freq in sorted_tokens:
    if token == EOW_TOKEN:
        continue # skip eow token
    print(f"{token:<{max_token_length}} {freq}")





