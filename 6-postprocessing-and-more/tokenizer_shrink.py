"""
Tokenizer shrinking recipe from Stas Bekman and Anthony Moi.
Reference: https://discuss.huggingface.co/t/tokenizer-shrinking-recipes/8564 
For more details, please see: https://github.com/stas00/ml-engineering/  
"""

import json
from transformers import AutoTokenizer
from tokenizers import Tokenizer

vocab_keep_items = 5000
mname = "microsoft/deberta-base"


tokenizer = AutoTokenizer.from_pretrained(mname, use_fast=True)
assert tokenizer.is_fast, "This only works for fast tokenizers."
tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
vocab = tokenizer_json["model"]["vocab"]
# Iterate over the vocabulary for different models and keep only the first `vocab_keep_items` tokens
if tokenizer_json["model"]["type"] == "BPE":
    new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items }
    merges = tokenizer_json["model"]["merges"]
    new_merges = []
    # handle merges: keep only the merge rules for which the (merging) pair of tokens and the merged token are in the new vocab
    for i in range(len(merges)):
        a, b = merges[i].split()
        new_token = "".join((a, b))
        if a in new_vocab and b in new_vocab and new_token in new_vocab:
            new_merges.append(merges[i])
    tokenizer_json["model"]["merges"] = new_merges
elif tokenizer_json["model"]["type"] == "Unigram":
    new_vocab = vocab[:vocab_keep_items]
elif tokenizer_json["model"]["type"] == "WordPiece" or tokenizer_json["model"]["type"] == "WordLevel":
    new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items }
else:
    raise ValueError(f"don't know how to handle {tokenizer_json['model']['type']}")

# a hack for GPT2, since a special token is at the END of the vocab - most tokenizers have it at the beginning
if "gpt2" in mname:
        new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items-1 }
        new_vocab["<|endoftext|>"] = vocab_keep_items-1
else:
    new_vocab = { token: i for token, i in vocab.items() if i < vocab_keep_items }

tokenizer_json["model"]["vocab"] = new_vocab
tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
save_name = mname.split("/")[-1] + "_tiny"
tokenizer.save_pretrained(save_name)