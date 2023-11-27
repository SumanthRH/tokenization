
# Table of Contents
<!-- toc -->

- [Byte-Pair Encoding](#byte-pair-encoding)
  * [Why is subword tokenization so popular?](#why-is-subword-tokenization-so-popular)
  * [Training](#training)
  * [Test time](#test-time)
- [Implementation](#implementation)
- [Step into the walkthrough](#step-into-the-walkthrough)
- [Next Chapter](#next-chapter)

<!-- tocstop -->

# Byte-Pair Encoding
Byte-Pair Encoding (BPE) is perhaps the most popular tokenization algorithm right now, used by GPT, OPT, BLOOM, Llama, Falcon, etc. Byte-pair encoding/ digram coding is a _compression algorithm_ that comes from information theory, and was first proposed in 1994 (Web archive). The original BPE algorithm  iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte. (In the sense that a sequence which only contains bytes 00000000, 00000001 and 00000010 might get compressed by using bytes like 00000011). [Sennrich et al.](https://arxiv.org/abs/1508.07909) proposed to use BPE for tokenization, where you apply the algorithm to merge characters/ character sequences. Their work is now considered to be a breakthrough moment for subword tokenization, quoting Mielke _et al_. 

Let's now go over the training and the test time algorithm for BPE. The focus in this chapter will be on _training_ a BPE model. We'll dive deeper into the implementation for merging at test time when we implement a GPT2 tokenizer (almost) from scratch in [chapter-3](/3-hf-tokenizer/). 


## Why is subword tokenization so popular?
A quick digression. Why is subword tokenization is the dominant one? Let's revisit the downsides of character and word-based tokenization methods. A character-based tokenization algorithm has good generalization capabilities - note that we're referring to the tokenizer - in that the training corpus can be widely different from the test corpus, and you'll rarely have out-of-vocabulary issues (happens only when a new character comes along), and the _fertility_ (the number of tokens a word is split into) is pretty predictable. On the other hand, imagine the input to the neural network - these are the embeddings corresponding to each character. The network has to learn the meanings for different words based on the sequence of character embeddings. Each embedding will likely contain very little word-specific information, because, well, the same character appears in a lot of words. Thus, a shortcoming is that this is not an informative _input representation_. On the other hand, you have word-based tokenization. The upside is that the neural network can learn a good embedding for each word that can summarize the meaning and the context in which the word appears accurately. However, you're stuck with dealing with a large vocabulary, because of all the different variations for each word. Some words are also naturally split into multiple meaningful sub-words in many languages. An example from Sennrich _et al._ is the German word "Abwasser|behandlungs|anlange", which means ("a sewage water treatment plant"). In such cases, it makes more sense to have a sequence of embeddings of the subwords to represent the word, instead of one vector. The fancy term for languages which extensively use such compounds is [agglutinative](https://en.wikipedia.org/wiki/Agglutinative_language) (Ex: Japanese, Turkish, etc). Thus, with this motivation, we want a subword tokenization algorithm because they can (1) represent unseen data atleast with a character-level tokenization (2) learn meaningful subwords that can be useful for the neural network you train.

## Training
The steps for training a BPE model are as follows:
1. Extract a list of words from the training corpus. In Sennrich et al, this pre-tokenization step simply removed whitespace information. They also added an end-of-word token (like `</w>`) for each word to get word boundaries. (A simple extension used in implementations like in Sentencepiece makes sure to preserve whitespace, etc)
2. Make a word counter dictionary with keys being words and values being frequencies in the training corpus.
3. Keep a vocabulary of symbols (variable length strings), initialized with unique characters present in the training corpus.
4. Initialize a list of _merge rules_ to be an empty. Each merge rule is a tuple of symbol/token to merge at test time.
4. Iteratively do the following:
    - Get the most frequent pair of symbols in the current vocabulary, by going over the word counter. (Ex: `("l", "e")`)
    - Merge the two symbols into a new symbol, and _add_ this to the vocabulary. This is like a new byte in the original BPE except we have variable length strings (_character n-gram_). 
    - Add our tuple of symbols to the list of merge rules.
    - Replace all occurences of the pair of symbols with the new symbol. For example, a word, segmented as `("l", "e", "a", "r", "n")` becomes `("le", "a", "r", "n")`. 
    - Repeat until you reach the target vocabulary size. (a hyperparameter)

This is it! The full algorithm is pretty simple.

## Test time
At test time, the algorithm is very similar to training, except you're doing lookups in the set of merge rules:
1. Perform character-level tokenization for input text.
2. Find all pairs of symbols/tokens in the current words.
3. Start merging pairs by going in order of merge rules: merges learnt earlier in the training process have higher priority, and are performed earlier.
4. Repeat until you can't merge anymore.


# Implementation
A minimal implementation for training a BPE tokenizer is in `orig_bpe.py`. The code is almost the same as the one in the original paper, with minor edits for clarity and for loading our own text corpus. To run the training on your machine, `cd` into the current directory and run
```
python orig_bpe.py
```
Now, as mentioned, we'd ideally like to keep whitespace information, but that is a detail that can be distracting while doing a minimal implementation. The BPE tokenizer implementated in [chapter-3](/3-hf-tokenizer/) will work with all special characters, so we'll ignore this detail for now.

# Step into the walkthrough
Head over to [walkthrough.ipynb](walkthrough.ipynb) for a simple guide to training a BPE model. This is a notebook version of the code in `orig_bpe.py`, and should be easier to digest.

# Next Chapter
We'll take a close look at the Python implementation for a 🤗 tokenizer and implement a minimal version of GPT2's tokenizer ourselves!

**References** 
- Neural Machine Translation of Rare Words with Subword Units (BPE): https://arxiv.org/abs/1508.07909 
- Lei Mao's BPE guide: https://leimao.github.io/blog/Byte-Pair-Encoding/ . The code here is also from the original paper. 
