<!-- toc -->

# Byte-Pair Encoding
Byte-Pair Encoding (BPE) is perhaps the most popular tokenization algorithm right now, used by GPT, OPT, BLOOM, Llama, Falcon, etc. Byte-pair encoding/ digram coding is a _compression algorithm_ that comes from information theory, and was first proposed in 1994 (Web archive). The original BPE algorithm  iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte. (In the sense that a sequence which only contains bytes 00000000, 00000001 and 00000010 might get compressed by using bytes like 00000011). [Sennrich et al.](https://arxiv.org/abs/1508.07909) proposed to use BPE for tokenization, where you apply the algorithm to merge characters/ character sequences. Their work is now considered to be a breakthrough moment for subword tokenization, quoting Mielke _et al_. 

Let's now go over the training and the test time algorithm for BPE. The focus in this chapter will be on _training_ a BPE model. We'll dive deeper into the implementation for merging at test time when we implement a GPT2 tokenizer (almost) from scratch in [chapter-3](/3-hf-tokenizer/). 

## Training


## Why is subword tokenization so popular?


