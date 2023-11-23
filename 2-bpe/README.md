# Byte-Pair Encoding
Byte-Pair Encoding (BPE) is perhaps the most popular tokenization algorithm right now, used by GPT, OPT, BLOOM, Llama, Falcon, etc. Byte-pair encoding/ digram coding is a _compression algorithm_ comes from information theory, and was first proposed in 1994 (Web archive). The original BPE algorithm  iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte. (In the sense that a sequence which only contains bytes 00000000, 00000001 and 00000010 might get compressed by using bytes like 00000011)

[Sennrich et al.](https://arxiv.org/abs/1508.07909) proposed to use BPE for tokenization, where you apply the algorithm to merge characters/ character sequences. 

https://simonwillison.net/2023/Jun/8/gpt-tokenizers/