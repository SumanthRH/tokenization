

# Tokenization

Tokenization is an oft-neglected part of natural language processing. With the recent blow-up of interest in language models, it might be good to step back and really get into the guts of what tokenization is. This repo is meant to serve as a deep dive into different aspects of tokenization. It's been organized as bite-size chapters for easy navigation, with some code samples and (poorly designed) walkthrough notebooks. This is NOT meant to be a complete reference in itself, and is meant accompany other excellent resources like [HuggingFace's NLP course](https://huggingface.co/learn/nlp-course/chapter6/1). The following topics are covered: 

1. [Intro](/1-intro/): A quick introduction on tokens and the different tokenization algorithms out there. 
2. [BPE](/2-bpe/): A closer look at the Byte-Pair Encoding tokenization algorithm. We'll also go over a minimal implementation for training a BPE model.
3. [🤗 Tokenizer](/3-hf-tokenizer/): The internals of HuggingFace tokenizers! We look at state (what's saved by a tokenizer), data structures (how does it store what it saves), and methods (what functionality do you get). We also implement a minimal 🤗 Tokenizer in Python for GPT2.
4. [Challenges with Tokenization](/4-tokenization-is-hard/): Challenges with integer tokenization, tokenization for non-English languages and going multilingual, with a focus on the recent No Language Left Behind (NLLB) effort from Meta.
5. [Puzzles](/5-puzzles/): Some simple puzzles to get you thinking about pre-tokenization, vocabulary size, etc.
6. [PostProcessing and more](/6-postprocessing-and-more/): A look at special tokens and postprocessing, glitch tokens and why you might want to shrink your tokenizer.
7. [Galactica](/7-galactica/): Thinking about tokenizer design by diving into the Galactica paper.

## Requirements
To run the notebooks in the repo, you only need two libraries: `transformers` and `tiktoken`:

```
pip install transformers tiktoken
```

Code has been tested with `transformers==4.35.0` and `tiktoken==0.5.1`.

## Recommended Prerequisites
A basic understanding of language models and tokenization is a must: 
- [A Hackers' Guide to Language Models](https://youtu.be/jkrNMKz9pWU?si=y06_GUgoaG8_ASyd) by Prof. Jeremy Howard.
- [What makes LLM tokenizers different from each other?](https://youtu.be/rT6wVLEDC_w?si=v58zCYEIf0pheaEo) by Jay Alammar.
- [ChatGPT has Never Seen a SINGLE Word (Despite Reading Most of The Internet). Meet LLM Tokenizers.](https://youtu.be/uSinkCeUg9U?si=P25RHVkMKlm-Qtd6) by Jay Alammar
- [Optional] [Chapter on tokenizers from The 🤗 NLP Course](https://huggingface.co/learn/nlp-course/chapter6/1)

## Contributing
If you notice any mistake/bug, or feel you could make an improvement to any section of the repo, please open an issue or make a PR 🙏
