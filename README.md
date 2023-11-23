

# Tokenization

Tokenization is an oft-neglected part of natural language processing. With the recent blow-up of interest in language models, it might be good to step back and really get into the guts of what tokenization is. This repo is meant to serve as a deep dive into different aspects of tokenization. It's been organized as bite-size chapters for easy navigation. This is NOT meant to be a complete reference in itself, and is meant accompany other excellent resources like [HuggingFace's NLP course](https://huggingface.co/learn/nlp-course/chapter6/1). The following topics are covered: 

1. [Intro](/1-intro/): A quick introduction on tokens and the different tokenization algorithms out there. 
2. [BPE](/2-bpe/): A closer look at the Byte-Pair Encoding tokenization algorithm and some variants. 
3. [🤗 Tokenizer](/3-hf-tokenizer/): The internals of HuggingFace tokenizers
4. [Tokenization Challenges](/4-tokenization-is-hard/): Challenges with integer tokenization, tokenization for non-English languages and going multilingual.
5. [Puzzles](/5-puzzles/): Some simple puzzles to get you thinking about pre-tokenization, vocabulary size, etc
6. [Galactica](/6-galactica/): Thinking about tokenizer design by diving into the Galactica paper.
7. [Other Topics](/7-misc/): A look at special tokens and postprocessing, recent alternatives to subword tokenization, and going tiny.

## Recommended Prerequisites
A basic understanding of language models and tokenization is a must: 
- [A Hackers' Guide to Language Models](https://youtu.be/jkrNMKz9pWU?si=y06_GUgoaG8_ASyd) by Hacker-in-Chief, Prof. Jeremy Howard
- [What makes LLM tokenizers different from each other?](https://youtu.be/rT6wVLEDC_w?si=v58zCYEIf0pheaEo) by Jay Alammar
- [ChatGPT has Never Seen a SINGLE Word (Despite Reading Most of The Internet). Meet LLM Tokenizers.](https://youtu.be/uSinkCeUg9U?si=P25RHVkMKlm-Qtd6) by Jay Alammar

- [Optional] [Chapter on tokenizers from The 🤗 NLP Course](https://huggingface.co/learn/nlp-course/chapter6/1)


