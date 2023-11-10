# Applying what we've learned

## INput Preprocessing
This is different for different tasks: you'd want to add special tokens for intrsuction tuning, not for causal language modelling, where you typically chunk your data. 

## Tokenizer Puzzle 1: The Effect of Vocabulary Size

Inference and Training speed/throughput are often reported in tokens/ sec. However, the number of tokens in a model's vocabulary can differ widely - GPT 2 has a 50k vocab size, Llama 2 has a 32k vocab size, and GPT4 has a whopping 100k vocab size. This means that comparisons based solely on token counts might not make sense. You might ask: How _exactly_ does this affect sequence length: Are there heuristics to predict sequence lengths based on vocab sizes or other data? Well, that's a very hard question. Firstly, a lot of current tokenizers are BPE-based, so let's say we're only looking at BPE tokenizers. Now, the difference in vocabulary size comes from, you guessed it, the training corpus. One training corpus might include a lot of code, and thus the vocabulary would have a bunch of code-specific tokens, while another might include very little, and you might end up with only character-level tokens for code. With such variability, it's not easy to just look at vocabulary size and say that for this data, I will get x times more tokens with LLama 2 vs GPT4. Indeed, you can see the same from Thomas Wolf's tokenizer puzzle:

> Sunday small guessing puzzle
> Let's say I have 3 tokenizers:
> - llama2: 32k vocab
> - falcon: 65k vocab
> - GPT4: 100k vocab  

> I take ~2M random documents from the web  
> (let’s say 10 random parquet files from RefinedWeb from https://huggingface.co/datasets/tiiuae/falcon-refinedweb roughly 1B tokens). I tokenize them with the tokenizers.
> What will be the relative fertilities of these 3 tokenizers? ie. how many more tokens with falcon and llama2 versus gpt4 for instance would you expect. And why does this matter?

A general heuristic is that a bigger vocab will lead to fewer tokens. Having more tokens means that longer character sequences might get represented with the additional tokens present, and you can get a shorter overall sequence length using the additional token ids. With BPE, you can simply say that you're defintely making more merges, and thus overall sequence length reduces.

Here's the answer: 
> Running on 1B tokens from the RefinedWeb dataset. 
> - GPT4 tokenizer (100k vocab) gives you 0.997B tokens 
> - Falcon tokenizer (64k vocab) gives you ~5% more tokens (1.04B)
> - Llama2 tokenizer (32k vocab) gives you ~20% more tokens (1.18B)

These numbers are... definitely non-trivial to see. 
You should do a comparison yourself on colab.

## Tokenizer Puzzle 2: White spaces matter