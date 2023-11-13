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

A general heuristic is that a bigger vocab will lead to fewer tokens. Having more tokens means that longer character sequences might get represented with the additional tokens present, and you can get a shorter overall sequence length using the additional token ids. With BPE, you can simply say that you're defintely making more merges, and thus overall sequence length reduces (Of course, this is some more nuance here - you need to have more tokens dedicated for the given domain/ corpus you're dealing with).

Here's the [answer](https://x.com/Thom_Wolf/status/1701206627859206450?s=20): 
> Running on 1B tokens from the RefinedWeb dataset. 
> - GPT4 tokenizer (100k vocab) gives you 0.997B tokens 
> - Falcon tokenizer (64k vocab) gives you ~5% more tokens (1.04B)
> - Llama2 tokenizer (32k vocab) gives you ~20% more tokens (1.18B)

These numbers are... definitely non-trivial to see. Notice that the absurdly large differences in vocab size do not get reflected as much in the number of tokens. Of course, one would have to go through GPT4's vocab to see what the representation for different data domains (code? other languages?) are like. For example, if the difference between Falcon and Llama2 tokenizers are that the extra tokens in Falcon's vocab were all for code, then you shouldn't expect to see a big difference in tokenized sequence length when you use a corpus of English text. To test this out, let's try this: We'll use a small corpus of plain English text - what better source than Paul Graham's essays - and see the differences in number of tokens. 

The script `paul_graham_essay_scraper.py` will scrape all the text from Paul Graham's essays. I'm not adding the processed, combined plain text file with all essays since that file is large, but this is a small peak at what it looks like:

```
February 2007A few days ago I finally figured out something I've wondered about
for 25 years: the relationship between wisdom and intelligence.
Anyone can see they're not the same by the number of people who are
smart, but not very wise.  And yet intelligence and wisdom do seem
related.  How?What is wisdom?  I'd say it's knowing what to do in a lot of
situations.  I'm not trying to make a deep point here about the
true nature of wisdom, just to figure out how we use the word.  A
wise person is someone who usually knows the right thing to do.And yet isn't being smart also knowing what to do in certain
situations?  For example, knowing what to do when the teacher tells
your elementary school class to add all the numbers from 1 to 100?
[1]Some say wisdom and intelligence apply to different types of
problems—wisdom to human problems and intelligence to abstract
```

It's definitely not cleaned up (wrt separators, random new lines, etc), but it'll do. Locally, once you run 
```
python paul_graham_essay_scraper.py
```
You'll have a `all_pg_essays.txt` file. Then, run

```
python get_token_counts.py
```

To get the following counts (I've added GPT2 as well, because why not):

```
Number of tokens for GPT2: 716928 (716K)
Number of tokens for GPT4: 697456 (697K)
Number of tokens for Llama: 791441 (791K)
Number of tokens for Falcon: 732809 (732K)
```
The vocab sizes for GPT2, GPT4, Llama and Falcon are 50K, 100K, 32K, 64K respectively. You can see the trend in number of tokens follow the inverse of vocab sizes roughly: LLama > Falcon > GPT2 > GPT4. The exact details of merges (all are BPE tokenizers), the training corpus used, as well as the nature of the test corpus used ( Paul Graham's essays are certainly not the same kind of charactersists as datasets like CommonCrawl which are used to train these tokenizers), etc can affect the numbers you see. But you can see that the numbers are actually very close to each other! If all of the extra tokens in GPT-4 were dedicated for English text, you would certainly see a much bigger decrease in number of tokens. 


## Tokenizer Puzzle 2: White spaces matter