from datasets import load_dataset 
from transformers import AutoTokenizer 
import time 
from functools import partial 
tiny_tokenizer = AutoTokenizer.from_pretrained(".")
mname = "microsoft/deberta-base"
tokenizer = AutoTokenizer.from_pretrained(mname, use_fast=True)

dataset = load_dataset("imdb")["train"]

def get_example(ex, tokenizer):
    inps = tokenizer(ex["text"])
    return inps

start_time = time.time()
func = partial(get_example, tokenizer=tokenizer)
mapped = dataset.map(func, batched=True, load_from_cache_file=False)
end_time = time.time()
print("time: ", end_time - start_time)



start_time = time.time()
func = partial(get_example, tokenizer=tiny_tokenizer)
mapped = dataset.map(func, batched=True, load_from_cache_file=False)
end_time = time.time()
print("time: ", end_time - start_time)