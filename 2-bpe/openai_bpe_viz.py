"""
BPE Tokenizer Visualisation from OpenAI: https://github.com/openai/tiktoken 
"""
from tiktoken._educational import *

# Visualise how the GPT-4 encoder encodes text
enc = SimpleBytePairEncoding.from_tiktoken("cl100k_base")
a = enc.encode("hello world aaaaaaaaaaaa")
print("Final tokens: ", [enc.decode([i]) for i in a]) # see the final tokens