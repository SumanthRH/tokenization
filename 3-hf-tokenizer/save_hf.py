# Simple script to save gpt2, BERT and Llama tokenizers
from transformers import AutoTokenizer

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

gpt2_tokenizer.save_pretrained("gpt2")
bert_tokenizer.save_pretrained("bert-base-uncased")
llama_tokenizer.save_pretrained("meta-llama/Llama-2-7b-hf")