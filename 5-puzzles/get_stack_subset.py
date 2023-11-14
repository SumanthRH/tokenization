from datasets import load_dataset
from transformers import AutoTokenizer
import tiktoken
from functools import partial
from dataclasses import dataclass

@dataclass
class Tokenizers:
    gpt2_tokenizer: AutoTokenizer
    gpt4_tokenizer: tiktoken.Encoding
    llama_tokenizer: AutoTokenizer
    falcon_tokenizer: AutoTokenizer

def tokenized_lengths(examples, tokenizers: Tokenizers):
    examples["gpt2_length"] = [len(t) for t in tokenizers.gpt2_tokenizer(examples["content"])["input_ids"]]
    
    examples["gpt4_length"] = [len(t) for t in tokenizers.gpt4_tokenizer.encode_batch(examples["content"], disallowed_special=())]
    
    examples["llama_length"] = [len(t) for t in tokenizers.llama_tokenizer(examples["content"])["input_ids"]]
    
    examples["falcon_length"] = [len(t) for t in tokenizers.falcon_tokenizer(examples["content"])["input_ids"]]
    return examples



if __name__ == "__main__":
    data_files = [f"data/python/train-0000{i}-of-00206.parquet" for i in range(5)] 
    dataset = load_dataset("bigcode/the-stack",data_files=data_files)
    dataset = dataset["train"] # by default a train split is created
    print("Number of examples:", len(dataset))
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt4_tokenizer = tiktoken.encoding_for_model("gpt-4")
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
    falcon_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b")

    tokenizers = Tokenizers(gpt2_tokenizer, gpt4_tokenizer, llama_tokenizer, falcon_tokenizer)

    mapper = partial(tokenized_lengths, tokenizers=tokenizers)

    dataset = dataset.map(mapper, batched=True)
    gpt2_tokens = sum(dataset["gpt2_length"])
    print(f"Number of GPT2 tokens: {gpt2_tokens:,}")

    gpt4_tokens = sum(dataset["gpt4_length"])
    print(f"Number of GPT4 tokens: {gpt4_tokens:,}")

    llama_tokens = sum(dataset["llama_length"])
    print(f"Number of Llama tokens: {llama_tokens:,}")

    falcon_tokens = sum(dataset["falcon_length"])
    print(f"Number of Falcon tokens: {falcon_tokens:,}")