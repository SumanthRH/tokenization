from transformers import AutoTokenizer
import argparse
import tiktoken

parser = argparse.ArgumentParser()
parser.add_argument("--file_path" , type=str, default="all_pg_essays.txt")

if __name__ == "__main__":
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt4_tokenizer = tiktoken.encoding_for_model("gpt-4")
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
    falcon_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b")

    args = parser.parse_args()
    file_path = args.file_path
    with open(file_path, "r") as f:
        data = f.read()
    
    gpt2_tokens = len(gpt2_tokenizer.encode(data))
    gpt4_tokens = len(gpt4_tokenizer.encode(data))
    llama_tokens = len(llama_tokenizer.encode(data))
    falcon_tokens = len(falcon_tokenizer.encode(data))
    print(f"Number of tokens for GPT2: {gpt2_tokens}")
    print(f"Number of tokens for GPT4: {gpt4_tokens}")
    print(f"Number of tokens for Llama: {llama_tokens}")
    print(f"Number of tokens for Falcon: {falcon_tokens}")
