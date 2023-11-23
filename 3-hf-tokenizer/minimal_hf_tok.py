from typing import Any
from transformers.tokenization_utils import Trie
from transformers import AutoTokenizer
import json
from typing import Dict, Tuple, Union, List
import regex as re # regex is cooler than re
from bpe import BPE

EOS_TOKEN = "<|endoftext|>"
class MyTrie(Trie):
    """
    HF's Trie implementation for added tokens. Minor changes for clarity.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.data = {} - our trie graph, stored as a dict of dicts
        # self._tokens = set() - set of tokens in the trie

    def add(self, word: str):
        """
        Adds a word to the trie
        """
        return super().add(word)
    
    def split(self, word: str):
        """
        Splits a word into chunks based on the trie
        """
        return super().split(word)
    
    def __repr__(self) -> str:
        # format data graph into a nice dict
        return json.dumps(self.data, indent=4)

class MySlowTokenizer:
    """
    A minimal implementation of HF's slow tokenizer, based on GPT2's tokenizer
    References:
    https://github.com/huggingface/transformers/blob/8aca43bdb3cb9a5020f6d57589d85679dc873b1c/src/transformers/models/gpt2/tokenization_gpt2.py
    """
    def __init__(self, init_vocab_file: str = None):
        self.added_tokens_trie = MyTrie() # trie for added tokens only
        self.bpe = BPE(init_vocab_file)
        self.vocab = self.bpe.token_to_id # nice to have vocab accessible here
        self.byte_encoder = self.bpe.byte_encoder
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.unk_token = EOS_TOKEN

        # Regex for pre-tokenization - breaking up a piece of text into words, splitting at whitespaces, contractions, etc. Borrowed from GPT-2
        self.pattern_for_splitting = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.load_added_tokens()
    
    def load_added_tokens(self):
        # loads added tokens from json and adds them to the trie
        # Hard coded for GPT2 demonstration
        self.added_tokens_trie.add(EOS_TOKEN)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.encode(*args, **kwargs)
    
    def encode(self, text: str, **kwargs: Any) -> Any:
        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        # split text into chunks based on added tokens
        # "This isn't<|endoftext|> what you think" -> ["This isn't", "<|endoftext|>", " what you think"] 
        chunks = self.added_tokens_trie.split(text)
        bpe_tokens = []
        for chunk in chunks:
            if chunk in self.added_tokens_trie._tokens:
                # if chunk is an added token, add it to bpe_tokens
                bpe_tokens.append(chunk)
            else:
                tokens = self._tokenize(chunk)
                bpe_tokens.extend(tokens)
        # Convert tokens to ids
        bpe_tokens = [self.convert_token_to_id(token) for token in bpe_tokens]
        return bpe_tokens

    def decode(self, ids: List[int], **kwargs: Any) -> str:
        tokens = [self.convert_id_to_token(id_) for id_ in ids] # convert ids to tokens
        text = "".join(tokens) # join tokens
        # replace unicode symbols with normal characters
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8")
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        all_tokens = []
        # split each chunk into words based on regex. "This isn't" -> ["This", " isn", "'t"]
        for word in self.pattern_for_splitting.findall(text):
            # " isn" -> bytes object -> "Ġisn"
            word = "".join([self.byte_encoder[b] for b in word.encode("utf-8")]) 
            tokens = self.bpe(word, dont_byte_encode=True).split(" ") # we already encoded the chunk to unicode strings
            all_tokens.extend(tokens)
        return all_tokens
    
    def prepare_for_tokenization(self, text: str, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        """
        In HF, this method performs any pre-processing needed before tokenization. Dummy function for now.
        """
        # returns text and kwargs
        return (text, kwargs)

    def to_dict(self):
        dict1 = {}
        dict1["model"]["type"] = "BPE"
        dict1["model"]["vocab"] = self.vocab
        dict1["model"]["merges"] = self.bpe.merges
        return dict1
    
    def add_tokens(self, new_tokens: Union[str, List[str]]):
        """
        Adds new tokens to the tokenizer.
        """
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        # add to vocab first
        for token in new_tokens:
            self.bpe.add_token(token)
            self.added_tokens_trie.add(token)
            print(f"Added {token} to the vocabulary.")
    
    def convert_token_to_id(self, token: str) -> int:
        """
        Converts a token to its id. Returns unk token id if token is not in vocab.
        """
        return self.vocab.get(token, self.vocab[self.unk_token])

    def convert_id_to_token(self, index: int) -> str:
        """
        Converts an id to its token. Returns unk token if id is not in vocab.
        """
        return self.bpe.id_to_token.get(index, self.unk_token)

    def __repr__(self) -> str:
        string = "MySlowTokenizer("
        string += f"vocab_size={self.nvocab}, unk_token={self.unk_token}, added_tokens={str(self.added_tokens_trie._tokens)}, "
        string += f"bpe={str(self.bpe)})"
        return string
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

if __name__ == "__main__":
    input_text = "This isn't<|myspecialtoken|> that simple"
    new_token = "<|myspecialtoken|>"
    my_tokenizer = MySlowTokenizer("./2-bpe/vocab.json")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    my_tokenizer.add_tokens(new_token)
    gpt2_tokenizer.add_tokens(new_token)
    my_enc = my_tokenizer.encode(input_text)
    gpt2_enc = gpt2_tokenizer.encode(input_text)
    
    print("Input text:", input_text)
    print("My tokenizer encoding:", my_enc)
    print("GPT2 tokenizer encoding:", gpt2_enc)

    print("My tokenizer decoding:", my_tokenizer.decode(my_enc))
    print("GPT2 tokenizer decoding:", gpt2_tokenizer.decode(gpt2_enc))
