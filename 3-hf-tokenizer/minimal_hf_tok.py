from typing import Any
from transformers.tokenization_utils import Trie
import json
from typing import Dict, Tuple, Union


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.

    Copied from: 
    https://github.com/huggingface/transformers/blob/8aca43bdb3cb9a5020f6d57589d85679dc873b1c/src/transformers/models/gpt2/tokenization_gpt2.py#L63
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

class MyTrie(Trie):
    """
    Adds a custom repr method to HF's Trie class.
    Other methods are same as the parent class, but shown here for clarity.
    """
    def __init__(self):
        # data is our trie graph, stored as a nested dict, with characters as keys and dicts as values
        # ex:  {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}
        self.data = {}
        self._tokens = set()
        
    def add(self, word: str):
        super().add(word)
    
    def split(self, word: str):
        super().split(word)

    def __repr__(self) -> str:
        # add a custom repr based on data
        return json.dumps(self.data, ensure_ascii=False, indent=2)

class MySlowTokenizer:
    """
    Slow tokenizer implementation for educational purposes.
    A minimal implementation of HF's tokenizer class, based on GPT2's tokenizer
    References:
    https://github.com/huggingface/transformers/blob/8aca43bdb3cb9a5020f6d57589d85679dc873b1c/src/transformers/models/gpt2/tokenization_gpt2.py
    """
    def __init__(self, init_vocab_file: str = None):
        self.added_tokens_trie = Trie() # trie for added tokens only
        self.nvocab = 0
        self.vocab = dict()
        self.byte_encoder = bytes_to_unicode()
        if init_vocab_file is not None:
            self.load_vocab(init_vocab_file)
        self.load_added_tokens()

    def load_vocab(self, vocab_file: str):
        # loads vocab from json and sets nvocab value
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
        self.vocab = vocab
        self.nvocab = len(vocab)
    
    def load_added_tokens(self):
        # loads added tokens from json and adds them to the trie
        self.added_tokens_trie.add("<|endoftext|>")
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.encode(*args, **kwargs)
    
    def encode(self, text: str, **kwargs: Any) -> Any:
        # returns a list of token ids
        tokens = []
        text, kwargs = self.prepare_for_tokenization(text, **kwargs)
        # split text into chunks based on added tokens
        # "This is not<|endoftext|> what you think" -> ["This is not", "<|endoftext|>", " what you think"] 
        chunks = self.added_tokens_trie.split(text)
        bpe_tokens = []
        for chunk in chunks:
            chunk = "".join([self.byte_encoder[b] for b in chunk.encode("utf-8")])
            tokens = self.bpe(chunk)
            bpe_tokens.append(tokens)
        return tokens
    
    def prepare_for_tokenization(self, text: str, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        """
        In HF, this method performs any pre-processing needed before tokenization.
        """
        # returns text and kwargs
        return (text, kwargs)

    def bpe(self, text: str):
        raise NotImplementedError
