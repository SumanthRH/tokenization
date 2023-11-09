# Diving into the HF tokenizer
## What makes up a HF tokenizer?
Well, let's first think about state: what information does a tokenizer need to save? Let's consider a BPE tokenizer. In HuggingFace, you can save a tokenizer by callign the `save_pretained` method. Typically, you will see the following files for a BPE tokenizer:
- added_tokens.json: Contains 
- merges.txt : BPE-specific. Contains a list of BPE merge rules to be used while encoding a text sequence
- special_tokens_map.json: A dictionary of special token attribute names ("bos_token", etc) and their values ("\<BOS\>") and some metadata. What makes special tokens so special? These are commonly used tokens that are not a part of the corpus but have certain important designations (BOS- beginning of sequence, EOS-end of sequence, etc). All of these special tokens are accesible as attributes of the tokenizer directly i.e you can call `tokenizer.eos_token` for any HF tokenizer, since they all derive the [`SpecialTokensMixin`](https://github.com/huggingface/transformers/blob/ced9fd86f55ebb6b656c273f6e23f8ba50652f83/src/transformers/tokenization_utils_base.py#L795). Maintaining them separately is a good idea for obvious reasons- none of these are actually a part of your training corpus.
- tokenizer_config.json
- tokenizer.json
- vocab.json


https://github.com/huggingface/transformers/blob/ced9fd86f55ebb6b656c273f6e23f8ba50652f83/src/transformers/tokenization_utils_base.py#L1543