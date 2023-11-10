# Diving into the HF tokenizer
## What makes up a HF tokenizer?
Well, let's first think about state: what information does a tokenizer need to save? Let's consider a BPE tokenizer. In HuggingFace, you can save a tokenizer by callign the `save_pretained` method. Typically, you will see the following files for a BPE tokenizer:
- added_tokens.json: Part of the older format for saving HF tokenizers. A little hard to figure out what this is for, since we have an "added_tokens" entry in the tokenizer.json file itself. Further, this doesn't actually have all the [AddedTokens](https://huggingface.co/docs/tokenizers/api/added-tokens) of your tokenizer (this inc. special tokens for some tokenizers like DeBERTa, Llama). 
- merges.txt : BPE-specific. Contains a list of BPE merge rules to be used while encoding a text sequence. This is for the older tokenizers. 
- special_tokens_map.json: A dictionary of special token attribute names ("bos_token", etc) and their values ("\<BOS\>") and some metadata. What makes special tokens so special? These are commonly used tokens that are not a part of the corpus but have certain important designations (BOS- beginning of sequence, EOS-end of sequence, etc). All of these special tokens are accesible as attributes of the tokenizer directly i.e you can call `tokenizer.eos_token` for any HF tokenizer, since they all subclass the [`SpecialTokensMixin`](https://github.com/huggingface/transformers/blob/ced9fd86f55ebb6b656c273f6e23f8ba50652f83/src/transformers/tokenization_utils_base.py#L795). Maintaining them separately is a good idea for obvious reasons- none of these are actually a part of your training corpus.
- tokenizer_config.json: Some tokenizer specific config parameters such as max sequence length the model was trained on (`model_max_length`), some information on special tokens, etc
- tokenizer.json: Some notable entries:
    - `add_bos_token`: State for whether to add BOS token by default when you call the tokenizer. Caveats on this later. 
    - `added_tokens`: a list of tokens that are special AddedTokens objects. When you call `tokenizer.add_tokens`, the new token added is by default an [AddedToken](https://huggingface.co/docs/tokenizers/api/added-tokens) object, and not just a string. The difference is that an AddedToken can have special behaviour - you might match both " \<ADD\>" and "\<ADD\>" to be the same token, whether the token should be matched in a normalized version of the text, etc. 
- vocab.json : Saved in the older format 


https://github.com/huggingface/transformers/blob/ced9fd86f55ebb6b656c273f6e23f8ba50652f83/src/transformers/tokenization_utils_base.py#L1543


kto be continued: https://github.com/huggingface/transformers/blob/ced9fd86f55ebb6b656c273f6e23f8ba50652f83/src/transformers/tokenization_utils_base.py#L795 