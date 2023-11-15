# Diving into the HuggingFace tokenizer
## What makes up a HuggingFace tokenizer?
Well, let's first think about state: what information does a tokenizer need to save? 
### BPE Tokenizer
Let's consider a BPE tokenizer. In HuggingFace, you can save a tokenizer by calling the `save_pretained` method. Typically, you will see the following files for a BPE tokenizer:
- [DEPR] added_tokens.json: Part of the older format for saving HF tokenizers. A little hard to figure out what this is for, since we have an "added_tokens" entry in the tokenizer.json file itself. Further, this doesn't actually have all the [AddedTokens](https://huggingface.co/docs/tokenizers/api/added-tokens) of your tokenizer (this inc. special tokens for some tokenizers like DeBERTa, Llama). 
- merges.txt : Saved in the older format for BPE tokenizers. Contains a list of BPE merge rules to be used while encoding a text sequence. This is for the older tokenizers. 
- special_tokens_map.json: A dictionary of special token attribute names ("bos_token", etc) and their values ("\<BOS\>") and some metadata. What makes special tokens so special? These are commonly used tokens that are not a part of the corpus but have certain important designations (BOS- beginning of sequence, EOS-end of sequence, etc). All of these special tokens are accesible as attributes of the tokenizer directly i.e you can call `tokenizer.eos_token` for any HF tokenizer, since they all subclass the [`SpecialTokensMixin`](https://github.com/huggingface/transformers/blob/ced9fd86f55ebb6b656c273f6e23f8ba50652f83/src/transformers/tokenization_utils_base.py#L795) class. Maintaining such additional information is a good idea for obvious reasons- none of these are actually a part of your training corpus. You'd also want to add certain special tokens when you encode a piece of text by default (EOS or BOS+EOS, etc). 
- tokenizer_config.json: Some tokenizer specific config parameters such as max sequence length the model was trained on (`model_max_length`), some information on special tokens, etc
- tokenizer.json: Some notable entries:
    - `add_bos_token`: State for whether to add BOS token by default when you call the tokenizer. Caveats on this later. 
    - `added_tokens`: a list of new tokens added via `tokenizer.add_tokens`. When you call `tokenizer.add_tokens`, the new token added is, by default, maintained as an [AddedToken](https://huggingface.co/docs/tokenizers/api/added-tokens) object, and not just a string. The difference is that an AddedToken can have special behaviour - you might match both " \<ADD\>" and "\<ADD\>" to be the same token, specify whether the token should be matched in a normalized version of the text, etc. 
    - `model`:  Information about the tokenizer architecture/ algorithm ("type" -> BPE for ex). Also includes the vocabulary (mapping tokens -> token ids), and additional state such as merge rules for BPE.Each merge rule is really just a tuple of tokens to merge. Huggingface stores this tuple as one string, space-separated ex: "i am". 
    - `normalizer`: Normalizer to use before segmentation.  
- [DEPR] vocab.json : Saved in the older format. Contains a dictionary mapping tokens to token ids. This information is now stored in `tokenizer.json`. 

### WordPiece tokenizer
WordPiece tokenizer 
https://github.com/huggingface/transformers/blob/ced9fd86f55ebb6b656c273f6e23f8ba50652f83/src/transformers/tokenization_utils_base.py#L1543


kto be continued: https://github.com/huggingface/transformers/blob/ced9fd86f55ebb6b656c273f6e23f8ba50652f83/src/transformers/tokenization_utils_base.py#L795 



## INput Preprocessing
This is different for different tasks: you'd want to add special tokens for intrsuction tuning, not for causal language modelling, where you typically chunk your data. 