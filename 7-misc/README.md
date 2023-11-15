
## The PostProcessor: Be careful with special tokens
One aspect we didn't get into until now was the postprocessor: typically, certain special tokens are added at the beginning or end (or both) of your sequence. For example, if you encode "Hello there!" with BERT, you get:

```
['[CLS]', 'Hello', 'there', '!', '[SEP]']
```

With T5, you get:

```
['Hello', 'there', '!', '</s>']
```
Here, `</s>` is the EOS token. If you do the same with Llama, you get:

```
['<s>', 'Hello', 'there', '!']
```

Here, `<s>` is the BOS token. Well, why are there such differences between different tokenizers, even though you're just doing the same `tokenizer.encode()` call? 

### What is your task?

The difference in default behaviour is simply because of the task each model was trained on, and of course the settings for special tokens used (T5, for ex, doesn't use a BOS token).

BERT is a encoder for which the CLS token acts as a start-of-sequence token (and for classification tasks, the hidden state produced for this token is used as the sequence embedding). The SEP token (`sep_token`) serves as the separator between two sequences/sentences as well as an end-of-sequence token. For example, for predicting entailment on the [RTE dataset](https://huggingface.co/datasets/glue/viewer/rte), your input would look like `[CLS] sentA [SEP] sentB [SEP]`.

For the task of causal language modelling, you're working with autoregressive models like Llama. Fundamentally, what it is that want? You provide an input prompt to a Llama model, and you expect your Llama to provide a good completion to your prompt by autoregressively generating next tokens. When you see this, it becomes obvious why the default behaviour should not include the EOS token: you are looking for a completion of the current sequence of text, and the last thing you want to do is add an end-of-sequence token here! 

Thus, you have different behaviours depending on the pre-training task and the specific model. That said, let's take a look at what you want to be doing during _fine-tuning_. I'll only cover two popular cases: causal language modelling and instruction-tuning.

## Causal language modelling

Here, for for each element/ piece of text in your dataset, you would want to add an EOS token at the very end of each element. Note that when you chunk your data, you do NOT want to be adding EOS tokens for every output chunk! 

This is different for different tasks: you'd want to add special tokens for instruction tuning, not for causal language modelling, where you typically chunk your data. 


## Instruction-tuning


## What's so special about the EOS token anyway?