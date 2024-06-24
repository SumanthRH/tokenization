# Chat Templates

This is a short section on chat templates and some tokenization gotchas you should be aware of while fine-tuning. Let's focus on two of the most popular open-source models: Mistral-8B and Llama-3. Chat templating is simple: Given a conversation between the user and the assistant i.e a list of messages , you want to apply a template to convert this into plain text input for the language model. Each "template" consists of special indicators/tags for different roles along with your regular BOS and EOS tokens.

Let's consider the example chat template for `mistralai/Mistral-7B-Instruct-v0.1` (from the[ 🤗 docs](https://huggingface.co/docs/transformers/main/en/chat_templating)):

```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

print(tokenizer.apply_chat_template(chat, tokenize=False))
# <s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]
```

The output is as follows:

<span style="color: #ccf2ff;">&lt;s&gt;[INST]</span><span style="background-color: #ccf2ff;"> </span>Hello, how are you?<span style="background-color: #ccf2ff;"> </span><span style="color: #ccf2ff;">[/INST]&lt;/s&gt;</span><span style="background-color: #ccf2ff;"> </span>I'm doing great. How can I help you today?<span style="background-color: #ccf2ff;"> </span><span style="color: #ccf2ff;">&lt;s&gt;[INST]</span><span style="background-color: #ccf2ff;"> </span>I'd like to show off how chat templating works!<span style="background-color: #ccf2ff;"> </span><span style="color: #ccf2ff;">[/INST]&lt;/s&gt;</span>


I've colored characters added by the template in light blue. Coming back to tokenization, the central issue is as follows:  
- Typically, we want to have different `labels` for different roles in the message. While fine-tuning a chat model, you'd want to ignore all the messages from the user/system and compute the loss for the model only on the assistant messages. Specifically, for building the `labels` sequence for the given input ids, you want to use the label ignore token -100 for user/system tokens while copying assistant tokens, so that Pytorch's cross entropy loss function will ignore user/system tokens. Unfortunately, `tokenizer.apply_chat_template` is not enough (as of June 2024) since the default behaviour just tokenizes the input after application of the chat template, which means you lose role information. 
- This means that you have to apply the chat template youself. You'll have to format the input conversation messsage by message and compute the `labels` entry for each message in parallel. 


## Applying the chat template message by message
The issue with doing this message by message formatting yourself is that there are many places where you can make a mistake. But first, we need to go over two simple properties of tokenization.

**Tokenization is not invertible**: This is especially important to understand while trying to design tests and comparing your implementation to a reference. The obvious issue is normalization and pre-tokenization steps that can lead to loss of information (whitespaces, accents, etc). Thus, when talking about "correctness" the equation to have in mind is something like this:
```
my_token_ids == tokenizer.encode(reference_text, add_special_tokens=False)
```

instead of something like this:

```
tokenizer.decode(my_token_ids) == reference_text
```

Fun fact: For Mistral (and Llama-2), there's an interesting bug where even without loss of information in normalization/pre-tokenization, the second equality doesn't hold:

```
tokenizer.decode(tokenizer.encode("<s>[INST]", add_special_tokens=False)) == "<s> [INST]"
```
Note the extra space added after the BOS token (happens with `legacy=True` as well for Llama-2 btw). 

**Tokenizing message by message and then concatenating is not the same as tokenizing the concatenate messages.** : The more general rule from which this comes from: Tokenizing segments of text and then concatenating is not the same as tokenizing the combined text. For the latter statement, a simple example is: 
```
tokenizer.tokenize("man") + tokenizer.tokenize("go") != tokenizer.tokenize("mango")
```

The first sequence will be `["_man", "_go"]` while the second yields `["_m", "ango"]` (Consider the same Mistral tokenizer). In this case, the two segments joined such that at the borders, characters combined to yield a new token. Coming back to the case with messages, while a case like the above is rare, you will still notice the following with Llama-2 and Mistral:

`tokenizer.tokenize("[INST]") + tokenizer.tokenize("hi") == tokenizer.tokenize("[INST] hi")` (Notice the extra space in the middle)

When you combine tokenized sequences, an additional space gets added in the text space here. This means that, your code for chat templating (when you do it message by message) will be wrong if you're not careful!






