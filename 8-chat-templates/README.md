# Chat Templates

This is a short section on chat templates and some tokenization gotchas you should be aware of. Let's focus on two of the most popular open-source models: Mistral-8B and Llama-3. Chat templating is simple: Given a conversation between the user and the assistant i.e a list of messages , you want to apply a template to convert this into plain text input for the language model. Each "template" consists of special indicators/tags for different roles along with your regular BOS and EOS tokens.

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
- Typically, we want to have different `labels` for different roles in the message. While fine-tuning a chat model, you'd typically want to ignore all the messages from the user/system and compute the loss for the model only on the assistant messages. Specifically, for building the `labels` sequence for the given input ids, you want to use the label ignore token -100 for user/system tokens while copying assistant tokens, so that Pytorch's cross entropy loss function will ignore user/system tokens. Unfortunately, `tokenizer.apply_chat_template` is not enough (as of June 2024) since the default behaviour just tokenizes the input after application of the chat template, which means you lose role information. 
- This means that you are left to maintain your own version of applying the chat template messsage by message and computing the `labels` entry for each message in parallel. 