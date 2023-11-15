# Numbers and Tokenization

https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/ 

what is an ideal way to tokenize numbers? Well, the main reference we have is the way we represent numbers. In the decimal number system, we assign unique symbols to numbers 0 to 9, and then all other numbers can be represented using these symbols (along with the "." for fractinal parts). So, one expectation you could have is that tokenization should also follow this uniformity (along with a special token for continuity of a number, like "##" for BERT). that is far from the case. Let's take T5, for example.

# Multilingual tokenization

Being able to process text in more than one language is an essential component of many NLP applications. For example, Whisper, OpenAI's speech-to-text model (which is a plain old encoder-decoder transformer model) [can process English and Chinese speech flawlessly](https://x.com/jeremyphoward/status/1721696652506100175?s=20). Of course, this is not a plain text-in -> text-out application, but the point is that tokenization in other languages, especially low-resource languages is comes with it's own set of challenges. The first, and probably biggest challenge is lack of high-quality training data for many languages. Languages like English, Chinese, Russian, etc have a lot more content available on the internet than say, Kannada and Swahili. There are also fundamental differences across languages, such as the absence of a typographic separator (Ex: a whitespace in English) in some languages like Chinese and Japanese.

If you're building a machine-translation system, then one approach can be to learn one tokenizer per language. For example, you can imagine an encoder-decoder transformer trained to translate English to French. In this case, your input text will be tokenized and numericalized based on, say, a BPE tokenizer trained on English text. You would retrieve appropriate embeddings for each token, pass it through the Transformer. At the decoder's output, you select the most probably class/ID for each position in the output sequence, where IDs are based on the vocabulary of a BPE tokenizer trained in French. When decoded tokens are fed back into the decoder, you make use of a different, output embedding layer that maps decoded tokens to embeddings which are passed to the decoder.

Of course, one would want to build multilingual models that can translate between a lot more than 2 languages (and more than just 1 way translation). In this case, a simple approach  would be to mix all available data for all the languages. 

## Metrics
Efforts such as sacrebleu. However, these tokenizers do not extend to many languages. Goyal et al. (2022) propose spBLEU, a BLEU metric based on a standardized SentencePiece model (SPM) covering 101 languages, released with Flores- 101. In this work, we provide SPM-200 along with Flores-200 to enable measurement of spBLEU.28 We describe this in greater detail in Section 8

# No Language Left Behind

NLLB was a massive effort from Meta AI to improve machime translation models for low-resouce languages. This is the first time we crossed the 200 language count in terms of datasets and models available. Key contributions include new datasets, models and benchmarks, focusing on languages never targeted at scale before. To break this down, they first conducted surveys of native speakers in different low resource languages to understand their needs reg. machine translation, then developed an automatic data generation pipeline focusing on said languages. They utilized smart data mining techniques (essentially, an improved version of [bitext mining](https://paperswithcode.com/task/cross-lingual-bitext-mining)) to collect quality training data. Using this mined data along with human-translated seed data, they trained multilingual Mixtures-of-Experts models (specifically, sparsely gated MoEs).

The full NLLB paper is 192 pages long! Needless to say, even a good summary of their contributions is going to be pretty long. Let's get back to our centre of discussion: tokenization. 

## Why tokenization matters

 The results for the languages with the lowest performance in Figure 28, i.e., Hindi (hin_Deva), Kannada (kan_Knda), Maithili (mai_Deva), Telugu (tel_Telu), and Magahi (mag_Deva), may be partially explained by the fact that the scripts in which these languages are written are not always adequately tokenized by our detectors.

.. https://github.com/gordicaleksa/Open-NLLB
