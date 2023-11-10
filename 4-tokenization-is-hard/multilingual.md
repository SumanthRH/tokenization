# Multilingual tokenization

Tokenization in other languages, especially low-resource languages is again challenging. Languages like English, Chinese, Russian, etc have a lot more content available on the internet than say, Kannada and Swahili.

## Metrics
 Efforts such as sacrebleu. However, these tokenizers do not extend to many languages. Goyal et al. (2022) propose spBLEU, a BLEU metric based on a standardized SentencePiece model (SPM) covering 101 languages, released with Flores- 101. In this work, we provide SPM-200 along with Flores-200 to enable measurement of spBLEU.28 We describe this in greater detail in Section 8

# No Language Left Behind

NLLB was a massive effort from Meta AI to improve machime translation models for low-resouce languages. This is the first time we crossed the 200 language count in terms of datasets and models available. Ky contributions include new datasetse, models and benchmarks, focusing on languages never targeted at scale before. To break this down, they first conducted surveys of native speakers in different low resource languages, developed an automatic data generation pipeline, 
In terms of models trained they developed a "conditional compute model based on Sparsely Gated Mixture of Experts that is trained on data obtained with novel and effective data mining techniques tailored for low-resource languages". They build Flores-200, a many-to-many (i.e some languages to other languages) multilingual dataset that allows us to measure translation quality through any of the 40,602 total translation directions. We developed a distillation-based sentence encoding technique, LASER3 (Heffernan et al., 2022), that helped us mine web data to create parallel datasets for low-resource languages. Using both mined data and a set of human-translated seed data, we trained multilingual Mixtures-of-Experts models with state of the art performance.

The full NLLB paper is 192 pages long, and I want to relate all of this to our centre of discussion: tokenization. 

## Why tokenization matters

 The results for the languages with the lowest performance in Figure 28, i.e., Hindi (hin_Deva), Kannada (kan_Knda), Maithili (mai_Deva), Telugu (tel_Telu), and Magahi (mag_Deva), may be partially explained by the fact that the scripts in which these languages are written are not always adequately tokenized by our detectors.

