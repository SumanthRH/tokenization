# Numbers and Tokenization

https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/ 

what is an ideal way to tokenize numbers? Well, the main reference we have is the way we represent numbers. In the decimal number system, we assign unique symbols to numbers 0 to 9, and then all other numbers can be represented using these symbols (along with the "." for fractinal parts). So, one expectation you could have is that tokenization should also follow this uniformity (along with a special token for continuity of a number, like "##" for BERT). that is far from the case. Let's take T5, for example.
