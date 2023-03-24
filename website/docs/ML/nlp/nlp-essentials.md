---
sidebar_label: Essentials of NLP
title: Essentials of Natural Language Processing
sidebar_position: 1
---

## Text Normalization

It is a pre-processing step required to be done to improve the quality of text and making it suitable for the machine learning process.

## Main steps of text normalization are:

### Case normalization 
all words are converted to small case.
  
### Tokenization 
Sentences converted to list of tokens (words).

```
"This world shall know pain"
                |
["This", "world", "shall", "know", "pain"]
```

### Stop word removal
    
Removing common words like articles(the, an) and conjunctions (and, but).


### Parts of Speech 
Convert peiece of text and tag it with POS identifier.

Like ADJ for adjectives, ADV for adverd.


### Stemming or lemmatization 

In stemming, the ends are choped off to get the root words. Porter stemmer is a famous stemmer.

In lemmatization, the root words is obtained using morphological information.

### Vectorization  
  
Converting a word to a vector of numbers the contains the information of the words.

The simplest way is to count the number of words.

The more sophisticated approach is TF-IDF.

And the third approach is Word2Vec. It uses RNNs to generate vectors.

