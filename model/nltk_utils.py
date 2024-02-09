import numpy as np
import nltk
#nltk.download('punkt')
import os

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    sentence = sentence.encode('utf-8').decode('utf-8')
    return nltk.word_tokenize(sentence)


def stem(word):
    word = word.encode('utf-8').decode('utf-8')
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        w = w.encode('utf-8').decode('utf-8')
        if w in sentence_words: 
            bag[idx] = 1

    return bag