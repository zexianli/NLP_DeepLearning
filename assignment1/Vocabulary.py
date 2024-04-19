from collections import Counter 
from re import sub, compile
import string
from tokenize import tokenize
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


class UnimplementedFunctionError(Exception):
    pass

class Vocabulary:

    def __init__(self, corpus):

        self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
        self.size = len(self.word2idx)

    def most_common(self, k):
        freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
        return [t for t,f in freq[:k]]


    def text2idx(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

    def idx2text(self, idxs):
        return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


    ###########################
    ## TASK 1.1           	 ##
    ###########################
    def tokenize(self, text):
        """
        
        tokenize takes in a string of text and returns an array of strings splitting the text into discrete tokens.

        :params: 
        - text: a string to be tokenize, e.g. "The blue dog jumped, but not high."

        :returns:
        - tokens: a list of strings derived from the text, e.g. ["the", "blue", "dog", "jumped", "but", "not", "high"] for word-level tokenization
        
        """ 

        # tokens = nltk.tokenize.word_tokenize(text)
        stop_words = set(stopwords.words("english") + list(string.digits))
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        filtered_tokens = [t for t in tokens if not t in stop_words]
        
        # self.tokens.extend(filtered_tokens)
        return filtered_tokens


    ###########################
    ## TASK 1.2            	 ##
    ###########################
    def build_vocab(self, corpus):
        """
        
        build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

        :params:
        - corpus: a list string to build a vocabulary over

        :returns: 
        - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK": 129}
        - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129: "UNK"}
        - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog": 102, "the": 18023, ...}

        """ 

        # REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
        word2idx = defaultdict(int)
        idx2word = defaultdict(str)
        freq = defaultdict(int)
        unfiltered_tokens = []
        self.tokens = []
        
        for line in corpus:
            unfiltered_tokens.extend(self.tokenize(line))
        
        for tk in unfiltered_tokens:
            freq[tk] += 1
        
        idx = 0
        for tk in unfiltered_tokens:
            if freq[tk] < 50:
                self.tokens.append("UNK")
            self.tokens.append(tk)
            
        for tk in unfiltered_tokens:
            if tk not in word2idx:
                word2idx[tk] = idx
                idx2word[idx] = tk
                idx += 1
        
        return word2idx, idx2word, freq    
        


    ###########################
    ## TASK 1.3              ##
    ###########################
    def make_vocab_charts(self):
        """
        
        make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details

        
        """  
        
        sorted_tokens = [v for v in sorted(self.freq.values(), reverse=True)]
        id_range = range(0, len(sorted_tokens))
        
        plt.plot(id_range, sorted_tokens)
        plt.title("Token Frequency Distribution")
        plt.ylabel("Frequency")
        plt.yscale("log")
        plt.xlabel("Token ID (sorted by frequency)")
        plt.savefig("frequency")
        plt.clf()
        plt.close()
        
        total_freq = sum(sorted_tokens)
        percentage = np.cumsum(np.array(sorted_tokens) / total_freq)
        
        plt.plot(id_range, percentage)
        plt.title("Cumulative Fraction Covered")
        plt.ylabel("Fraction of Token Occurences Covered")
        plt.xlabel("Token ID (sorted by frequency)")
        plt.savefig("fraction")
        plt.clf()
        plt.close()
        
        
        

