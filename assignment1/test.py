from nltk.tokenize import RegexpTokenizer
from Vocabulary import Vocabulary
from datasets import load_dataset

test = "The blue dog jumped, but not high."

tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(test)
print(tokens)

dataset = load_dataset("ag_news")
dataset_text =  [r['text'] for r in dataset['train']]
vocab = Vocabulary(dataset_text)
print({k: v for k, v in sorted(vocab.freq.items(), key=lambda item: item[1])})