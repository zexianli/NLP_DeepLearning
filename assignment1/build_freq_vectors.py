
import token
import tokenize
from datasets import load_dataset
from Vocabulary import Vocabulary
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.utils.extmath import randomized_svd
import logging
import itertools
from sklearn.manifold import TSNE

import random
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class UnimplementedFunctionError(Exception):
    pass

np.seterr(divide='ignore', invalid='ignore')


###########################
## TASK 2.2              ##
###########################

def compute_cooccurrence_matrix(corpus, vocab):
    """
        
        compute_cooccurrence_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns 
        an N x N count matrix as described in the handout. It is up to the student to define the context of a word

        :params:
        - corpus: a list strings corresponding to a text corpus
        - vocab: a Vocabulary object derived from the corpus with N words

        :returns: 
        - C: a N x N matrix where the i,j'th entry is the co-occurrence frequency from the corpus between token i and j in the vocabulary

        """ 

    N = vocab.size
    C = np.zeros(shape=(N, N))
    k = 3

    for line in corpus:
        tokens = vocab.tokenize(line)
        for i in range(k, len(tokens) - k):
            i_word = vocab.word2idx[tokens[i]]
            for j in range(i - k, i + k):
                j_word = vocab.word2idx[tokens[j]]
                C[i_word, j_word] += 1

    return C
    
    

###########################
## TASK 2.3              ##
###########################

def compute_ppmi_matrix(corpus, vocab):
    """
        
        compute_ppmi_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns 
        an N x N positive pointwise mutual information matrix as described in the handout. Use the compute_cooccurrence_matrix function. 

        :params:
        - corpus: a list strings corresponding to a text corpus
        - vocab: a Vocabulary object derived from the corpus with N words

        :returns: 
        - PPMI: a N x N matrix where the i,j'th entry is the estimated PPMI from the corpus between token i and j in the vocabulary

    """ 

    N = vocab.size
    C = compute_cooccurrence_matrix(corpus, vocab)    
    eps = 0.00001

    PPMI = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            PPMI[i, j] = max(0, np.log(((C[i, j] + eps) * N) / ((C[i, i] + eps) * (C[j, j] + eps))))

    return PPMI

    

################################################################################################
# Main Skeleton Code Driver
################################################################################################
def main_freq():

    logging.info("Loading dataset")
    dataset = load_dataset("ag_news")
    dataset_text =  [r['text'] for r in dataset['train']]
    dataset_labels = [r['label'] for r in dataset['train']]


    logging.info("Building vocabulary")
    vocab = Vocabulary(dataset_text[:100])
    vocab.make_vocab_charts()
    plt.close()
    plt.pause(0.01)


    logging.info("Computing PPMI matrix")
    PPMI = compute_ppmi_matrix( [doc['text'] for doc in dataset['train']], vocab)


    logging.info("Performing Truncated SVD to reduce dimensionality")
    word_vectors = dim_reduce(PPMI)


    logging.info("Preparing T-SNE plot")
    plot_word_vectors_tsne(word_vectors, vocab)


def dim_reduce(PPMI, k=16):
    U, Sigma, VT = randomized_svd(PPMI, n_components=k, n_iter=10, random_state=42)
    SqrtSigma = np.sqrt(Sigma)[np.newaxis,:]

    U = U*SqrtSigma
    V = VT.T*SqrtSigma

    word_vectors = np.concatenate( (U, V), axis=1) 
    word_vectors = word_vectors / np.linalg.norm(word_vectors, axis=1)[:,np.newaxis]

    return word_vectors


def plot_word_vectors_tsne(word_vectors, vocab):
    word_vectors = np.nan_to_num(word_vectors)
    coords = TSNE(metric="cosine", perplexity=50, random_state=42).fit_transform(word_vectors)

    plt.cla()
    top_word_idx = vocab.text2idx(" ".join(vocab.most_common(1000)))
    plt.plot(coords[top_word_idx,0], coords[top_word_idx,1], 'o', markerfacecolor='none', markeredgecolor='k', alpha=0.5, markersize=3)

    for i in tqdm(top_word_idx):
        plt.annotate(vocab.idx2text([i])[0],
            xy=(coords[i,0],coords[i,1]),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom',
            fontsize=5)
    plt.savefig("word_vecotrs.pdf")
    plt.show()


if __name__ == "__main__":
    main_freq()

