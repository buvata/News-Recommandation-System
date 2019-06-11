import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, filename="build/log.txt")
from gensim.models.word2vec import Word2Vec, FAST_VERSION
import random
import re, os, sys
import unicodedata

import numpy as np
from numpy import nan
from nltk import word_tokenize
from nltk.corpus import reuters
from nltk.corpus import stopwords
from sklearn.utils import class_weight
import gensim

from six.moves import cPickle as pickle

import utils
import itertools
import numpy as np
from collections import Counter

UNK = "<UNK>"
PAD = "<PAD>"

def load_raw_data(params, content_columns=['title_token', 'sapo_token', 'content_token'], target_column='catId'):
    items = pickle.load(open(params.items_path, mode='rb'))
    logging.info("Num doc: {}".format(len(items)))
    X, y = [], []  
    for itemId, item in items.items():
        content_pieces = [item[column].strip('. ') for column in content_columns]
        content = ' . '.join(content_pieces)
        X.append(content)
        y.append(item[target_column])
    class_index = dict()
    for yi in y: 
        if yi not in class_index:
            class_index[yi] = len(class_index)
    y = [class_index[yi] for yi in y]
    return X, y, class_index
    
def build_vocab(sentences, min_count=5, max_size=10000, train_new=False, save_path=""):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    vocab_freq = word_counts.most_common(max_size)
    vocab_freq = [x for x in vocab_freq if x[1] >= min_count]
    num_token = sum([x[1] for x in vocab_freq])
    logging.info("Num token: {}".format(num_token))
    # Mapping from index to word
    vocabulary_inv = [UNK] + [x[0] for x in vocab_freq]
    logging.info("Min frequency: {}".format(min([x[1] for x in vocab_freq])))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    # Train with gensim
    if train_new:
        train_with_gensim(sentences, vocab_freq, save_path=save_path)
    return [vocabulary, vocabulary_inv]

def pad_sentences(sentences, padding_word="<PAD>", pad_type="max", fixed_length=None):
    """
    pad_type: 
        `max` for padding to the max doc length of list
        `fixed` for padding to fixed length
    return
        padded docs in list format
    """
    if pad_type == 'max':
        sequence_length = max(len(x) for x in sentences)
    elif pad_type == 'fixed':
        if fixed_length is None:
            print("fixed_length is required")
            return
        sequence_length = fixed_length

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < sequence_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences

def transform(sents, labels, vocab):
    x = np.array([[vocab[word] if word in vocab else vocab[UNK] for word in sentence] for sentence in sents])
    y = np.array(labels)
    return x, y

def load_data(params, train_new_w2v = False, save_w2v_path="build/gensim.w2v.pickle"):
    """
    - Load training data
    - Normalize
    - Create dictionary
    - Transform to vector
    - Return train set, dictionary, inverted_dicitonary
    """
    X, y, class_index = load_raw_data(params, content_columns=['title_token', 'sapo_token', 'content_token'])
    X_normalized = [utils.normalize_text(X_i).split(" ") for X_i in X]
    X_padded = pad_sentences(X_normalized, padding_word=PAD, pad_type="fixed", fixed_length=params.fixed_length)
    vocab, vocab_inv = build_vocab(X_padded, max_size=params.max_vocab_size, train_new=train_new_w2v, save_path=save_w2v_path)
    x_train, y_train = transform(X_padded, y, vocab)
    return x_train, y_train, vocab, vocab_inv, class_index

def train_with_gensim(corpus, vocab_freq, WORD_EMBEDDING_SIZE = 300, WINDOW_SIZE=5, N_WORKERS=4, save_path="build/gensim.w2v.pickle"):
    ALG_ID = {
        'CBOW': 0,
        'SKIP_GRAM': 1
    }
    corpus_iter = corpus
    vocab_freq = dict(vocab_freq)
    model = Word2Vec(sg=ALG_ID['SKIP_GRAM'], size=WORD_EMBEDDING_SIZE,
                        window=WINDOW_SIZE, workers=N_WORKERS, negative=20, iter=50, sample=1e-5)
    # model.build_vocab([[PAD] * 10, [UNK] * 10])
    # model.build_vocab(corpus_iter, progress_per=1000000, update=True)
    model.build_vocab_from_freq(vocab_freq, corpus_count=len(corpus_iter))
    model.train(corpus_iter, total_examples=model.corpus_count, epochs=model.iter)

    model.wv.syn0[model.wv.vocab[PAD].index] = np.zeros_like(model[PAD])
    model.wv.syn0norm = None
    model.wv.init_sims()
    # Get inverted index for words in dictionary    
    logging.info("Saving dictionary at " + save_path)
    model.save(save_path)
    return model

def load_pretrain_embedding(vocab_inv, w2v_model_path='./data/w2v_models/glove.6B.300d.txt', name="google"):
    # dictionary, where key is word, value is word vectors
    print("Loading {}...".format(w2v_model_path.split('/')[-1]))
    if name == "google":
        dictionary = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
        pretrain_embedding = {w: dictionary.word_vec(w) for w in dictionary.vocab}
    elif name == "gensim":
        dictionary = Word2Vec.load(w2v_model_path)
        pretrain_embedding = {w: dictionary.wv[w] for w in dictionary.wv.vocab}
    elif name == "preprocess":
        basedir = os.path.dirname(w2v_model_path)
        wordpath = os.path.join(basedir, "words.dat")
        with open(wordpath, 'r') as f:
            words = f.read().split("\n")
        embeddings = pickle.load(open(w2v_model_path, 'rb'))
        pretrain_embedding = {words[i]: embeddings[i] for i in range(len(words))}
    else:
        pretrain_embedding = {}
        for line in open(w2v_model_path, 'r'):
            tmp = line.strip().split()
            word, vec = tmp[0], list(map(float, tmp[1:]))
            if word not in pretrain_embedding:
                pretrain_embedding[word] = vec
    embed_dim = len(list(pretrain_embedding.values())[0])
    embedding = []
    found = 0
    for w in vocab_inv:
        if w in pretrain_embedding:
            embedding.append(pretrain_embedding[w])
            found += 1
        else:
            embedding.append(np.random.uniform(-0.25, 0.25, embed_dim))
    logging.info("found: {}/{}={}".format(found, len(embedding), found * 1./len(embedding)))
    return np.array(embedding)

if __name__ == "__main__":
    # res, label_descriptions, num_label = load_reuters()
    # tsv2matrix('build/train_cooccur.tsv')
    pass