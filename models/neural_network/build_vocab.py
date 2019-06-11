# -*- coding: utf-8 -*-

from gensim.models.word2vec import Word2Vec, FAST_VERSION
import multiprocessing
from six.moves import cPickle as pickle
import re
import sys
import logging
import utils
import argparse
import os
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print 'Current gensim FAST_VERSION: %d' % FAST_VERSION
assert FAST_VERSION > -1

UNK = "<UNK>"
PAD = "<PAD>"

N_WORKERS = max(1, multiprocessing.cpu_count())
WORD_EMBEDDING_SIZE = 300
WINDOW_SIZE = 5
ALG_ID = {
    'CBOW': 0,
    'SKIP_GRAM': 1
}


# columns = ['newsId', 'title_token', 'sapo_token', 'content_token', 'tag_token']


class DictionaryGenerator(object):
    def __init__(self, docs, stopwords=set(), min_count=5, max_count=0.7):
        self.docs = docs
        self.min_count = min_count
        self.max_count = max_count
        self.sentences = []
        self.count = {}
        self.doc_count = len(docs)
        self.stopwords = stopwords
        self.inverted_index = dict()
        for newsId, content in docs.items():
            doc_words = content.strip().split()
            uniq_words = set(doc_words)
            for word in uniq_words:
                if word in self.inverted_index:
                    self.inverted_index[word].append(newsId)
                else:
                    self.inverted_index[word] = [newsId]
                # unique word
                if word in self.count:
                    self.count[word] += 1
                else:
                    self.count[word] = 1

    def format_word(self, word):
        if self.count[word] < self.min_count or self.count[word] > self.max_count * self.doc_count:
            word = UNK
        return word

    def __iter__(self):
        for newsId, content in self.docs.items():
            yield [self.format_word(word) for word in content.strip().split()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default="build/soha/dictionary.pickle",
                        help="path to save word2vec model and special words")
    parser.add_argument('--data-path', type=str, default="data/test_items.pickle", help="item file which data taken from")
    parser.add_argument('--stopwords', type=str, default="data/vietnamese-stopwords.txt", help="stopwords")
    args = parser.parse_args()

    stopwords = utils.load_stopwords(args.stopwords)

    if os.path.isfile(args.save_path):
        model = Word2Vec.load(args.save_path)
    else:
        items = pickle.load(open(args.data_path, 'rb'))
        normalized_content_path = os.path.join(os.path.dirname(args.save_path), 'normalized_sentences.pkl')
        if os.path.exists(normalized_content_path):
            normalized_content = pickle.load(open(normalized_content_path, 'rb'))
        else:
            normalized_content = utils.get_sentences(items, columns=['title_token', 'sapo_token', 'content_token'])
            del items
            # DEBUG
            pickle.dump(normalized_content, open(normalized_content_path, 'wb'))

        sentence_iterator = DictionaryGenerator(
            normalized_content,
            stopwords=stopwords)
        print "\nNumber of sentences: %d" % len(sentence_iterator.sentences)

        model = Word2Vec(sg=ALG_ID['SKIP_GRAM'], size=WORD_EMBEDDING_SIZE,
                         window=WINDOW_SIZE, workers=N_WORKERS, negative=20, iter=30, sample=1e-5)
        model.build_vocab([[PAD] * 10, [UNK] * 10])
        model.build_vocab(sentence_iterator, progress_per=1000000, update=True)
        model.train(sentence_iterator, total_examples=model.corpus_count, epochs=model.iter)

        model.wv.syn0[model.wv.vocab[PAD].index] = np.zeros_like(model[PAD])
        model.wv.syn0norm = None
        model.wv.init_sims()
        # Get inverted index for words in dictionary
        inverted_index = {word: sentence_iterator.inverted_index[word] for word in model.wv.vocab if word != PAD and word != UNK}
        with open('build/inverted_index.pickle', 'wb') as f:
            pickle.dump(inverted_index, f)
        # del sentences
        del sentence_iterator

        print "Saving dictionary at " + args.save_path
        model.save(args.save_path)

    word_vectors = model.wv
    del model
    print "Done. Vocabulary size is: %d" % len(word_vectors.vocab)
