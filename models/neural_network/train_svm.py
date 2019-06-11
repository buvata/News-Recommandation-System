# encoding: utf-8
import pandas as pd
import numpy as np
from scipy import sparse
import os, sys, numpy as np, torch, time
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import argparse
from argparse import Namespace
from six.moves import cPickle as pickle
from pathlib import Path
import utils

parser = argparse.ArgumentParser("Train Neural Network")
# parser.add_argument("--use_pretrain", type=str2bool, default=False, help="Load pretrain model for continue training")
parser.add_argument("--dictionary_path", type=str, default="build/soha/dictionary.pickle")
parser.add_argument("--items_path", type=str, default="data/test_items.pickle")
parser.add_argument("--save_path", type=str, default="build/soha/svm.model", help="path to save trained model")
parser.add_argument("--data_path", type=str, default="build/soha/data.pickle", help="path to save preprocessed data")
parser.add_argument("--save_folder", type=str,  default='build/soha/', help="model to save generated files")
parser.add_argument("--cates_path", type=str, default='./build/soha/categories.pkl')
parser.add_argument("--path_normalized_sents", type=str, default= './build/soha/normalized_sentences.pkl')

args = parser.parse_args()

params = Namespace(
    kernel_sizes=[3,4,5],
    num_kernels=256,
    min_offset=10,
    hidden_size=512,
    n_epochs=40,
    batch_size=32,
    learning_rate=1.0,
    momentum=0.9,
    log_interval=1,
    display_step=1,
    weight_decay=0,
    use_dropout=True,
    min_doc_length=5,
    max_length=30,
    save_path=args.save_path,
    data_path=args.data_path,
    items_path=args.items_path,
    dictionary_path=args.dictionary_path,
    cates_path=args.cates_path,
    path_normalized_sentences = args.path_normalized_sents
)

def load_categories(params): 
    if os.path.exists(params.cates_path): 
        items_cate, cates = pickle.load(open(params.cates_path, 'rb'))
    else:
        items = pickle.load(open(params.items_path, mode='rb'))
        cnt = 0
        cates = dict() 
        items_cate = dict()
        for itemId, item in items.items():
            catId = item['catId']
            if catId not in cates:
                cates[catId] = cnt
                cnt += 1
            items_cate[itemId] = cates[catId]
        del items
        pickle.dump((items_cate, cates), open(params.cates_path, 'wb'))
    print("NUM CATEGORIES: ", len(cates))
    return items_cate, cates

def load_data_for_training(params):
    # get normalized sentences
    if os.path.isfile(params.path_normalized_sentences):
        print("Loading normalized docs ...")
        items = None
        normalized_content = pickle.load(open(params.path_normalized_sentences, 'rb'))
    else:
        print("Loading items ...")
        items = pickle.load(open(params.items_path, mode='rb'))
        normalized_content = utils.get_sentences(items, columns=['title_token', 'sapo_token', 'content_token'])
    print("Preprocessing ...")
    items_cate, cates = load_categories(params)
    return items_cate, cates, normalized_content

def train_with_svm(params):
    items_cate, cates, normalized_content = load_data_for_training(params)
    cnt= 0        

    w_vectorizer = TfidfVectorizer(max_features=10000, analyzer="word")
    c_vectorizer = TfidfVectorizer(max_features=5000, analyzer="char")

    model = LogisticRegression()
    clf = Pipeline([
        ('tfidf', FeatureUnion([
            ('wf', w_vectorizer),
            ('cf', c_vectorizer)
        ])),
        ('model', model)
    ])

    x_train = []
    y_train = []
    for docId, content in normalized_content.items():
        x_train.append(content)
        y_train.append(items_cate[docId])
    # clf.fit(x_train, y_train)
    # pred = clf.predict(x_train)
    # res = accuracy_score(y_train, pred)
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    for cv, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_train_batch = [x_train[idx] for idx in train_index]
        y_train_batch = [y_train[idx] for idx in train_index]
        x_test_batch = [x_train[idx] for idx in test_index]
        y_test_batch = [y_train[idx] for idx in test_index]
        clf.fit(x_train_batch, y_train_batch)
        pred = clf.predict(x_test_batch)
        res = accuracy_score(y_test_batch, pred)
        prob = clf.decision_function(x_test_batch)
        pred_based_prob = np.argmax(prob, axis=1)
        res2 = accuracy_score(y_test_batch, pred_based_prob)
        print("CV: {}, acc: {}, acc2: {}".format(cv, res, res2))

    # res = cross_val_score(estimator=clf, X=x_train, y=y_train, scoring='accuracy', cv=5, n_jobs=-1)
    # print(len(clf.named_steps.tfidf['wf'].vocabulary_))
    # print(len(clf.named_steps.tfidf['cf'].vocabulary_))
    print("RES: ", res)

st = time.time()
train_with_svm(params)
en = time.time()
print("Runtime: ", (en-st) * 1./60)