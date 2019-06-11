# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import pandas as pd 
import numpy as np 
import logging
import re
import pickle
import test_LDA 
from bs4 import BeautifulSoup
from markdown import markdown
from pyvi import ViTokenizer
import sys
import config
reload(sys)
sys.setdefaultencoding('utf8')

sys.path.append("../../")

cat2int=config.cat2int
set_cat=list(cat2int.keys())
list_cat=list(map(str, set_cat))

with open('./Data/stop_words.txt') as f:
    stopwords = []
    for line in f:
        stopwords.append("_".join(line.strip().split()))

def remove_links_content(text):
    text = re.sub(r"http\S+", "", text)
    return text

def remove_multiple_space(text):
    return re.sub("\s\s+", " ", text)

def remove_punctuation(text):
    import string 
    text=re.sub(r"[.,)@(?<>, “”!~#$%*^&+''-/\;:]" ," ",text)
    #table = str.maketrans({key: None for key in string.punctuation})
    #return text.translate(table)
    return text

def remove_numeric(text): 
     return ''.join(c for c in text if c not in "1234567890")

def remove_stopwords(text, stopwords):
    return " ".join([word for word in text.split(" ") if word not in stopwords])

def process_text(text):
    text = remove_links_content(text)
    text = remove_punctuation(text)
    text = text.replace('\n', ' ')
    text = remove_numeric(text)
    text = remove_multiple_space(text)
    text = text.lower().strip()
    #text = ViTokenizer.tokenize(text)
    text = remove_stopwords(text, stopwords=stopwords)
    return text 

def load_LDA(L=15, G=50, vocab_path='./Data/vocab.txt'):
    C = len(cat2int)
    vocab=list()
    with open(vocab_path,'r') as f:
        for line in f:
            vocab.append(line.strip())

    # Initialize the variational distribution q(beta|lambda)
    lambda_local = np.loadtxt("./Data/lambda_local.dat")
    lambda_glob= np.loadtxt("./Data/lambda_glob.dat")
    _lambda=np.loadtxt("./Data/lambda.dat")

    olda = test_LDA.online_LDA(vocab,C, L, G, 1./100, 1./100, 64., 0.6, lambda_local, lambda_glob, _lambda)
    olda.cat2int = cat2int
    return olda

def item_emb(olda,dict_items):
    "input {item_id: {cat_index : , content_token: , cat_id: }}"
    "output {item_id : emb}"
    dict_item=list(dict_items.values())
    dict_item_ids=list(dict_items.keys())
    dict_item=pd.DataFrame.from_dict(dict_item)
    dict_item['content_token']=dict_item['content_token'].apply(lambda x: process_text(str(x)))
    dict_item.update(dict_item['content_token']) 
    dict_item['cat_index'] = dict_item.cat_id.apply(lambda x: cat2int[x]) 
    (embeddings,_,_,_,_,_)=olda.do_e_step(dict_item.content_token.values.tolist(), dict_item.cat_index.values.tolist())     
    embeddings=list(embeddings)
    dict_emb=dict(zip(dict_item_ids,embeddings))
    return dict_emb