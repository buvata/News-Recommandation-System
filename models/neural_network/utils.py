# -*- coding: utf-8 -*- 
import random, pymysql, six
import re, os, sys
sys.path.append('src/models/')
import unicodedata
import time, json
import numpy as np
from numpy import nan
from nltk import word_tokenize
from nltk.corpus import reuters
# from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection.iterative_stratification import IterativeStratification
from sklearn.utils import class_weight
# import matplotlib.pyplot as plt
# from matplotlib.legend import Legend
from six.moves import cPickle as pkl

# cachedStopWords = stopwords.words("english")

UNK = "<UNK>"
PAD = "<PAD>"
def average_ranking_score(Y_true, Y_score, ks=[1,3,5]):
    precisions = dict((k,0) for k in ks)
    recalls = dict((k,0) for k in ks)
    for y_true, y_score in zip(Y_true, Y_score):
        sample_precision = ranking_score(y_true, y_score, ks, 'p')
        sample_recall = ranking_score(y_true, y_score, ks, 'r')
        for k in ks:
            precisions[k] += sample_precision[k]
            recalls[k] += sample_recall[k]
    bs = len(Y_true)
    precisions = dict((k, float(precisions[k])/bs) for k in ks)
    recalls = dict((k, float(recalls[k])/bs) for k in ks)
    return precisions, recalls
    
def ranking_score(y_true, y_score, ks=[1], metric='p'):
    """Precision/Recall at rank k [https://gist.github.com/mblondel/7337391]
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    ks : int-list,
        Ranks.
    Returns
    -------
    dict object contain correspond to ks
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    results = dict()
    for k in ks:
        y_true_temp = np.take(y_true, order[:k])
        n_relevant = np.sum(y_true_temp == pos_label)
        # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
        if metric == 'p':
            results[k] = float(n_relevant) / k 
        elif metric == 'r':
            results[k] = float(n_relevant) / n_pos
        else: print("wrong metric")
    return results

def get_word_index(word, vocab):
    if word in vocab:
        return vocab[word].index
    else:
        return vocab[UNK].index


def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def vectorize(docs, vocab, need_padded=True, max_kernel_size=10, max_doc_len=500):
    """
    - truncate doc length to max_doc_len
    - mapping word to index
    """
    res = []
    padded_doc_idx = []
    for idx, sentences in enumerate(docs):
        words = " ".join(sentences).strip().split()
        if len(words) < max_kernel_size and need_padded:
            padded_doc_idx.append((idx, len(words)))
            words += [PAD] * (max_kernel_size - len(words))
        # elif len(words) > max_doc_len:
        #     words = words[:max_doc_len]
        res.append([get_word_index(w, vocab) for w in words])
    return res, padded_doc_idx


def padding_truncate_list(arr, min_size=5, size=500, padding_value=0):
    size = max(size, min_size)
    res_arr = [0] * len(arr)
    for i in range(len(arr)):
        if len(arr[i]) >= size:
            res_arr[i] = arr[i][:size]
        else:
            res_arr[i] = arr[i] + [padding_value] * (size - len(arr[i]))
    return res_arr

def create_minibatch(samples, batch_size, shuffle=False, drop_tail=False):
    labels = np.array(samples[1])
    num_samples = len(samples[0])

    batches = []
    pseudo_index = [i for i in range(num_samples)]
    if shuffle: random.shuffle(pseudo_index)
    num_batches = num_samples // batch_size
    if drop_tail:
        if num_samples - num_batches * batch_size > 1:
            # ignore batch_size with 1 sample, because of BatchNorm error
            num_batches += 1
    else:
        if num_samples - num_batches * batch_size != 0:
            num_batches += 1

    for i in range(num_batches):
        st = i * batch_size
        en = st + batch_size
        batch_idx = pseudo_index[st:en]
        inputs = [samples[0][i] for i in batch_idx]
        outputs = labels[batch_idx]
        batches.append((inputs, outputs))
    return batches, pseudo_index

def word_vectors_2_embedding(wv):
    return np.array([wv[word] for word in wv.index2word])


def print_params(params, excludes=[]):
    if 'self' in params:
        print("Parameters: ", params['self'])
    else:
        print("Parameters: ")
    for key, val in sorted(params.items(), key=lambda x: x[0]):
        if key in ['self', '__class__']: continue
        if key in excludes:
            print("\t{}:\t{}".format(key, type(val)))
        else:
            print("\t{}:\t{}".format(key, val))

def assign_params(obj, params):
    for p, v in params.items():
        if p in ['self', '__class__']: continue
        setattr(obj, p, v)


def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def load_stopwords(path):
    """
    Return:
        stopwords: set
    """
    stop_words = []
    with open(path, 'r') as f:
        for word in f.readlines():
            word = word.strip()
            stop_words.append(word)
    return set(stop_words)

def sylabelize(text):
    """
    copyleft pyvi
    """
    text = unicodedata.normalize('NFC', text)

    specials = ["==>", "->", "\.\.\.", ">>"]
    digit = "\d+([\.,_]\d+)+"
    email = "(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    web = "http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    datetime = [
        "\d{1,2}\/\d{1,2}(\/\d+)?",
        "\d{1,2}-\d{1,2}(-\d+)?",
    ]
    word = "\w+"
    # non_word = "[^\w\s]"
    abbreviations = [
        "[A-ZĐ]+\.(?:[A-ZĐ]+\.*)*",
        "Tp\.",
        "Mr\.", "Mrs\.", "Ms\.",
        "Dr\.", "ThS\."
    ]

    patterns = []
    patterns.extend(abbreviations)
    patterns.extend(specials)
    patterns.extend([web, email])
    patterns.extend(datetime)
    patterns.extend([digit, word])
    patterns = "(" + "|".join(patterns) + ")"

    if sys.version_info < (3, 0):
        patterns = patterns.decode('utf-8')
    tokens = re.findall(patterns, text, re.UNICODE)

    return [token[0] for token in tokens]

def normalize_text_vietnamese(sentence):
    toks = sylabelize(sentence.decode('utf-8'))
    pattern = re.compile(r"[\W]+", flags=re.UNICODE)
    vietnamese_regex = re.compile(
        r"[a-zA-ZĐ0-9_.,\u00C0-\u00C3\u00C8-\u00CA\u00CC-\u00CD\u00D0\u00D2-\u00D5\u00D9-\u00DA\u00DD\u00E0-\u00E3\u00E8-\u00EA\u00EC-\u00ED\u00F2-\u00F5\u00F9-\u00FA\u00FD\u0102-\u0103\u0110-\u0111\u0128-\u0129\u0168-\u0169\u01A0-\u01A3\u1EA0-\u1EF9]+",
        flags=re.UNICODE
    )
    # https://int3ractive.com/2010/06/optimal-unicode-range-for-vietnamese.html
    temp = [word.strip("_.") for word in toks
            if vietnamese_regex.match(word) and not pattern.match(word) and len(word.strip("_.")) > 1]
    res = " ".join(temp).lower().encode('utf8')
    return res

def normalize_text(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = " ".join([w for w in string.split() if len(w) > 1])
    return string.strip().lower()

def get_sentences(items, columns = ['title', 'content']):
    results = {}
    count = 0
    # print 'Number of items: {}'.format(len(items))
    for newsId, item in list(items.items()):
        content_pieces = [item[column].strip('. ') for column in columns]
        content = ' . '.join(content_pieces)
        normalized_sentences = normalize_text(content)
        results[newsId] = normalized_sentences
        count += 1
        if count % 100 == 0:
            print('\r\033[K\rCalculating item sentences {}/{}'.format(count, len(items))),
            sys.stdout.flush()
    return results

def load_soha(path):
    pass

def get_word_index(wv, token):
    if token not in wv.vocab:
        token = UNK
    return wv.vocab[token].index

# columns = ['newsId', 'title_token', 'sapo_token', 'content_token', 'tag_token']
def pre_process_dataset(docs_sentences, wv, min_doc_length):
    batch = {}
    count = 0
    # # DEBUG
    # pickle.dump(docs_sentences, open(path_normalized_sentences, 'wb'))
    # # END
    for newsId, sentences in docs_sentences.items():
        if count % 100 == 0 or count == len(docs_sentences):
            print('\r\033[K\rPre-process items {}/{}'.format(count, len(docs_sentences))),
            sys.stdout.flush()
        count += 1
        words = [w for s in sentences for w in s.strip().split()]
        if len(words) < min_doc_length:
            # batch[newsId] = 0
            continue
        words_indices = [get_word_index(wv, word) for word in words]
        batch[newsId] = words_indices
    return batch

def create_word2vec_model_from_pretrain_model(vocabulary_inv, embedding_model, num_features=300):
    # embedding_weights = [embedding_model[w] if w in embedding_model
    #                         else np.random.uniform(-0.25, 0.25, num_features)
    #                     for w in vocabulary_inv]
    embedding_weights = []
    cnt = 0
    for w in vocabulary_inv:
        if w in embedding_model:
            cnt += 1
            embedding_weights.append(embedding_model[w])
        else:
            embedding_weights.append(np.random.uniform(-0.25, 0.25, num_features))
    print("#pretrain words: {}/{}={:.2f}".format(cnt, len(vocabulary_inv), cnt * 1./len(vocabulary_inv)))
    embedding_weights = np.array(embedding_weights).astype('float64')
    return embedding_weights


def train_val_split(samples, val_size=0.5):
    """split multilabel datset
    
    Arguments:
        samples {tuple} -- (input, target): target is vector with same dimension with number of class
    
    Keyword Arguments:
        val_size {float} -- ratio of validation set (default: {0.5})
    
    Returns:
        (train_set, val_set) -- 2 datasets
    """

    np.random.seed(42)
    X = np.array(samples[0]).reshape(-1, 1)
    y = np.array(samples[1])

    stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[1 - val_size, val_size])
    val_indexes, train_indexes = next(stratifier.split(X, y))
    X_train, y_train = X[train_indexes, :], y[train_indexes, :]
    X_val, y_val = X[val_indexes, :], y[val_indexes, :]

    X_train, X_val = X_train.reshape(len(X_train), ), X_val.reshape(len(X_val))
    # because of the size of val set is larger than that of train set
    # so we use val for train and vice versa
    print("#train: {}, #val: {}".format(len(X_train), len(X_val)))
    return (X_train, y_train), (X_val, y_val)

def train_test_split(samples, test_size = 0.3):
    """split single label dataset
    
    Arguments:
        samples {tuple} -- (input, target): target is index of right target
    """
    X = samples[0]
    y = samples[1]
    train_index = []
    test_index = []
    stats = {}
    for idx in range(len(X)):
        cat = y[idx]
        if cat in stats:
            stats[cat].append(idx)
        else:
            stats[cat] = [idx]
    for cat, idxes in stats.items():
        split_point = int(test_size * len(idxes))
        test_index.extend(idxes[:split_point])
        train_index.extend(idxes[split_point:])
    train_set = (X[train_index], y[train_index])
    test_set = (X[test_index], y[test_index])
    return train_set, test_set

def get_item(newsIds):
    DOMAINS = ['GenK']
    # Connect to the database
    connection = pymysql.connect(host='192.168.23.191',
                                 user='recommender',
                                 password='lga5QenoQEuksNy',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

    columns = ['newsId', 'catId']
    sources = ["'{}'".format(domain) for domain in DOMAINS]
    items = dict()
    with connection.cursor() as cursor:
        sql = "SELECT {} from news.news_resource as t1 " \
              "JOIN recsys.news_token as t2 " \
              "ON t1.newsId = t2.news_id WHERE sourceNews IN ({}) ".format(",".join(columns), ",".join(sources)) \
              + "AND t1.newsId IN ({})".format(", ".join(newsIds))
        # print(sql)
        cursor.execute(sql)
        results = cursor.fetchall()
        count = 0
        for row in results:
            item = dict()
            for column in columns:
                if isinstance(row[column], six.string_types):
                    item[column] = "{}".format(row[column].encode('utf-8'))
                else:
                    item[column] = "{}".format(row[column])
            count += 1
            if count % 100 == 0:
                print('\r\033[K\r{}'.format(count)),
                sys.stdout.flush()
            items[str(row['newsId'])] = item
    connection.close()
    return items

def get_cates(ids):
    cates = dict()
    cnt = 0
    try:
        while len(ids) > 0:
            curr_ids = ids[:500]
            ids = ids[500:]
            cates.update(get_item(curr_ids))
            cnt += len(curr_ids)
            if cnt % 100 == 0:
                print('\r\033[K\r{}'.format(cnt)),
                sys.stdout.flush()
            
    except Exception as e:
        print(e)
    finally:
        pkl.dump(cates, open('data/items_cates.genk.pickle', 'wb'))

if __name__ == "__main__":
    items = pkl.load(open('data/items-genk.pickle', 'rb'))
    ids = list(items.keys())
    del items
    get_cates(ids)