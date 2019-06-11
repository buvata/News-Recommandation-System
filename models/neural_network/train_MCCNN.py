# encoding: utf-8
import os, sys, numpy as np, torch, time
import logging
rootLogger = logging.getLogger()
logFormatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
fileHandler = logging.FileHandler("build/log.txt")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)


from six.moves import cPickle as pickle
from pathlib import Path
import json, warnings, argparse
from gensim.models.word2vec import Word2Vec, FAST_VERSION
import torch
import torch.optim as optim
import torch.nn as nn
from argparse import Namespace
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
np.random.seed(42)
import time
from mc_cnn import MultiChannelCNN, YoonKimCNN, XMLCNN
from gated_cnn import GatedCNNWrapper
import utils
from utils import print_params
import load_data_helper


PAD = "<PAD>"
def filter_params(params, _class):
    new_params = dict()
    for key in params:
        if key in _class._get_param_names():
            new_params[key] = params[key]
    return new_params

def _get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def train_one_split(params, cv, X, Y, train_index, test_index, net = None):
    # shuffle train set
    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]

    x_train = torch.from_numpy(x_train).long()
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).long()
    y_test = torch.from_numpy(y_test).long()

    dataset_train = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset_train, batch_size=params.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    dataset_test = TensorDataset(x_test, y_test)
    test_loader = DataLoader(dataset_test, batch_size=params.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    if params.model_type == 'xmlcnn':
        net = XMLCNN(params.vocab_size, embedding_dim=params.embed_dim, kernel_sizes=params.kernel_sizes, 
                num_filters=params.num_kernels, num_classes=len(params.class_index), 
                pretrained_embeddings=params.embedding, sentence_len = params.sentence_len, 
                use_cuda=torch.cuda.is_available())
        criterion = nn.BCEWithLogitsLoss()
    elif params.model_type == 'yoonkim':
        net = YoonKimCNN(params.vocab_size, embedding_dim=params.embed_dim, kernel_sizes=params.kernel_sizes, 
                num_filters=params.num_kernels, num_classes=len(params.class_index), 
                pretrained_embeddings=params.embedding, sentence_len = params.sentence_len, 
                use_cuda=torch.cuda.is_available())
        criterion = nn.CrossEntropyLoss()
    elif params.model_type == 'gatedcnn':
        net = GatedCNNWrapper(kernel_size=3, num_channels=[128], num_classes=len(params.class_index),
                pretrained_embeddings=params.embedding, use_cuda=torch.cuda.is_available())
        criterion = nn.BCEWithLogitsLoss()

    if cv == 0: logging.info(net)
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    # optimizer = optim.Adadelta([{'params': parameters}], lr=params.learning_rate, rho=0.9, eps=1e-06)
    optimizer = torch.optim.Adam(parameters, lr=params.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=1, min_lr=1e-3, verbose=True)

    if os.path.exists(params.save_path):
        logging.info("Loading pretrain model ...")
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(params.save_path))
        else:
            net.load_state_dict(torch.load(params.save_path, map_location=lambda storage, loc: storage))

    for epoch in range(params.n_epochs):
        st = time.time()
        net.train()
        acc_loss = 0.0
        train_acc = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = Variable(inputs), Variable(targets)
            if params.model_type in ['xmlcnn', 'gatedcnn']:
                targets_onehot = torch.FloatTensor(len(targets), len(params.class_index)).zero_()
                targets_onehot.scatter_(1, targets.view(-1, 1), 1)
                if torch.cuda.is_available():
                    targets_onehot = targets_onehot.cuda()
            
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = Variable(inputs), Variable(targets)
            # zero the parameter gradients
            logits, predictions = net(inputs)
            if params.model_type in ['xmlcnn', 'gatedcnn']:
                loss = criterion(logits, targets_onehot)
            elif params.model_type == 'yoonkim':
                loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            # torch.nn.utils.clip_grad_norm(net.network.cnn.parameters(), 0.25)
            acc_loss += batch_loss
            train_acc += (predictions.data == targets).sum().item()
            if batch_idx % params.log_interval == 0:
                print('\r\033[K\rTrain Epoch: {} [{} / {} ({:.0f}%)]   Learning Rate: {}   Loss: {:.6f}'
                        .format(epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader),
                                _get_learning_rate(optimizer)[0], batch_loss)),
                sys.stdout.flush()

        acc_loss /= len(train_loader)
        train_acc *= 1./len(x_train)

        # Save the model if the validation loss is the best we've seen so far.
        if not scheduler.best or scheduler.is_better(acc_loss, scheduler.best):
            with open(params.save_path, 'wb') as f:
                torch.save(net.state_dict(), f)
        scheduler.step(acc_loss)

        # Testing
        net.eval()
        eval_acc = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = Variable(inputs), Variable(targets)
            # zero the parameter gradients
            logits, predictions = net(inputs)
            eval_acc += (predictions.data == targets).sum().item()
        eval_acc *= 1.0/ len(x_test)
        # print statistics
        if epoch % params.display_step == 0:
            logging.info('\n[{:3d}] loss: {:.5f} - learning rate: {} - train acc: {} - test acc: {}  ({:2f}s)'.format(epoch, acc_loss, _get_learning_rate(optimizer)[0], train_acc, eval_acc, time.time() - st))
    return acc_loss, train_acc, eval_acc

def train(params):
    ### LOAD DATA
    if os.path.exists(params.vocab_path) and os.path.exists(params.data_path):
        logging.info("Loading data")
        X, Y, class_index = pickle.load(open(params.data_path, 'rb'))
        vocab_inv = pickle.load(open(params.vocab_path, 'rb'))
        vocab = {x: i for i, x in enumerate(vocab_inv)}
    else:
        X, Y, vocab, vocab_inv, class_index = load_data_helper.load_data(params, train_new_w2v=True, save_w2v_path=params.pretrain_w2v_model)
        logging.info("Saving data ...")
        pickle.dump((X, Y, class_index), open(params.data_path, 'wb'))
        pickle.dump(vocab_inv, open(params.vocab_path, 'wb'))
    
    params.sentence_len = len(X[0])
    logging.info("Sentence lengh: {}".format(params.sentence_len))
    if os.path.exists(params.embedding_path):
        logging.info("Loading embedding ...")
        params.embedding = pickle.load(open(params.embedding_path, 'rb'))
    else:
        logging.info("Loading pretrain embedding ...")
        params.embedding = load_data_helper.load_pretrain_embedding(vocab_inv, w2v_model_path=params.pretrain_w2v_model, name="gensim")
        pickle.dump(params.embedding, open(params.embedding_path, 'wb'))
    ### CONFIG
    params.vocab_size = len(vocab)
    params.embed_dim = len(params.embedding[0])
    params.class_index = class_index

    num_pooling_feature = 1
    # net = MultiChannelCNN(vocab_size, embed_dim, params.kernel_sizes, params.num_kernels, len(cates), 
    #                     num_pooling_feature, params.hidden_size, embedding, static=False, dropout=0.5, 
    #                     use_batchnorm=True, threshold=0.5, padding_value=0, max_length=300)
    
    print("Using CUDA: {}".format(torch.cuda.is_available()))

    start_at = time.time()
    logging.info("Training...")
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for cv, (train_index, test_index) in enumerate(kf.split(X)):
        train_loss, train_acc, test_acc = train_one_split(params, cv, X, Y, train_index, test_index)
        print('cv = {}    train size = {}    test size = {}\n'.format(cv, len(train_index), len(test_index)))
        break
    end_at = time.time()
    logging.info("start at: {}\nend_at: {}\nruntime: {} min".format(time.ctime(start_at), time.ctime(end_at),
                                                             (end_at - start_at) / 60))
    logging.info('Finished Training\n')

def main():
    str2bool = lambda x: True if x.lower() == 'true' else False
    
    parser = argparse.ArgumentParser("Train Neural Network")
    # parser.add_argument("--use_pretrain", type=str2bool, default=False, help="Load pretrain model for continue training")
    parser.add_argument("--pretrain_w2v_model", type=str, default="./Data/baomoi.window2.vn.model.bin")
    parser.add_argument("--items_path", type=str, default="data/test_items.pickle")
    parser.add_argument("--save_folder", type=str,  default='build/test', help="model to save generated files")
    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    params = Namespace(
        kernel_sizes=[2,4,8],
        num_kernels=128,
        min_offset=10,
        hidden_size=512,
        n_epochs=50,
        batch_size=32,
        learning_rate=0.0005,
        momentum=0.9,
        log_interval=1,
        display_step=1,
        weight_decay=0,
        use_dropout=True,
        min_doc_length=5,
        fixed_length=200,
        save_path=os.path.join(args.save_folder, "model.pickle"),
        data_path=os.path.join(args.save_folder, "data.pickle"),
        vocab_path=os.path.join(args.save_folder, "vocab.pickle"),
        embedding_path = os.path.join(args.save_folder, "embedding.pickle"),
        items_path=args.items_path,
        pretrain_w2v_model=args.pretrain_w2v_model,
        max_vocab_size=50000,
        model_type="gatedcnn"
    )
    train(params)
if __name__ == "__main__":
    main()