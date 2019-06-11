import numpy as np
import pandas as pd
import math
import os
import argparse
import heapq
import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader, Dataset
from GMF_torch import GMF, train, evaluate, checkpoint
from MLP_torch import MLP
from Dataset import Dataset 
from time import time
from util import *
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="./Data/",
        help="data directory.")
    parser.add_argument("--modeldir", type=str, default="./models/",
        help="models directory")
    parser.add_argument("--dataname", type=str, default="soha",
        help="chose a dataset.")
    parser.add_argument("--epochs", type=int, default=20,
        help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256,
        help="batch size.")
    parser.add_argument("--lr", type=float, default=0.001,
        help="learning rate.")
    parser.add_argument("--learner", type=str, default="adam",
        help="Specify an optimizer: adagrad, adam, sgd")
    parser.add_argument("--n_emb", type=int, default=380,
        help="embedding size for the GMF part.")
    parser.add_argument("--layers", type=str, default="[760,32,16,8]", help="layer architecture")
    parser.add_argument("--dropouts", type=str, default="[0.,0.,0.]",
        help="dropout per dense layer. len(dropouts) = len(layers)-1")
    parser.add_argument("--validate_every", type=int, default=1,
        help="validate every n epochs")
    parser.add_argument("--save_model", type=int , default=1)
    parser.add_argument("--n_neg", type=int, default=4,
        help="number of negative instances ")
    parser.add_argument("--topK", type=int, default=10,
        help="number of items for recommendation.")
    return parser.parse_args()

class NeuMFWrapper(nn.Module):
    def __init__(self, n_emb, layers, dropouts, user_emb=None, item_emb=None):
        super(NeuMFWrapper, self).__init__()

        self.network = NeuMF(n_emb, layers, dropouts)

        if user_emb is not None and item_emb is not None:     
            weight_user = torch.FloatTensor(user_emb)
            weight_item = torch.FloatTensor(item_emb)    
            self.embeddings_user = nn.Embedding.from_pretrained(weight_user)
            self.embeddings_item = nn.Embedding.from_pretrained(weight_item)
            self.embeddings_user.requires_grad=False
            self.embeddings_item.requires_grad=False

    def forward(self, users, items, user_embs=None, item_embs=None):
        if user_embs is not None and item_embs is not None:
            mf_emb_vector=user_embs*item_embs
            mlp_emb_vector=torch.cat([user_embs,item_embs],dim=1)
            preds=self.network(mf_emb_vector, mlp_emb_vector)
        else:
            mf_user_emb = self.embeddings_user(users)
            mf_item_emb = self.embeddings_item(items)
            mlp_user_emb = self.embeddings_user(users)
            mlp_item_emb = self.embeddings_item(items)
            mf_emb_vector = mf_user_emb*mf_item_emb
            mlp_emb_vector = torch.cat([mlp_user_emb,mlp_item_emb], dim=1)
            preds = self.network(mf_emb_vector, mlp_emb_vector)
       
        return preds

class NeuMF(nn.Module):

    def __init__(self, n_emb, layers, dropouts):
        super(NeuMF, self).__init__()

        self.layers = layers
        self.n_emb=n_emb
        self.n_layers = len(layers)
        self.dropouts = dropouts   
        self.mlp = nn.Sequential()
        for i in range(1,self.n_layers):
            self.mlp.add_module("linear%d" %i, nn.Linear(layers[i-1],layers[i]))
            self.mlp.add_module("relu%d" %i, torch.nn.ReLU())
            self.mlp.add_module("dropout%d" %i , torch.nn.Dropout(p=dropouts[i-1]))
        self.out = nn.Linear(in_features=(n_emb+layers[-1]), out_features=1)

    def forward(self, mf_emb_vector, mlp_emb_vector):
        mlp_emb_vector=self.mlp(mlp_emb_vector)
        emb_vector=torch.cat([mf_emb_vector,mlp_emb_vector],dim=1)
        preds=torch.sigmoid(self.out(emb_vector))
        return preds
    
if __name__ == '__main__':
    args = parse_args()
    datadir = args.datadir
    dataname = args.dataname
    modeldir = args.modeldir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    learner = args.learner
    n_emb = args.n_emb
    user_emb=np.loadtxt("./Data/user_emb.dat")
    item_emb=np.loadtxt("./Data/item_emb.dat")
    layers = eval(args.layers)
    dropouts = eval(args.dropouts)
    validate_every = args.validate_every
    save_model = bool(args.save_model)
    n_neg = args.n_neg
    topK = args.topK
    modelfname = "NeuMF" + \
        "_".join(["_bs", str(batch_size)]) + \
        "_".join(["_lr", str(lr).replace(".", "")]) + \
        "_".join(["_layers", str(layers)]) + \
        "_".join(["_learn", str(learner)]) + \
        ".pt"
    modelpath = os.path.join(modeldir, modelfname)
    resultsdfpath = os.path.join(modeldir, 'results.p')
    dataset = Dataset(os.path.join(datadir, dataname))
    trainRatings, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    n_users, n_items = trainRatings.shape
    test_dataset = get_test_instances(testRatings, testNegatives)
    test_loader = DataLoader(dataset=test_dataset,
        batch_size=100,
        shuffle=False)

    model = NeuMFWrapper( n_emb, layers, dropouts,user_emb,item_emb)
   
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    if learner.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif learner.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_hr, best_ndcgm, best_iter=0,0,0
    for epoch in range(1,epochs+1):
        t1 = time()
        train(model, criterion, optimizer, epoch, batch_size, use_cuda,
            trainRatings,n_items,n_neg,testNegatives)
        t2 = time()
        if epoch % validate_every == 0:
            (hr, ndcg) = evaluate(model, test_loader, use_cuda, topK)
            print("Epoch: {} {:.2f}s, HR = {:.4f}, NDCG = {:.4f}, validated in {:.2f}s".
                format(epoch, t2-t1, hr, ndcg, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter, train_time = hr, ndcg, epoch, t2-t1
                if save_model:
                    checkpoint(model, modelpath)
    print("End. Best Iteration {}:  HR = {:.4f}, NDCG = {:.4f}. ".format(best_iter, best_hr, best_ndcg))
    if save_model:
        print("The best NeuMF model is {}".format(modelpath))
    if save_model:
        if not os.path.isfile(resultsdfpath):
            results_df = pd.DataFrame(columns = ["modelname", "best_hr", "best_ndcg", "best_iter",
                "train_time"])
            experiment_df = pd.DataFrame([[modelfname, best_hr, best_ndcg, best_iter, train_time]],
                columns = ["modelname", "best_hr", "best_ndcg", "best_iter","train_time"])
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_pickle(resultsdfpath)
        else:
            results_df = pd.read_pickle(resultsdfpath)
            experiment_df = pd.DataFrame([[modelfname, best_hr, best_ndcg, best_iter, train_time]],
                columns = ["modelname", "best_hr", "best_ndcg", "best_iter","train_time"])
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_pickle(resultsdfpath)

    
