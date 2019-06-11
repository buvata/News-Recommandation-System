import numpy as np
import pandas as pd
import os
import torch
import argparse
import heapq
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader, Dataset
from Dataset import Dataset 
from time import time
from util import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="./Data/",
        help="data directory.")
    parser.add_argument("--modeldir", type=str, default="./models",
        help="models directory")
    parser.add_argument("--dataname", type=str, default="soha",
        help="chose a dataset.")
    parser.add_argument("--epochs", type=int, default=10,
        help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256,
        help="batch size.")
    parser.add_argument("--n_emb", type=int, default=380,
        help="embedding size.")
    parser.add_argument("--lr", type=float, default=0.0001,
        help="learning rate.")
    parser.add_argument("--learner", type=str, default="adam",
        help="Specify an optimizer: adagrad, adam, rmsprop, sgd")
    parser.add_argument("--validate_every", type=int, default=1,
        help="validate every n epochs")
    parser.add_argument("--save_model", action="store_false")
    parser.add_argument("--n_neg", type=int, default=4,
        help="number of negative instances to consider per positive instance.")
    parser.add_argument("--topK", type=int, default=10,
        help="number of items to retrieve for recommendation.")

    return parser.parse_args()

class GMF(nn.Module):
    def __init__(self, n_user, n_item, n_emb):
        super(GMF, self).__init__()

        self.n_emb = n_emb
        self.n_user = n_user
        self.n_item = n_item
        user_emb=np.loadtxt("./Data/user_emb.dat")
        item_emb=np.loadtxt("./Data/item_emb.dat")
        weight_user = torch.cuda.FloatTensor(user_emb)
        weight_item = torch.cuda.FloatTensor(item_emb)
        self.embeddings_user = nn.Embedding.from_pretrained(weight_user)
        self.embeddings_item = nn.Embedding.from_pretrained(weight_item)
        self.embeddings_user.requires_grad=False
        self.embeddings_item.requires_grad=False
        #self.embeddings_user = nn.Embedding(n_user, n_emb)
        #self.embeddings_item = nn.Embedding(n_item, n_emb)
        self.out = nn.Linear(in_features=n_emb, out_features=1)

    def forward(self, users, items,user_embs,item_embs):
        if user_embs is None and item_embs is None :
            user_emb = self.embeddings_user(users)
            item_emb = self.embeddings_item(items)
            prod = user_emb*item_emb
            preds = torch.sigmoid(self.out(prod))

        if users is None and items is None :
            prod=user_embs*item_embs
            preds=torch.sigmoid(self.out(prod))
        
        return preds

def train(model, criterion, optimizer, epoch, batch_size, use_cuda,
    trainRatings,n_items,n_neg,testNegatives):
    model.train()
    train_dataset = get_train_instances(trainRatings,
        n_items,
        n_neg,
        testNegatives,
         mode="pytorch")
    train_loader = DataLoader(dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)
    n_batches = (train_dataset.shape[0]//batch_size)+1
    for batch_idx, data in enumerate(train_loader):
        users = Variable(data[:,0])
        items = Variable(data[:,1])
        labels = Variable(data[:,2]).float()
        if use_cuda:
            users, items, labels = users.cuda(), items.cuda(), labels.cuda()        
        user=torch.cuda.LongTensor(users)
        item=torch.cuda.LongTensor(items)
        optimizer.zero_grad()
        preds =  model(user, item,None,None)
        loss = criterion(preds.squeeze(1), labels)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader, use_cuda, topK):
    model.eval()
    hits, ndcgs = [],[]
    n_batches = test_loader.dataset.shape[0] // test_loader.batch_size
    for batch_idx, data in enumerate(test_loader):
        users = Variable(data[:,0])
        items = Variable(data[:,1])
        labels = Variable(data[:,2]).float()
        if use_cuda:
            users, items, labels = users.cuda(), items.cuda(), labels.cuda()
        user=torch.cuda.LongTensor(users)
        item=torch.cuda.LongTensor(items)
        preds = model(user, item,None,None)
        gtItem = items[0].item()
        map_item_score = dict( zip(items.cpu().numpy(), preds.squeeze(1).detach().cpu().numpy()) )
        ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        ndcg = getNDCG(ranklist, gtItem)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (np.array(hits).mean(),np.array(ndcgs).mean())

def checkpoint(model, modelpath):
    torch.save(model.network.state_dict(), modelpath)

if __name__ == '__main__':
    args = parse_args()
    datadir = args.datadir
    dataname = args.dataname
    modeldir = args.modeldir
    n_emb = args.n_emb
    batch_size = args.batch_size
    epochs = args.epochs
    learner = args.learner
    lr = args.lr
    validate_every = args.validate_every
    save_model = args.save_model
    topK = args.topK
    n_neg = args.n_neg
    modelfname = "pytorch_GMF" + \
        "_".join(["_bs", str(batch_size)]) + \
        "_".join(["_lr", str(lr).replace(".", "")]) + \
        "_".join(["_n_emb", str(n_emb)]) + ".pt"
    modelpath = os.path.join(modeldir, modelfname)
    resultsdfpath = os.path.join(modeldir, 'results_df.p')
    dataset = Dataset(os.path.join(datadir, dataname))
    trainRatings, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    n_users, n_items = trainRatings.shape
    test_dataset = get_test_instances(testRatings, testNegatives)
    test_loader = DataLoader(dataset=test_dataset,
        batch_size=100,
        shuffle=False
        )
    model = GMF(n_users, n_items, n_emb=n_emb)
    if learner.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif learner.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif learner.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
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
        print("The best GMF model is saved to {}".format(modelpath))
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
