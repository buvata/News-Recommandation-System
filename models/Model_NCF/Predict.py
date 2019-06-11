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
from NeuCF_torch import NeuMFWrapper 
from Dataset import Dataset 
from time import time
from util import *
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser()
    
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
    return parser.parse_args([])

args = parse_args()
n_emb=args.n_emb
layers = eval(args.layers)
dropouts = eval(args.dropouts)

def load_model(modelpath):
    model = NeuMFWrapper(n_emb, layers, dropouts)
    model.network.load_state_dict(torch.load(modelpath, map_location=lambda storage, loc: storage))
    return model

def predict(model,user_emb,item_embs):
    '''
    input dict_user,dict_item={id1:emb1,id2:emb2}
    return dict{itemid:score}
    '''
    #model=load_model(modelpath)
    model.eval()
    list_item_emb=list(item_embs.values())
    list_itemids=list(item_embs.keys())
    user_emb=list(user_emb.values())
    user_emb=[user_emb[0]]*len(list_itemids)
    user_emb=torch.FloatTensor(user_emb)
    item_embs=torch.FloatTensor(list_item_emb)
    preds=model(None,None,user_emb,item_embs)
    map_item_score = dict(zip(list_itemids, preds.squeeze(1).detach().numpy()))
    rank_dict=OrderedDict(sorted(map_item_score.items(), key=lambda t: t[1],reverse=True))
    
    return dict(rank_dict)



