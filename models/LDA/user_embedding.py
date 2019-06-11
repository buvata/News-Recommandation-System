import pandas as pd 
import numpy as np 
data=pd.read_csv('./Data/soha.train.rating')

def createEmptyList(length,fill=None):
    lists=[fill] * length
    return lists

guid={}
for key in data['user_index']:
    guid[key]=[]
for i,value in enumerate(data['user_index']):
    guid[data['user_index'][i]].append(data['item_index'][i])
user_index=list(guid.keys())
item_index=list(guid.values())
item_embbeding=np.loadtxt("./Data/item_emb.dat")

def user_embedding():
    user_embedding=createEmptyList(len(user_index),0)
    for i in user_index:
        for j in item_index[i]:
            user_embedding[i]+=item_embbeding[j]
        user_embedding[i]=user_embedding[i]/len(item_index[i])
    return user_embedding 

user_emb=user_embedding()
np.savetxt("./Data/user_emb.dat",user_emb)

