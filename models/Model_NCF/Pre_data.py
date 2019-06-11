import pandas as pd
import numpy as np
import argparse
import os
import pickle
import json 
from cat_sql import * 

def prepare_data(input_dir, input_fname, out_dir):
    df= pd.read_csv(os.path.join(input_dir,input_fname))
    columns=['d_path']
    df=df.drop(columns,axis=1)
    df=df.drop_duplicates(['itemId','guid'],keep='last').reset_index(drop=True)
    guid = {}
    for key in df['guid']:
        guid[key] = []
    for i, value in enumerate(df['guid']):
        guid[df['guid'][i]].append(df['itemId'][i])

    items = list(guid.values())
    users = list(guid.keys())
    datas = []
    items = list(guid.values())
    for i in range(len(items)):
        for j in range(len(items[i])):
            if len(items[i]) > 7 and len(str(items[i][j])) > 6:
                datas.append(users[i])
    
    df=df.loc[df['guid'].isin(datas)]  
    df['itemId']=df['itemId'].astype(str)
    itemids=list(set(df.itemId))
    cats,item_ids=get_cate(itemids)
    newids=list(map(str, item_ids))
    content,news_ids=get_content(newids)
    newids=list(map(str, news_ids)) 
    df=df.loc[df['itemId'].isin(newids)]
    items = set(df.itemId) 
    item2int = {}
    int2item = {}
    for i,item in enumerate(items):
        item2int[item] = i
        int2item[i] = item

    users = set(df.guid) 
    user2int = {}
    int2user = {}
    for i,user in enumerate(users):
        user2int[user] = i
        int2user[i] = user 

    user_index=json.dumps(user2int)
    f=open(os.path.join(out_dir,'user2int.txt'),'w')
    f.write(user_index)
    f.close() 
    
    item_index=json.dumps(item2int)
    f=open(os.path.join(out_dir,'item2int.txt'),'w')
    f.write(item_index)
    f.close()

    df['item_index']=df['itemId'].apply(lambda x: item2int[x])
    df['user_index']=df['guid'].apply(lambda x: user2int[x])
    df=df.sort_values(by=['user_index']) 
    df.to_csv(os.path.join(out_dir,'all_data.csv'), index=False)
    test_data=df.groupby('user_index').tail(1)
    train = pd.concat([df, test_data])
    train = train.drop_duplicates(keep=False, inplace=False)
    all_items = test_data.item_index.unique()
    negative = (test_data.groupby("user_index")['item_index']
                .apply(list)
                .reset_index())
    np.random.seed=1981
    negative['negative'] = negative.item_index.apply(lambda x: np.random.choice(np.setdiff1d(all_items,x), 99))
    negative.drop('item_index', axis=1, inplace=True)
    negative=test_data.merge(negative, on='user_index')
    negative['positive'] = negative[['user_index', 'item_index']].apply(tuple, axis=1)
    negative.drop(['guid','itemId','user_index','item_index'], axis=1, inplace=True)
    negative = negative[['positive','negative']]
    negative[['item_n'+str(i) for i in range(99)]] = pd.DataFrame(negative.negative.values.tolist(), index= negative.index)
    negative.drop('negative', axis=1, inplace=True)
    
    train.to_csv(os.path.join(out_dir,'soha.train'),  index=False)
    test_data.to_csv(os.path.join(out_dir,'soha.test'),  index=False)
    negative.to_csv(os.path.join(out_dir,'soha.test.negative'), index=False)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_dir",type=str, default="./Data/")
    parser.add_argument("--input_fname",type=str, default="soha_data_raw.csv")
    parser.add_argument("--out_dir",type=str, default="./Data/")
    args = parser.parse_args()
    
    prepare_data(
		args.input_dir,
		args.input_fname,
        args.out_dir)
   



