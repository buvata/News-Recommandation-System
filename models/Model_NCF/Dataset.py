import scipy.sparse as sp
import pandas as pd
import numpy as np

class Dataset(object):
    def __init__(self, path):  
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train")
        self.testRatings = self.load_rating_file_as_list(path + ".test")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        df = pd.read_csv(filename)
        ratingList = list(zip(df.user_index.tolist(), df.item_index.tolist()))
        return ratingList

    def load_negative_file(self, filename):
        df = pd.read_csv(filename)
        negativeList = df.iloc[:, 1:].values.tolist()
        return negativeList

    def load_rating_file_as_matrix(self, filename):    
        df = pd.read_csv(filename)
        num_users = df.user_index.max()
        num_items = df.item_index.max()
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        interactions = df[['user_index','item_index']].values.tolist()
        for user, item in interactions:
                mat[user, item] = 1.
        return mat
