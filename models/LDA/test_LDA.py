# -*- coding: utf-8 -*-
#import LDA_VB 
import sys, re, time ,string 
import numpy as np
from scipy.special import gammaln, psi
from gensim.matutils import dirichlet_expectation
import pandas as pd 


def parse_doc_list(docs,vocab):
    """
    Parse a document into a list of word ids and a list of counts,
    

    Arguments: 
    docs:  List of D documents. Each document must be represented as
           a single string. (Word order is unimportant.) Any
           words not in the vocabulary will be ignored.

    vocab: Dictionary mapping from words to integer ids.
    Returns a pair of lists of lists.

    The first, wordids, says what vocabulary tokens are present in
    each document. wordids[i][j] gives the jth unique token present in
    document i. (Don't count on these tokens being in any particular
    order.)
    The second, wordcts, says how many times each vocabulary token is
    present. wordcts[i][j] is the number of times that the token given
    by wordids[i][j] appears in document i.
    """
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)
    
    wordids = list()
    wordcts = list()
    for d in range(0, D):
       # docs[d] = docs[d].lower()
       # docs[d] = re.sub(r'-', ' ', docs[d])
       # docs[d] = re.sub(r'[^a-z ]', '', docs[d])
       # docs[d] = re.sub(r' +', ' ', docs[d])
        words = str.split(str(docs[d]))
        ddict = dict()
        for word in words:
            if (word in vocab):
                wordtoken = vocab[word] 
                if (not wordtoken in ddict):
                    ddict[wordtoken] = 0
                ddict[wordtoken] += 1
        wordids.append(ddict.keys())
        wordcts.append(ddict.values())

    return((wordids, wordcts))

np.random.seed(100000001)
meanchangethresh = 0.001

'''
α:1×k  Model parameter - vector of topic distribution probabilities for each document

β:k×v  (CxL+G)xV Model parameter - matrix of word probabilities for each topic

ϕ:D×Nd×k Variational parameter - matrix of topic probabilities for each word in each document

γ:D×k  Dx(C*L+G) Variational parameter - matrix of topic probabilities for each document
'''

class online_LDA:
    def __init__(self,vocab,C,L,G,alpha,eta,tau0,kappa, _lambda_local=None, _lambda_glob=None, _lambda=None):
        
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. Any word not in this set will be ignored.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
            : Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.
        Note that if you pass the same set of D documents in every time and
        set lr=0 this class can also be used to do batch VB. 

        """

        self._vocab=dict()
        for word in vocab:
            word=word.lower()
            #word=re.sub(r'[^a-z]','',word)
            self._vocab[word]=len(self._vocab)

        self._W = len(self._vocab)
        #self._D=D 
        self._alpha=alpha 
        self._eta= eta 
        self._tau0= tau0 +1 
        self._updatect =0  
        self._C=C 
        self._G=G 
        self._L=L 
        self._kappa = kappa
        # Initialize the variational distribution q(beta|lambda)
        if _lambda_local is not None: 
            self._lambda_local = _lambda_local
        else:
            self._lambda_local = 1*np.random.gamma(100., 1./100., ( self._C*self._L, self._W))
        
        if _lambda_glob is not None:
            self._lambda_glob= _lambda_glob
        else:
            self._lambda_glob= 1*np.random.gamma(100., 1./100., ( self._G, self._W))

        if _lambda is not None:
            self._lambda= _lambda
        else:
            self._lambda=1*np.random.gamma(100.,1./100.,(self._C*self._L+self._G,self._W))

        self._Elogbeta_local = dirichlet_expectation(self._lambda_local)
        self._expElogbeta_local = np.exp(self._Elogbeta_local)

        self._Elogbeta_glob = dirichlet_expectation(self._lambda_glob)
        self._expElogbeta_glob = np.exp(self._Elogbeta_glob)

    def do_e_step(self, docs,catid):
        
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.
        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.
        Returns a tuple containing the estimated values of gamma,
        as well as sufficient statistics needed to update lambda.
        """
        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        # if (type(docs).__name__ == 'string'):
           # temp = list()
           # temp.append(docs)
           # docs = temp
        (wordids, wordcts) = parse_doc_list(docs,self._vocab)
        batchD = len(docs)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch

        gamma_local = 1*np.random.gamma(100., 1./100, (batchD, self._C*self._L))
        gamma_glob= 1*np.random.gamma(100.,1./100, self._G)

        Elogtheta_local = dirichlet_expectation(gamma_local)
        expElogtheta_local = np.exp(Elogtheta_local)

        Elogtheta_glob=dirichlet_expectation(gamma_glob)
        expElogtheta_glob=np.exp(Elogtheta_glob) 

        it = 0
        meanchange = 0 
        sstats_glob = np.zeros(self._lambda_glob.shape)
        sstats_local=np.zeros(self._lambda_local.shape)
        
        for d in range(0, batchD):
            ids = list(wordids[d])
            cts = list(wordcts[d])
            gammad_glob = gamma_glob
            Elogthetad_glob = Elogtheta_glob
            expElogthetad_glob = expElogtheta_glob
            expElogbetad_glob = self._expElogbeta_glob[:, ids]
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm_glob = np.dot(expElogthetad_glob, expElogbetad_glob) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad_glob
               
                gammad_glob = self._alpha + expElogthetad_glob * np.dot(cts / phinorm_glob, expElogbetad_glob.T)
                Elogthetad_glob = dirichlet_expectation(gammad_glob)
                expElogthetad_glob = np.exp(Elogthetad_glob)
                phinorm_glob = np.dot(expElogthetad_glob, expElogbetad_glob) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = np.mean(abs(gammad_glob - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma_glob = gammad_glob

            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats_glob[:, ids] += np.outer(expElogthetad_glob.T, cts/phinorm_glob)

            ######## update for local #############################333            
            c=catid[d]
            R_gammad_local=gamma_local[d,c*self._L:(c+1)*self._L]
            Elogthetad_local= Elogtheta_local[d, c*self._L:(c+1)*self._L]
            expElogthetad_local = expElogtheta_local[d,c*self._L:(c+1)*self._L]
            expElogbetad_local = self._expElogbeta_local[c*self._L:(c+1)*self._L, ids]
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm_local = np.dot(expElogthetad_local, expElogbetad_local) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = R_gammad_local
                # the update for gamma 
                R_gammad_local = self._alpha + expElogthetad_local * np.dot(cts/phinorm_local, expElogbetad_local.T)
                Elogthetad_local = dirichlet_expectation(gamma_local[d,c*self._L:(c+1)*self._L])
                expElogthetad_local = np.exp(Elogthetad_local)
                phinorm_local = np.dot(expElogthetad_local, expElogbetad_local) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = np.mean(abs(R_gammad_local - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma_local[d, c*self._L:(c+1)*self._L] = R_gammad_local

            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats_local[c*self._L:(c+1)*self._L, ids] += np.outer(expElogthetad_local.T, cts/phinorm_local)
                
            # This step finishes computing the sufficient statistics for the
           
        sstats_local = sstats_local * self._expElogbeta_local
        sstats_glob = sstats_glob * self._expElogbeta_glob
        gamma_glob=np.asarray([gamma_glob]*batchD)
        gamma=np.concatenate((gamma_local,gamma_glob),axis=1)
        sstats=np.concatenate((sstats_local,sstats_glob),axis=0)

        return ((gamma,gamma_glob,gamma_local,sstats,sstats_glob,sstats_local))
    
    def update_lambda(self, docs,catid):
        """
        uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.
        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma,gamma_glob,gamma_local,sstats,sstats_glob,sstats_local) = self.do_e_step(docs,catid)
        
        # Estimate held-out likelihood for current values of lambda.
        # bound = self.approx_bound(docs, gamma)
        #bound = 0
        # Update lambda based on documents.
        self._lambda  = self._lambda * (1-rhot) + rhot * (self._eta + self._D * sstats / len(docs))
        self._lambda_glob  = self._lambda_glob * (1-rhot) + rhot * (self._eta + self._D * sstats_glob / len(docs))
        self._Elogbeta_glob = dirichlet_expectation(self._lambda_glob) 
        self._expElogbeta_glob = np.exp(self._Elogbeta_glob)

        self._lambda_local  = self._lambda_local * (1-rhot) + rhot * (self._eta + self._D * sstats_local / len(docs))
        self._Elogbeta_local = dirichlet_expectation(self._lambda_local) 
        self._expElogbeta_local = np.exp(self._Elogbeta_local)
        
        self._updatect += 1

        return (gamma,gamma_glob,gamma_local)

"""
def main():
    data=pd.read_csv('./Data/docs_process.csv')
    # The number of documents to analyze each iteration
    #D = len(data['content'])
    D=len(data)
    batchsize =D
    C = 22
    L=15
    G=50
    vocab = list()
    with open('./Data/vocab.txt','r') as f:
        for line in f:
            vocab.append(line.strip())  
    documentstoanalyze = int(D/batchsize)
    W = len(vocab)
    olda = online_LDA(vocab,C, L,G, D, 1./100, 1./100, 64., 0.6)
    (wordids, wordcts) = parse_doc_list(data.content.values.tolist(), olda._vocab)           
    ((t_gamma,gamma_glob,gamma_local,sstats,sstats_glob,sstats_local))=olda.do_e_step(data.content.values.tolist(),data.cat_index.values.tolist())
    np.savetxt('/data1/taibv/soha_data/item_emb.dat',t_gamma) 
"""         
        

   

   
