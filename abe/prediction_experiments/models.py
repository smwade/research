import numpy as np
import pandas as pd
#import gensim


from passage.preprocessing import Tokenizer
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.models import RNN
from passage.utils import load, save

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import LogisticRegression as lrc
from scipy import sparse as sp
from collections import Counter

#import sklearn_crfsuite

# model functions. These train and score models using given parameters, training data, and test data
def recurrent_nerual_net_model(params, tokens, truth, test_tokens, test_truth, return_prediction=False):
    layers = [
        Embedding(size=params["embed_size"], n_features=params["n_features"]),
        GatedRecurrent(size=params["recurrent_size"]),
        Dense(size=1, activation='sigmoid')
    ]
    model = RNN(layers=layers, cost='BinaryCrossEntropy')
    model.fit(tokens, truth,n_epochs=params["n_epochs"])
    pred = model.predict(test_tokens)
    score = roc_auc_score(test_truth, pred)
    if return_prediction:
        return score, model, pred
    else:
        return score, model

def random_forest_model(params, tokens, truth, test_tokens, test_truth):
    model = rfc(n_estimators=params["n_trees"], n_jobs=6, class_weight='auto')
    model.fit(tokens,truth)
    pred= model.predict_proba(test_tokens)[:,1]
    score = roc_auc_score(test_truth,pred)
    return score, model

def logistic_regression_model(params, tokens, truth, test_tokens, test_truth):
    """
    Example
    -------
    params = {'C':0.5}
    sparse_train_data = create_sparse_array_from_tokens(tokens, len(dictionary)+1)
    sparse_valid_data = create_sparse_array_from_tokens(valid_tokens, len(dictionary)+1)
    logistic_regression_model(params, sparse_train_data,list(data.loc[data.Train==1,"Diseased"].values),
                        sparse_valid_data, list(data.loc[data.Train==2,"Diseased"].values))
    """
    model = lrc(C = params["C"], n_jobs=6, class_weight='auto')
    model.fit(tokens,truth)
    pred = model.predict_proba(test_tokens)[:,1]
    score = roc_auc_score(test_truth,pred)
    return score, model

def conditional_random_field_model(params, tokens, truth, test_tokens, test_truth):
    """
    Example
    -------
    params = {'w':2}
    tokens = create_tokens(data.loc[data["Train"]==1, "ObsCodes"].values, params['w'])
    truth = [seq2labels(data.loc[ind,"ObsCodes"].split(","), data.loc[ind,"Diseased"]) 
             for ind in data.loc[data["Train"]==1].index]
    test_tokens = create_crf_tokens(data.loc[data["Train"]==2, "ObsCodes"].values, params['w'])
    test_truth = data.loc[data.Train==2,"Diseased"].values
    conditional_random_field_model(params, tokens, truth, test_tokens, test_truth)
    """
    model = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1, # vary?
        c2=0.1, # vary?
        max_iterations=100, # vary?
        all_possible_transitions=True
    )
    model.fit(tokens, truth)

    # test the model, display metrics
    res = model.predict_marginals(test_tokens)
    pred = [r[-1]['1'] for r in res]
    score = roc_auc_score(test_truth, pred)
    return score, model

# sparse feature extractor functions for RF and LR
def create_ccs_map(path_to_ccs):
    """
    Create a mapping from ICD-9 code to CCS group for dimensionality reduction.
    
    Parameters
    ----------
    path_to_ccs : string
        Gives the path to the ccs text file
    
    Returns
    -------
    ccs_map : dict
    """
    groups = []
    names = []
    with open(path_to_ccs, "r") as f:
        for i in xrange(4): # skip the first 4 lines
            f.readline()
        in_name_line=True
        for l in f:
            if in_name_line:
                names.append(" ".join(l.split()[1:]))
                group = []
                in_name_line=False
            else:
                if l == "\n":
                    groups.append(group)
                    in_name_line=True
                else: 
                    group.extend(l.split())
        if not in_name_line:
            groups.append(group)
    ccs_map = {c:i for i in xrange(len(groups)) for c in groups[i]}
    return ccs_map

def create_dictionary(sequences):
    """
    Create a dictionary from a list of sequences that maps symbols to nonnegative integers.

    Parameters
    ----------
    sequences : list of strings
        Each string is a |- and comma-separated list of symbols.

    Returns
    -------
    dictionary : dict
        Maps the symbols in the sequences to a contiguous list of nonnegative integers starting at 0.
    """
    symbols = set()
    for seq in sequences: # add all words to the dictionary
        for s in seq.split("|"):
            symbols.update(s.split(","))
    if "p_" in symbols:
        symbols.remove("p_")
    if "" in symbols:
        symbols.remove("")
    dictionary = {s:i for i, s in enumerate(symbols)}
    return dictionary

def create_tokens(sequences, dictionary):
    """
    Map each sequence (string with comma-separated symbols) in the list to 
    a list of nonnegative numbers, and prepend a fixed unique integer to each.

    Parameters
    ----------
    sequences : list of strings
        Each string is a |- and comma-separated list of symbols
    dictionary : dict
        Keys are symbols found in sequences, values are nonnegative ints

    Returns
    -------
    tokens : list of lists, corresponding to the list of sequences
    
    Example
    -------
    dictionary = {'a':0, 'b':1, 'c':2}
    sequences = ["a,a,b", "b,c,a,c,b"]
    create_tokens(sequences, dictionary) # returns [[3,0,0,1], [3,1,2,0,2,1]]
    """
    p = len(set(dictionary.values()))
    tokens=[]
    for seq in sequences:
        l = [p]
        for s in seq.split("|"):
            for c in s.split(","):
                try:
                    l.append(dictionary[c])
                except KeyError:
                    continue
        tokens.append(l)
    return tokens  

def create_sparse_array_from_tokens(tokens, N_cols):    
    """
    Create sparse matrix BOW representation for use in classifiers.

    Parameters
    ----------
    tokens : list of lists of nonnegative integers
        Should be the output of create_tokens
    N_cols : positive integers
        Gives the number of unique tokens
    
    Returns
    -------
    mat : sparse csc_matrix of shape (len(tokens),N_cols)
    """
    csc_dat = []
    csc_row = []
    csc_col = []
    for i in xrange(len(tokens)):
        for j, d in Counter(tokens[i]).iteritems():
            csc_dat.append(d)
            csc_row.append(i)
            csc_col.append(j)
    return sp.csc_matrix((csc_dat, (csc_row, csc_col)), shape=(len(tokens), N_cols))

# feature extractor functions for conditional_random_field_model
def code2features(seq,ccs_map,i,w):
    """
    Return the feature vector for the element at position i of input sequence seq.
    
    Parameters
    ----------
    seq : list of strings
    i : integer, between 0 and len(seq)-1 inclusive
    w : nonnegative integer, radius of window size
    
    Returns
    -------
    features : dict giving the feature vector
    """
    features = {
        'bias':1.0,
        'code':seq[i],
        'ccs':ccs_map[seq[i]]
    }
    for j in xrange(1,w+1):
        if i-j>=0:
            features.update({
                    '-{}code'.format(j):seq[i-j],
                    '-{}ccs'.format(j):ccs_map[seq[i-j]]
                })
        if i+j < len(seq):
            features.update({
                    '+{}code'.format(j):seq[i+j],
                    '+{}ccs'.format(j):ccs_map[seq[i+j]]
                })
    return features
def seq2features(seq,ccs_map,w):
    start_feat = {'bias':1.0, 'code':-1, 'ccs':-1}
    return [start_feat]+[code2features(seq,ccs_map,i,w) for i in xrange(len(seq))]
def seq2labels(seq,lab):
    return [str(lab)]*(len(seq))
def filter_sequence(seq,ccs_map):
    return [s for s in seq if s in ccs_map.keys()]
def create_crf_tokens(codes,ccs_map,w): 
    return [seq2features(filter_sequence(seq.split(","),ccs_map),ccs_map,w) for seq in codes]


