import pandas as pd
import numpy as np
import models
import itertools
import cPickle
import gensim

# initialize global params
disease_name = "CKD"
path_to_data = "./united/ckd_data_diagcodes.csv"
path_to_ccs = "./ccs_map.txt"
path_to_save_folder = "./ckd_prediction_results/diag_codes/united/"
col = {"Diseased":"Diseased",
        "FollowUp":"FollowUp",
        "Codes":"ObsCodes"}

##################################################
################ DATA PREPARATION ################
##################################################

# load the data from csv
data = pd.read_csv(path_to_data, index_col=0).dropna()

# create train, validation, and test sets -- change this section for different dataset sizes
dis_ids = np.random.permutation(data[(data[col["Diseased"]]==1)&(data[col["FollowUp"]]>30)].index)
ctl_ids = np.random.permutation(data[(data[col["Diseased"]]==0)&(data[col["FollowUp"]]>30)].index)
N_dis = len(dis_ids)
N_dis_train = int(.5*N_dis)
N_dis_valid = int(.25*N_dis)
# next three lines are for if you want number of control cases to be a fixed factor greater than diseased
factor = 10
N_ctl_train = N_dis_train*factor
N_ctl_valid = N_dis_valid*factor
N_ctl_test = N_dis_valid*factor
# next three lines are for if you want to use all available control cases
#N_ctl = len(ctl_ids)
#N_ctl_train = int(.5*N_ctl)
#N_ctl_valid = int(.25*N_ctl)
data["Train"] = -1
data.loc[dis_ids[:N_dis_train],"Train"] = 1 
data.loc[dis_ids[N_dis_train:N_dis_train+N_dis_valid],"Train"] = 2 
data.loc[dis_ids[N_dis_train+N_dis_valid:],"Train"] = 0
data.loc[ctl_ids[:N_ctl_train],"Train"] = 1
data.loc[ctl_ids[N_ctl_train:N_ctl_train+N_ctl_valid],"Train"] = 2
data.loc[ctl_ids[N_ctl_train+N_ctl_valid:N_ctl_train+N_ctl_valid+N_ctl_test],"Train"] = 0

# create raw tokenized data
dictionary   = models.create_dictionary(data.loc[data.Train==1,col["Codes"]].values)
train_tokens = models.create_tokens(data.loc[data.Train==1,col["Codes"]].values, dictionary)
valid_tokens = models.create_tokens(data.loc[data.Train==2,col["Codes"]].values, dictionary)
test_tokens  = models.create_tokens(data.loc[data.Train==0,col["Codes"]].values, dictionary)

# create ccs tokenized data
ccs_map = models.create_ccs_map(path_to_ccs)
unknown_codes = set(dictionary.keys()).difference(set(ccs_map.keys()))
p = len(set(ccs_map.values()))
for c in unknown_codes:
    ccs_map[c] = p
train_ccs_tokens = models.create_tokens(data.loc[data.Train==1,col["Codes"]].values, ccs_map)
valid_ccs_tokens = models.create_tokens(data.loc[data.Train==2,col["Codes"]].values, ccs_map)
test_ccs_tokens  = models.create_tokens(data.loc[data.Train==0,col["Codes"]].values, ccs_map)

# create sparse BOW from raw tokens
sparse_train_data = models.create_sparse_array_from_tokens(train_tokens,len(set(dictionary.values()))+1)
sparse_valid_data = models.create_sparse_array_from_tokens(valid_tokens,len(set(dictionary.values()))+1)
sparse_test_data  = models.create_sparse_array_from_tokens(test_tokens,len(set(dictionary.values()))+1)

# create sparse BOW from ccs tokens  
sparse_ccs_train_data = models.create_sparse_array_from_tokens(train_ccs_tokens,len(set(ccs_map.values()))+1)
sparse_ccs_valid_data = models.create_sparse_array_from_tokens(valid_ccs_tokens,len(set(ccs_map.values()))+1)
sparse_ccs_test_data  = models.create_sparse_array_from_tokens(test_ccs_tokens,len(set(ccs_map.values()))+1)


# save stuff for future use
np.savetxt("{}true_labels.txt".format(path_to_save_folder), data.loc[data.Train==0,col["Diseased"]].values)
np.savetxt("{}dis_ids.txt".format(path_to_save_folder), dis_ids)
np.savetxt("{}ctl_ids.txt".format(path_to_save_folder), ctl_ids)

#################################################
################## EXPERIMENTS ##################
#################################################

results = {}


# RNN EXPERIMENT ON RAW TOKENIZED DATA
print "RNN raw experiments"
emb_size = [64, 128, 256]
rec_size = [64, 128, 256]
epochs = [1,2]
p = len(set(dictionary.values()))
best_rnn_score = 0
best_rnn_model = None
best_rnn_params = None
for el in itertools.product(*[emb_size, rec_size, epochs]):
    params = {"embed_size":el[0], "recurrent_size":el[1], "n_epochs":el[2], "n_features":p+1}
    score, mod = models.recurrent_nerual_net_model(params, 
                                                   train_tokens, 
                                                   list(data.loc[data.Train==1,col["Diseased"]].values),
                                                   valid_tokens, 
                                                   list(data.loc[data.Train==2,col["Diseased"]].values))
    if score > best_rnn_score:
        best_rnn_score=score
        best_rnn_model=mod
        best_rnn_params=params
pred_rnn = best_rnn_model.predict(test_tokens)
results["rnn_raw"] = [pred_rnn, best_rnn_params] 


# RNN EXPERIMENT ON CCS TOKENIZED DATA
print "RNN CCS experiments"
emb_size = [64, 128, 256]
rec_size = [64, 128, 256]
epochs = [1,2]
p = len(set(ccs_map.values()))
best_rnn_score = 0
best_rnn_model = None
best_rnn_params = None
for el in itertools.product(*[emb_size, rec_size, epochs]):
    params = {"embed_size":el[0], "recurrent_size":el[1], "n_epochs":el[2], "n_features":p+1}
    score, mod = models.recurrent_nerual_net_model(params, 
                                                   train_ccs_tokens, 
                                                   list(data.loc[data.Train==1,col["Diseased"]].values),
                                                   valid_ccs_tokens, 
                                                   list(data.loc[data.Train==2,col["Diseased"]].values))
    if score > best_rnn_score:
        best_rnn_score=score
        best_rnn_model=mod
        best_rnn_params=params
pred_rnn = best_rnn_model.predict(test_ccs_tokens)
results["rnn_ccs"] = [pred_rnn, best_rnn_params]


# RF EXPERIMENT ON RAW TOKENIZED DATA
print "RF raw experiments"
best_rfc_score = 0
best_rfc_model = None
best_rfc_params = None
for n in [50, 100, 250, 500]:
    params={"n_trees":n}
    score, mod = models.random_forest_model(params, 
                                            sparse_train_data,
                                            list(data.loc[data.Train==1,col["Diseased"]].values),
                                            sparse_valid_data, 
                                            list(data.loc[data.Train==2,col["Diseased"]].values))
    if score > best_rfc_score:
        best_rfc_score=score
        best_rfc_model=mod
        best_rfc_params=params
pred_rfc = best_rfc_model.predict_proba(sparse_test_data)[:,1]
results["rfc_raw"] = [pred_rfc, best_rfc_params]

# RF EXPERIMENT ON CCS TOKENIZED DATA
print "RF CCS experiments"
best_rfc_score = 0
best_rfc_model = None
best_rfc_params = None
for n in [50, 100, 250, 500]:
    params={"n_trees":n}
    score, mod = models.random_forest_model(params, 
                                            sparse_ccs_train_data,
                                            list(data.loc[data.Train==1,col["Diseased"]].values),
                                            sparse_ccs_valid_data, 
                                            list(data.loc[data.Train==2,col["Diseased"]].values))
    if score > best_rfc_score:
        best_rfc_score=score
        best_rfc_model=mod
        best_rfc_params=params
pred_rfc = best_rfc_model.predict_proba(sparse_ccs_test_data)[:,1]
results["rfc_ccs"] = [pred_rfc, best_rfc_params]

# LR EXPERIMENT ON RAW TOKENIZED DATA
print "LR raw experiments"
best_lr_score = 0
best_lr_model = None
best_lr_params = None
for c in [.1, .5, 1.0, 5.0]:
    params={"C":c}
    score, mod = models.logistic_regression_model(params, 
                                                  sparse_train_data,
                                                  list(data.loc[data.Train==1,col["Diseased"]].values),
                                                  sparse_valid_data, 
                                                  list(data.loc[data.Train==2,col["Diseased"]].values))
    if score > best_lr_score:
        best_lr_score=score
        best_lr_model=mod
        best_lr_params=params
pred_lr = best_lr_model.predict_proba(sparse_test_data)[:,1]
results["lr_raw"] = [pred_lr, best_lr_params]

# LR EXPERIMENT ON CCS TOKENIZED DATA
print "LR CCS experiments"
best_lr_score = 0
best_lr_model = None
best_lr_params = None
for c in [.1, .5, 1.0, 5.0]:
    params={"C":c}
    score, mod = models.logistic_regression_model(params, 
                                                  sparse_ccs_train_data,
                                                  list(data.loc[data.Train==1,col["Diseased"]].values),
                                                  sparse_ccs_valid_data, 
                                                  list(data.loc[data.Train==2,col["Diseased"]].values))
    if score > best_lr_score:
        best_lr_score=score
        best_lr_model=mod
        best_lr_params=params
pred_lr = best_lr_model.predict_proba(sparse_ccs_test_data)[:,1]
results["lr_ccs"] = [pred_lr, best_lr_params]

# save results
with open('{}results.dat'.format(path_to_save_folder), 'wb') as outfile:
    cPickle.dump(results, outfile, protocol=cPickle.HIGHEST_PROTOCOL)

# CRF EXPERIMENTS
print "CRF experiments"
best_crf_score = 0
best_crf_model = None
best_crf_params = None
y_valid_crf = data.loc[data.Train==2,col["Diseased"]].values
for w in [0,1,3]:
    params = {'w':w}
    x_train = models.create_crf_tokens(data.loc[data.Train==1,col["Codes"]].values,
                                       ccs_map,
                                       w)
    x_valid = models.create_crf_tokens(data.loc[data.Train==2,col["Codes"]].values,
                                       ccs_map,
                                       w)
    y_train_crf = [[str(lab)]*len(seq) for seq,lab 
                   in itertools.izip(x_train,data.loc[data["Train"]==1,col["Diseased"]].values)]
    score, mod = models.conditional_random_field_model(params,
                                                       x_train,
                                                       y_train_crf,
                                                       x_valid,
                                                       y_valid_crf)
    if score > best_crf_score:
        best_crf_score = score
        best_crf_model = mod
        best_crf_params = params  
x_test = models.create_crf_tokens(data.loc[data.Train==0,col["Codes"]].values,
                                  ccs_map,
                                  best_crf_params['w'])
marg = best_crf_model.predict_marginals(x_test)
pred_crf = [r[-1]['1'] for r in marg]
results["crf"] = [pred_crf, best_crf_params]

# do random forest and LR model selection on LDA dimensionality reduction
print "LDA experiments"
best_lr_lda_model = None
best_lr_lda_score = 0
best_lr_gensim_model = None
best_lr_lda_params = None

best_rfc_lda_model = None
best_rfc_lda_score = 0
best_rfc_gensim_model = None
best_rfc_lda_params = None

gensim_d = gensim.corpora.dictionary.Dictionary([c.split(",") for c in data.loc[data.Train==1,col["Codes"]]])
gensim_train_corp = [gensim_d.doc2bow(c.split(",")) for c in data.loc[data.Train==1, col["Codes"]]]
gensim_valid_corp = [gensim_d.doc2bow(c.split(",")) for c in data.loc[data.Train==2, col["Codes"]]]
gensim_test_corp = [gensim_d.doc2bow(c.split(",")) for c in data.loc[data.Train==0, col["Codes"]]]
for n_tops in [10,30,50,100]:
    gensim_mod = gensim.models.ldamodel.LdaModel(corpus=gensim_train_corp,
                                          id2word=gensim_d,
                                          num_topics=n_tops, 
                                          #workers=1,
                                          chunksize=10,
                                          eval_every=0)

    # create low-dimensional vectors
    lda_train_vectors = [[t[1] for t in gensim_mod.__getitem__(c, eps=0)] 
                                   for c in gensim_train_corp]
    lda_valid_vectors = [[t[1] for t in gensim_mod.__getitem__(c, eps=0)] 
                                   for c in gensim_valid_corp]

    for n in [50, 100, 250, 500]:
        params={"n_trees":n, "n_topics":n_tops}
        score, mod = models.random_forest_model(params, 
                                                lda_train_vectors,
                                                list(data.loc[data.Train==1,"Diseased"].values),
                                                lda_valid_vectors, 
                                                list(data.loc[data.Train==2,"Diseased"].values))
        if score > best_rfc_lda_score:
            best_rfc_lda_score=score
            best_rfc_lda_model=mod
            best_rfc_lda_params=params
            best_rfc_gensim_model = gensim_mod
    
    for c in [.1, .5, 1.0, 5.0]:
        params={"C":c, "n_topics":n_tops}
        score, mod = models.logistic_regression_model(params, 
                                                      lda_train_vectors,
                                                      list(data.loc[data.Train==1,"Diseased"].values),
                                                      lda_valid_vectors, 
                                                      list(data.loc[data.Train==2,"Diseased"].values))
        if score > best_lr_lda_score:
            best_lr_lda_score=score
            best_lr_lda_model=mod
            best_lr_lda_params=params
            best_lr_gensim_model = gensim_mod
            
# predict using the best models
lda_test_vectors = [[t[1] for t in best_rfc_gensim_model.__getitem__(c, eps=0)] 
                    for c in gensim_test_corp]
pred_rfc_lda = best_rfc_lda_model.predict_proba(lda_test_vectors)[:,1]
results["rfc_lda"] = [pred_rfc_lda, best_rfc_lda_params]

lda_test_vectors = [[t[1] for t in best_lr_gensim_model.__getitem__(c, eps=0)] 
                    for c in gensim_test_corp]
pred_lr_lda = best_lr_lda_model.predict_proba(lda_test_vectors)[:,1]
results["lr_lda"] = [pred_lr_lda, best_lr_lda_params]

# save results
with open('{}results.dat'.format(path_to_save_folder), 'wb') as outfile:
    cPickle.dump(results, outfile, protocol=cPickle.HIGHEST_PROTOCOL)

