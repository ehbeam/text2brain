#!/usr/bin/python3

import neural_network
import pandas as pd
import numpy as np
np.random.seed(42)

import sys
sys.path.append("..")
import utilities


# Load the GloVe vector space model
vsm = pd.read_csv("../data/text/glove_gen_n100_win15_min5_iter500_190428.txt", 
				  sep = " ", index_col=0, header=0)
n_vocab, n_emb = vsm.shape

# Load the document-term matrix
dtm_bin = utilities.load_dtm(path="../")
dtm_bin = dtm_bin[dtm_bin.columns.intersection(vsm.index)]
m, n_terms = dtm_bin.shape

# Average embeddings for terms that occur in articles
emb_cen = np.zeros((m, n_emb))
for i in range(m):
    terms = dtm_bin.columns[dtm_bin.values[i,:] == 1]
    emb_cen[i,:] = np.mean(vsm.loc[terms].values, axis=0)
emb_cen = pd.DataFrame(emb_cen, index=dtm_bin.index, columns=range(n_emb)).fillna(0)

# Optimize the classifier over a random hyperparameter grid
neural_network.optimize_classifier(emb_cen, suffix="_emb_texts")


