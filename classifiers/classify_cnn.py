#!/usr/bin/python3

import argparse
import pandas as pd
import numpy as np
np.random.seed(42)

import sys
sys.path.append("..")
import utilities
import neural_network_cnn


parser = argparse.ArgumentParser(description='Neural network classifier with convolution of GloVe embeddings')
parser.add_argument('--data', type=str, default='titles',
					help='text inputs')
args = parser.parse_args()
data = args.data


# Load the GloVe vector space model
vsm = pd.read_csv("../data/text/glove_gen_n100_win15_min5_iter500_190428.txt", 
				  sep = " ", index_col=0, header=0)
n_vocab, n_emb = vsm.shape

# Load the term matrix
if data == "titles":
	X = utilities.load_ttm(path="../")
elif data == "texts":
	X = utilities.load_dtm(path="../")
else:
	raise ValueError("""An invalid option for `--data` was supplied,
					 options are ['titles', 'texts']""")
X = X[X.columns.intersection(vsm.index)]
m, n_terms = X.shape
lexicon = list(X.columns)
vsm = vsm.loc[lexicon]

# Load the data splits
splits = utilities.load_splits(splits=["train", "dev"], path="../", limit=5000)

# Zero out embeddings for terms that did not occur in articles
def load_emb(split):
	emb = np.zeros((n_terms, n_emb, len(splits[split])))
	occ = X.loc[splits[split]]
	for i, pmid in enumerate(splits[split]):
		terms = occ.columns[occ.values[i,:] == 0]
		emb_i = vsm.copy()
		emb_i.loc[terms] = 0
		emb[:,:,i] = emb_i.values
	return emb

emb_train = load_emb("train")
emb_dev = load_emb("dev")

# Optimize the classifier over a random hyperparameter grid
suffix = "_cnn_" + data
neural_network_cnn.optimize_classifier(emb_train, emb_dev, suffix=suffix)
