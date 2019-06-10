#!/usr/bin/python3

import argparse
import pandas as pd

import sys
sys.path.append("..")
import utilities
import neural_network_lstm


parser = argparse.ArgumentParser(description='Neural network classifier with LSTM encodings')
parser.add_argument('--data', type=str, default='titles',
					help='text inputs')
args = parser.parse_args()
data = args.data


# Load the GloVe vector space model
vsm = pd.read_csv("../data/text/glove_gen_n100_win15_min5_iter500_190428.txt", 
				  sep = " ", index_col=0, header=0)
n_vocab, n_emb = vsm.shape


# Load the lexicon
if data == "titles":
	ttm_bin = utilities.load_ttm(path="../")
	ttm_bin = ttm_bin[ttm_bin.columns.intersection(vsm.index)]
	lexicon = list(ttm_bin.columns)
elif data == "texts":
	dtm_bin = utilities.load_dtm(path="../")
	dtm_bin = dtm_bin[dtm_bin.columns.intersection(vsm.index)]
	lexicon = list(dtm_bin.columns)
else:
	raise ValueError("""An invalid option for `--data` was supplied,
					 options are ['titles', 'texts']""")

# Optimize the classifier over a random hyperparameter grid
suffix = "_con_e2e_" + args.data
neural_network_lstm.optimize_classifier(vsm, lexicon, data, suffix=suffix)

