#!/usr/bin/python3

import argparse
import neural_network
import pandas as pd

import sys
sys.path.append("..")
import utilities


parser = argparse.ArgumentParser(description='Neural network classifier with term occurrences')
parser.add_argument('--data', type=str, default='titles',
					help='text inputs')
args = parser.parse_args()
data = args.data


if data not in ["titles", "texts"]:
	raise ValueError("""An invalid option for `--data` was supplied,
					 options are ['titles', 'texts']""")


# Load the vector space model
vsm = pd.read_csv("../data/text/glove_gen_n100_win15_min5_iter500_190428.txt", 
				  sep=" ", index_col=0, header=0)

# Load the document-term matrix
dtm_bin = utilities.load_doc_term_matrix(path="../", inputs=data)
dtm_bin = dtm_bin[dtm_bin.columns.intersection(vsm.index)]

# Optimize the classifier over a random hyperparameter grid
neural_network.optimize_classifier(dtm_bin, suffix="_occ_{}".format(data))
