#!/usr/bin/python3

import neural_network
import pandas as pd

import sys
sys.path.append("..")
import utilities


# Load the vector space model
vsm = pd.read_csv("../data/text/glove_gen_n100_win15_min5_iter500_190428.txt", 
				  sep=" ", index_col=0, header=0)

# Load the title-term matrix
ttm_bin = utilities.load_ttm(path="../")
ttm_bin = ttm_bin[ttm_bin.columns.intersection(vsm.index)]

# Optimize the classifier over a random hyperparameter grid
neural_network.optimize_classifier(ttm_bin, suffix="_occ_titles")
