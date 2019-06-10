#!/usr/bin/python3

import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import utilities
from classifiers import neural_network, neural_network_cnn
print_fig = False


##########################################
########## Plotting parameters ###########
##########################################

color = "#6bbca7" # Color for plotting evaluation metrics and word clouds
cmap = "Greens" # Color map for plotting brain structures
prefix = "cnn" # Prefix for plot file names
n_top = 15 # Number of terms (i.e., inputs) to plot per brain structure (i.e., class)
fname = "style/computer-modern/cmunss.ttf" # Name of the font
inputs = ["texts", "titles"] # Whether the inputs are from article full texts or titles
split = "test"
batch_size = 64
verbose = True


##########################################
########### Inputs and labels ############
##########################################

print("Loading inputs and labels")

# Vector space model of GloVe embeddings trained on article full texts
vsm = pd.read_csv("../data/text/glove_gen_n100_win15_min5_iter500_190428.txt", sep = " ", index_col=0, header=0)
n_vocab, n_emb = vsm.shape

# Document-term matrix generated from article full texts or titles
dtm, n_terms = {}, {}
for inp in inputs:
	dtm_inp = utilities.load_doc_term_matrix(path="../", inputs=inp)
	dtm[inp] = dtm_inp[dtm_inp.columns.intersection(vsm.index)]

# Splits of the article PMIDs
splits = utilities.load_splits(splits=[split])

# Full embeddings for terms in full texts and titles
def load_emb(split):
    m, n_terms = dtm[inp].shape
    lexicon = list(dtm[inp].columns)
    emb = np.zeros((n_terms, n_emb, len(splits[split])))
    occ = dtm[inp].loc[splits[split]]
    for i, pmid in enumerate(splits[split]):
        terms = occ.columns[occ.values[i,:] == 0]
        emb_i = vsm.loc[lexicon].copy()
        emb_i.loc[terms] = 0
        emb[:,:,i] = emb_i.values
    return emb

X = {}
for inp in inputs:
	X[inp] = {}
	for split in splits.keys():
		X[inp][split] = load_emb(split)

# Output labels are brain activation coordinates
Y = utilities.load_coordinates(path="../")
m, n_structs = Y.shape



##########################################
############ Classifier fits #############
##########################################

print("Loading classifier fits")

import torch
from torch.autograd import Variable
import torch.optim as optim
torch.manual_seed(42)


loss, fit = {}, {}
params = {inp: {} for inp in inputs}
for inp in inputs:
	
	loss[inp] = pd.read_csv("../classifiers/data/loss_cnn_{}_3h.csv".format(inp), index_col=0, header=0)
	
	p = pd.read_csv("../classifiers/data/params_cnn_{}_3h.csv".format(inp), index_col=0, header=None)
	params[inp]["n_hid"] = int(p.loc["n_hid"].values[0])
	params[inp]["p_dropout"] = p.loc["p_dropout"].values[0]
	params[inp]["weight_decay"] = p.loc["weight_decay"].values[0]
	params[inp]["lr"] = p.loc["lr"].values[0]

	fit[inp] = neural_network_cnn.CNN2Net(n_input=100, n_output=n_structs, n_terms=dtm[inp].shape[1],
										  n_hid=params[inp]["n_hid"], p_dropout=params[inp]["p_dropout"])
	optimizer = optim.Adam(fit[inp].parameters(), lr=params[inp]["lr"], weight_decay=params[inp]["weight_decay"])
	net_file = "../classifiers/fits/classifier_cnn_{}_3h.pt".format(inp)
	fit[inp].load_state_dict(torch.load(net_file,  map_location="cpu"))


##########################################
########### Evaluation metrics ###########
##########################################

print("Plotting evaluation metrics")

for inp in inputs:

	utilities.plot_loss("{}_{}".format(prefix, inp), list(loss[inp]["LOSS"].values), 
						xlab="Epoch", ylab="Loss", alpha=0.65, color=color, print_fig=False)

	for split, pmids in splits.items():
		data_set = neural_network_cnn.load_mini_batches(X[inp][split], Y, splits[split], mini_batch_size=len(splits[split]))
		utilities.report_curves(data_set, fit[inp], "{}_{}_{}".format(prefix, inp, split), color=color, print_fig=False)
		report = utilities.report_metrics("results/eval_{}_{}_3h_{}.txt".format(prefix, inp, split), 
										  data_set, fit[inp], Y.columns)

