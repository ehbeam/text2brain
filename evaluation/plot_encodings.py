#!/usr/bin/python3

import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import utilities
from classifiers import neural_network
print_fig = False


##########################################
########## Plotting parameters ###########
##########################################

color = "#7f5b93" # Color for plotting evaluation metrics and word clouds
cmap = "Purples" # Color map for plotting brain structures
prefix = "lstm" # Prefix for plot file names
n_top = 15 # Number of terms (i.e., inputs) to plot per brain structure (i.e., class)
inputs = ["texts", "titles"] # Whether the inputs are from article full texts or titles
batch_size = 64
verbose = True


##########################################
########### Inputs and labels ############
##########################################

print("Loading inputs and labels")

# Encodings from forward propagation through LSTM
X = {}
for inp in inputs:
	X_inp = pd.DataFrame()
	for split in ["train", "dev", "test"]:
		X_inp = X_inp.append(pd.read_csv("../lstm/encodings/lstm_{}_h2_100d_{}.csv".format(inp, split), 
                          	index_col=0, header=0))
	print(X_inp)
	X[inp] = X_inp

# Output labels are brain activation coordinates
Y = utilities.load_coordinates(path="../")
m, n_structs = Y.shape

# Splits of the article PMIDs
splits = {}
for split in ["train", "dev", "test"]:
	splits[split] = [int(pmid.strip()) for pmid in open("../data/splits/{}.txt".format(split), "r").readlines()]


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
	
	loss[inp] = pd.read_csv("../classifiers/data/loss_con_{}_h2_3h.csv".format(inp), index_col=0, header=0)
	
	p = pd.read_csv("../classifiers/data/params_con_{}_h2_3h.csv".format(inp), index_col=0, header=None)
	params[inp]["n_hid"] = int(p.loc["n_hid"].values[0])
	params[inp]["p_dropout"] = p.loc["p_dropout"].values[0]
	params[inp]["weight_decay"] = p.loc["weight_decay"].values[0]
	params[inp]["lr"] = p.loc["lr"].values[0]

	fit[inp] = neural_network.Net(n_input=100, n_output=n_structs, n_hid=params[inp]["n_hid"], p_dropout=params[inp]["p_dropout"])
	optimizer = optim.Adam(fit[inp].parameters(), lr=params[inp]["lr"], weight_decay=params[inp]["weight_decay"])
	net_file = "../classifiers/fits/classifier_con_{}_h2_3h.pt".format(inp)
	fit[inp].load_state_dict(torch.load(net_file))


##########################################
########### Evaluation metrics ###########
##########################################

print("Plotting evaluation metrics")

for inp in inputs:

	utilities.plot_loss("{}_{}".format(prefix, inp), list(loss[inp]["LOSS"].values), 
						xlab="Epoch", ylab="Loss", alpha=0.65, color=color, print_fig=False)

	for split, pmids in splits.items():
		data_set = neural_network.load_mini_batches(X[inp], Y, splits[split], mini_batch_size=len(splits[split]))
		utilities.report_curves(data_set, fit[inp], "{}_{}_{}".format(prefix, inp, split), color=color, print_fig=False)
		report = utilities.report_metrics("results/eval_{}_{}_3h_{}.txt".format(prefix, inp, split), 
										  data_set, fit[inp], Y.columns)


