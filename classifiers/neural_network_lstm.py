#!/usr/bin/python3


import os
import argparse
import pandas as pd
import numpy as np
np.random.seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
torch.manual_seed(42)

import sys
sys.path.append("..")
import utilities
import neural_network


# Select device to train on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Creates an embedding layer for terms in the lexicon
def create_emb_layer(emb):
	n_terms, n_emb = emb.shape
	emb_layer = nn.Embedding(n_terms, n_emb)
	emb_layer.load_state_dict({"weight": torch.tensor(emb.values)})
	return emb_layer, n_terms, n_emb


# Neural network classifier with LSTM encoding
class LSTM2Net(nn.Module):
	def __init__(self, vsm, n_input=100, n_hid=100, n_output=100,
				 n_lay=3, p_dropout=0.5, batch_size=1024):
		super(LSTM2Net, self).__init__()
		self.n_lay = n_lay
		self.n_hid = n_hid
		self.embedding, n_terms, n_emb = create_emb_layer(vsm)
		self.lstm = nn.LSTM(n_emb, n_hid, n_lay)
		self.fc1 = nn.Linear(n_hid, n_hid)
		self.bn1 = nn.BatchNorm1d(n_hid)
		self.dropout1 = nn.Dropout(p=p_dropout)
		self.fc2 = nn.Linear(n_hid, n_hid)
		self.bn2 = nn.BatchNorm1d(n_hid)
		self.dropout2 = nn.Dropout(p=p_dropout)
		self.fc3 = nn.Linear(n_hid, n_hid)
		self.bn3 = nn.BatchNorm1d(n_hid)
		self.dropout3 = nn.Dropout(p=p_dropout)
		self.fc4 = nn.Linear(n_hid, n_output)

		# Xavier initialization for classifier weights
		for fc in [self.fc1, self.fc2, self.fc3, self.fc4]:
			nn.init.xavier_uniform_(fc.weight)

	# Forward propagation
	def forward(self, inputs, hidden):
		outputs, (ht, ct) = self.lstm(self.embedding(inputs), hidden)
		x = self.dropout1(F.relu(self.bn1(self.fc1(ht[-1])))) # Use the last hidden layer
		x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
		x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
		x = torch.sigmoid(self.fc4(x))
		return x, ht
	
	def init_hidden(self, batch_size):
		return(Variable(torch.randn(self.n_lay, batch_size, self.n_hid)).to(device),
			   Variable(torch.randn(self.n_lay, batch_size, self.n_hid)).to(device))


# Converts data from numpy to torch format
def numpy2torch(data):
	inputs, labels = data
	labels = Variable(torch.from_numpy(labels.T).float())
	return inputs, labels


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# Optimizes classifier hyperparameters
def optimize_classifier(vsm, lexicon, data, suffix="", batch_size=64, n_epochs=50, n_iter=50):

	from sklearn.metrics import roc_auc_score
	
	# Load the data splits
	splits = utilities.load_splits(splits=["train", "dev"], path="../")
	pmids = splits["train"] + splits["dev"]

	# Load the activation coordinate labels
	Y = utilities.load_coordinates(path="../")

	# Load the PMIDs
	X = pd.DataFrame(pmids, index=pmids)

	# Map terms to lexicon indices

	term2idx = {term: i for i, term in enumerate(lexicon)}

	# Map text inputs to lexicon indices
	pmid2idx = {}
	for pmid in pmids:
		terms = open("../data/text/corpus/{}/{}.txt".format(data, pmid)).read().split()
		terms = [term for term in terms if term in lexicon]
		pmid2idx[int(pmid)] = [term2idx[term] for term in terms]
	
	# Sizes for input and output layers
	n_input = len(lexicon)
	n_output = Y.shape[1]

	# Load the mini batches
	train_set = neural_network.load_mini_batches(X, Y, splits["train"], mini_batch_size=batch_size, seed=42)
	dev_set = neural_network.load_mini_batches(X, Y, splits["dev"], mini_batch_size=len(splits["dev"]), seed=42)
	dev_set = numpy2torch(dev_set[0])
	pmids_dev, labels_dev = dev_set[0][0], dev_set[1].to(device)
	inputs_dev = [pmid2idx[int(pmid)] for pmid in pmids_dev]
	max_len_dev = max([len(idx) for idx in inputs_dev])
	padded_dev = [idx + [0] * (max_len_dev - len(idx)) for idx in inputs_dev]
	inputs_dev = torch.t(torch.tensor(padded_dev).view(len(pmids_dev), max_len_dev)).to(device)

	# Specify the randomized hyperparameter grid
	param_list = neural_network.load_hyperparam_grid(n_iter=n_iter)
	criterion = F.binary_cross_entropy

	# Loop through hyperparameter combinations
	op_idx, op_params, op_score_dev, op_state_dict, op_loss = 0, 0, 0, 0, 0
	for params in param_list:

		print("-" * 75)
		print("   ".join(["{} {}".format(k.upper(), v) for k, v in params.items()]))
		print("-" * 75 + "\n")

		# Initialize variables for this set of parameters
		net = LSTM2Net(vsm.loc[lexicon], n_input=n_input, n_output=n_output, n_hid=params["n_hid"], 
					   n_lay=3, p_dropout=params["p_dropout"])
		optimizer = optim.Adam(net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
		net.apply(neural_network.reset_weights)
		net.to(device)

		# Loop over the dataset multiple times
		running_loss = []
		for epoch in range(n_epochs): 
			for n_batch, data in enumerate(train_set):
				
				# Get the inputs
				data = numpy2torch(data)
				batch_pmids, labels = data[0][0], data[1].to(device)
				inputs = [pmid2idx[int(pmid)] for pmid in batch_pmids]
				max_len = max([len(idx) for idx in inputs_dev])
				padded = [idx + [0] * (max_len - len(idx)) for idx in inputs]
				inputs = torch.t(torch.tensor(padded).view(len(batch_pmids), max_len)).to(device)
				
				# Zero the parameter gradients
				hidden = net.init_hidden(len(batch_pmids))
				hidden = repackage_hidden(hidden)
				optimizer.zero_grad()

				# Forward + backward + optimize
				outputs, hidden = net(inputs, hidden)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
			
			# Update the running loss
			running_loss += [loss.item()]
			if epoch % (n_epochs/5) == (n_epochs/5) - 1:
				print("   Epoch {:3d}\tLoss {:6.6f}".format(epoch + 1, running_loss[-1])) 
		
		# Evaluate on the validation set
		with torch.no_grad():
			hidden = net.init_hidden(len(pmids_dev))
			hidden = repackage_hidden(hidden)
			preds_dev, hidden = net(inputs_dev, hidden)
		score_dev = roc_auc_score(labels_dev.cpu(), preds_dev.cpu().float(), average="macro")
		print("\n   Dev set ROC-AUC {:6.4f}\n".format(score_dev))
		
		# Update outputs if this model is the best so far
		if score_dev > op_score_dev:
			if len(param_list) > 1:
				print("   Best so far!\n")
			op_score_dev = score_dev
			op_state_dict = net.state_dict()
			op_params = params
			op_loss = running_loss

			# Export the trained neural network
			fit_file = "fits/classifier{}_3h.pt".format(suffix)
			torch.save(op_state_dict, fit_file)

			# Export the hyperparameters
			param_file = "data/params{}_3h.csv".format(suffix)
			with open(param_file, "w+") as file:
				file.write("\n".join(["{},{}".format(param, val) for param, val in op_params.items()]))

			# Export the loss over epochs
			loss_file = "data/loss{}_3h.csv".format(suffix)
			pd.DataFrame(op_loss, index=None, columns=["LOSS"]).to_csv(loss_file)
