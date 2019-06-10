#!/usr/bin/python3

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


# Neural network classifier
class CNN2Net(nn.Module):
	def __init__(self, n_input=0, n_output=0, n_hid=100, p_dropout=0.5, n_terms=1542):
		super(CNN2Net, self).__init__()
		self.conv = nn.Conv2d(in_channels=1, out_channels=n_hid, kernel_size=(100, n_terms))
		self.pool = nn.MaxPool2d(2)
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
		
		# Xavier initialization for weights
		for fc in [self.fc1, self.fc2, self.fc3, self.fc4]:
			nn.init.xavier_uniform_(fc.weight)

	# Forward propagation
	def forward(self, x):
		x = F.relu(self.conv(x.unsqueeze(1)))
		x = self.pool(x).squeeze(2).squeeze(2)
		x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
		x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
		x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
		x = torch.sigmoid(self.fc4(x))
		return x


# Loads list of random mini-batches
def load_mini_batches(X, Y, split, mini_batch_size=64, seed=42, reshape_labels=False):
	"""
	Creates a list of random mini-batches from (X, Y)
	
	Arguments:
	X -- input data, of shape (input size, number of examples)
	Y -- true "label" vector (1 / 0), of shape (1, number of examples)
	mini_batch_size -- size of the mini-batches, integer
	
	Returns:
	mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
	"""

	import math

	np.random.seed(seed)			
	m = len(split) # Number of training examples
	mini_batches = []

	# Split the data
	Y = Y.loc[split].T.values
		
	# Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[:, :, permutation]
	shuffled_Y = Y[:, permutation]
	if reshape_labels:
		shuffled_Y = shuffled_Y.reshape((1,m))

	# Partition (shuffled_X, shuffled_Y), except the end case
	num_complete_minibatches = math.floor(m / mini_batch_size) # Mumber of mini batches of size mini_batch_size in your partitionning
	for k in range(0, int(num_complete_minibatches)):
		mini_batch_X = shuffled_X[:, :, k * mini_batch_size : (k+1) * mini_batch_size]
		mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	# Handle the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, :, -(m % mini_batch_size):]
		mini_batch_Y = shuffled_Y[:, -(m % mini_batch_size):]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	return mini_batches


# Optimizes classifier hyperparameters
def optimize_classifier(X_train, X_dev, suffix="", batch_size=64, n_epochs=50, n_iter=50):

	from sklearn.metrics import roc_auc_score
	
	# Select device to train on
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Load the data splits
	splits = utilities.load_splits(splits=["train", "dev"], path="../", limit=5000)

	# Load the activation coordinate labels
	Y = utilities.load_coordinates(path="../")

	# Sizes for input and output layers
	n_input = X_train.shape[1]
	n_output = Y.shape[1]

	# Load the mini batches
	train_set = load_mini_batches(X_train, Y, splits["train"], mini_batch_size=batch_size, seed=42)
	dev_set = load_mini_batches(X_dev, Y, splits["dev"], mini_batch_size=len(splits["dev"]), seed=42)
	dev_set = neural_network.numpy2torch(dev_set[0])
	# inputs_dev, labels_dev = dev_set[0][:,:,500].to(device), dev_set[1][:,500].to(device)
	inputs_dev, labels_dev = dev_set[0].to(device), dev_set[1].to(device)

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
		net = CNN2Net(n_input=n_input, n_output=n_output, 
				  n_hid=params["n_hid"], p_dropout=params["p_dropout"])
		optimizer = optim.Adam(net.parameters(), lr=params["lr"], 
							   weight_decay=params["weight_decay"])
		net.apply(neural_network.reset_weights)
		net.to(device)

		# Loop over the dataset multiple times
		running_loss = []
		for epoch in range(n_epochs): 
			for data in train_set:

				# Get the inputs
				data = neural_network.numpy2torch(data)
				inputs, labels = data[0].to(device), data[1].to(device)
				
				# Zero the parameter gradients
				optimizer.zero_grad()

				# Forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
			
			# Update the running loss
			running_loss += [loss.item()]
			print("   Epoch {:3d}\tLoss {:6.6f}".format(epoch+1, running_loss[-1])) 
		
			# Evaluate on the validation set
			with torch.no_grad():
				preds_dev = net(inputs_dev)
			score_dev = roc_auc_score(labels_dev.cpu(), preds_dev.cpu().float(), average="macro")
			print("   Dev set ROC-AUC {:6.4f}\n".format(score_dev))
			
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


