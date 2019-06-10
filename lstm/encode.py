#!/usr/bin/python3

import argparse
import data
import torch
torch.manual_seed(42)

import pandas as pd
import numpy as np
np.random.seed(42)

import sys
sys.path.append("..")
sys.path.insert(0, "./lstm")
import utilities


parser = argparse.ArgumentParser(description='Encoding for PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--input', type=str, default='../data/text/corpus/titles',
					help='location of the texts to encode')
parser.add_argument('--data', type=str, default='../data/text/lstm/titles',
					help='location of the data corpus used to train the model')
parser.add_argument('--model', type=str, default='fits/lstm_titles.pt',
					help='trained model for generating encodings')
parser.add_argument('--n_hid', type=int, default=100,
					help='number of hidden layers in model')
parser.add_argument('--state', type=str, default='hidden',
					help='state to encode (hidden or cell)')
parser.add_argument('--layer', type=int, default=0,
					help='layer to encode')
parser.add_argument('--save', type=str, default='encodings/lstm_titles',
					help='base path for saving the encodings')
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
args = parser.parse_args()


class Document(object):
	def __init__(self, doc_words):
		self.doc = self.tokenize(doc_words)

	def tokenize(self, doc_words):
		"""Tokenizes a single document."""
		doc_words += " <eos>"
		ids = []
		for token, word in enumerate(doc_words.split()):
			if word in corpus.dictionary.word2idx.keys():
				ids.append(corpus.dictionary.word2idx[word])
		return torch.LongTensor(ids)


# Select device to train on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the LSTM model
model = torch.load(open(args.model, "rb"), map_location=device).to(device)

# Index of the state
if args.state == "hidden":
	state = 0
elif args.state == "cell":
	state = 1
else:
	raise ValueError("""An invalid option for `--state` was supplied,
					 options are ['hidden', 'cell']""")

# Load the corpus
corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

# Load the data splits
splits = utilities.load_splits(path="../")

# Snippet length
n_snip = 100

# Generate document encodings
for split, pmids in splits.items():
	print("Encoding the {} set".format(split))
	encoding = np.zeros((len(pmids), args.n_hid))
	with torch.no_grad():
		for j, pmid in enumerate(pmids):
			hidden = model.init_hidden(1)
			text = open("{}/{}.txt".format(args.input, pmid)).read()
			words = text.split()
			idx = np.random.choice(range(len(words)-n_snip))
			text = " ".join(words[idx:(idx+n_snip)])
			tokens = Document(text).doc
			for token in tokens:
				token = token.view((1, 1)).to(device)
				output, hidden = model.forward(token, hidden)
			encoding[j,:] = hidden[state][args.layer].cpu().numpy()
			if j % 10 == 0:
				print("   Processed the {}th document".format(j))
		
		# Export the encodings
		output = pd.DataFrame(encoding, index=pmids)
		output.to_csv("{}_{}{}_{}d_{}snip_{}.csv".format(
					  args.save, args.state[0], args.layer, args.n_hid, n_snip, split))

