#!/usr/bin/python3

import argparse
import pandas as pd
import neural_network


parser = argparse.ArgumentParser(description='Neural network classifier with LSTM encodings')
parser.add_argument('--data', type=str, default='../lstm/encodings/lstm',
					help='base path for the text encodings')
parser.add_argument('--suffix', type=str, default='_con_titles_h0',
					help='suffix for output files')
args = parser.parse_args()


# Load the LSTM encodings
enc_train = pd.read_csv(args.data + "_train.csv", index_col=0, header=0)
enc_dev = pd.read_csv(args.data + "_dev.csv", index_col=0, header=0)
enc = enc_train.append(enc_dev)

# Optimize the classifier over a random hyperparameter grid
neural_network.optimize_classifier(enc, suffix=args.suffix)