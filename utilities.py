#!/usr/bin/python3

import numpy as np
import pandas as pd


def load_splits(splits=["train", "dev", "test"], path="", limit=False):
	split_dict = {}
	for split in splits:
		pmids = [int(pmid.strip()) for pmid in open("../data/splits/{}.txt".format(split), "r").readlines()]
		if int(limit) > 0:
			split_dict[split] = pmids[:limit]
		else: 
			split_dict[split] = pmids
	return split_dict


def load_coordinates(path=""):
	atlas_labels = pd.read_csv("{}data/brain/labels.csv".format(path))
	activations = pd.read_csv("{}data/brain/coordinates.csv".format(path), index_col=0)
	activations = activations[atlas_labels["PREPROCESSED"]].astype(float)
	return activations


def doc_mean_thres(df):
	doc_mean = df.mean()
	df_bin = 1.0 * (df.values > doc_mean.values)
	df_bin = pd.DataFrame(df_bin, columns=df.columns, index=df.index)
	return df_bin


def load_lexicon(sources, path=""):
	lexicon = []
	for source in sources:
		file = "{}data/text/lexicon_{}.txt".format(path, source)
		lexicon += [token.strip() for token in open(file, "r").readlines()]
	return sorted(lexicon)


def load_doc_term_matrix(version=190325, binarize=True, sources=["cogneuro"], path="", inputs="texts"):
	prefix = {"texts": "dtm", "titles": "ttm"}
	dtm = pd.read_csv("{}data/text/{}_{}.csv.gz".format(path, prefix[inputs], version), compression="gzip", index_col=0)
	lexicon = load_lexicon(sources, path=path)
	lexicon = sorted(list(set(lexicon).intersection(dtm.columns)))
	dtm = dtm[lexicon]
	if binarize:
		dtm = doc_mean_thres(dtm)
	return dtm.astype(float)


def load_dtm(version=190325, binarize=True, sources=["cogneuro"], path=""):
	dtm = pd.read_csv("{}data/text/dtm_{}.csv.gz".format(path, version), compression="gzip", index_col=0)
	lexicon = load_lexicon(sources, path=path)
	lexicon = sorted(list(set(lexicon).intersection(dtm.columns)))
	dtm = dtm[lexicon]
	if binarize:
		dtm = doc_mean_thres(dtm)
	return dtm.astype(float)


def load_ttm(version=190325, binarize=True, sources=["cogneuro"], path=""):
	ttm = pd.read_csv("{}data/text/ttm_{}.csv.gz".format(path, version), compression="gzip", index_col=0)
	lexicon = load_lexicon(sources, path=path)
	lexicon = sorted(list(set(lexicon).intersection(ttm.columns)))
	ttm = ttm[lexicon]
	if binarize:
		ttm = doc_mean_thres(ttm)
	return ttm.astype(float)


# Function for stemming, conversion to lowercase, and removal of punctuation
def preprocess(text):

	import re
	from nltk.stem import WordNetLemmatizer
	from nltk.corpus import stopwords
	
	# Stop words
	stops = stopwords.words("english")

	# Convert to lowercase, convert slashes to spaces, and remove remaining punctuation except periods
	text = text.replace("-\n", "").replace("\n", " ").replace("\t", " ")
	text = "".join([char for char in text.lower() if char.isalpha() or char.isdigit() or char in [" ", "."]])
	text = text.replace(".", " . ").replace("  ", " ").strip()
	text = re.sub("\. \.+", ".", text)

	# Perform lemmatization, excluding acronyms and names in RDoC matrix
	text = " ".join([WordNetLemmatizer().lemmatize(token) for token in text.split() if token not in stops])

	# Consolidate n-grams from RDoC then select ontologies
	lexicon = load_lexicon(["cogneuro"])
	ngrams = [word.strip().replace("_", " ") for word in lexicon if "_" in word]
	for ngram in ngrams:
		text = text.replace(ngram, ngram.replace(" ", "_"))
	return text


def plot_loss(prefix, loss, xlab="", ylab="", alpha=0.5, color="gray", 
			  fname="style/computer-modern/cmunss.ttf", print_fig=True):

	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams

	font = font_manager.FontProperties(fname=fname, size=22)
	rcParams["axes.linewidth"] = 1.5

	fig = plt.figure(figsize=[3.6, 3.2])
	ax = fig.add_axes([0,0,1,1])

	# Plot the loss curve
	plt.plot(range(len(loss)), loss, alpha=alpha, 
			 c=color, linewidth=2)

	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	plt.xticks(fontproperties=font)
	plt.yticks(fontproperties=font)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)

	plt.xlabel(xlab, fontproperties=font)
	plt.ylabel(ylab, fontproperties=font)
	ax.xaxis.set_label_coords(0.5, -0.165)
	ax.yaxis.set_label_coords(-0.275, 0.5)

	plt.savefig("plots/{}_loss.png".format(prefix), 
				bbox_inches="tight", dpi=250)
	if print_fig:
		plt.show()
	plt.close()


def report_curves(data_set, net, name, color="gray", print_fig=True): 

	import torch
	from torch.autograd import Variable

	with torch.no_grad():
		inputs, labels = data_set[0]
		inputs = Variable(torch.from_numpy(inputs.T).float())
		labels = Variable(torch.from_numpy(labels.T).float())
		pred_probs = net(inputs).float()
		fpr, tpr = compute_roc(labels, pred_probs)
		prec, rec = compute_prc(labels, pred_probs)
		plot_curves("{}_roc".format(name), fpr, tpr, diag=True, alpha=0.25,
					color=color, xlab="1 - Specificity", ylab="Sensitivity", print_fig=print_fig)
		plot_curves("{}_prc".format(name), rec, prec, diag=False, alpha=0.5,
					color=color, xlab="Recall", ylab="Precision", print_fig=print_fig)


def compute_roc(labels, pred_probs):

	from sklearn.metrics import roc_curve

	fpr, tpr = [], []
	for i in range(labels.shape[1]):
		fpr_i, tpr_i, _ = roc_curve(labels[:,i], 
									pred_probs[:,i], pos_label=1)
		fpr.append(fpr_i)
		tpr.append(tpr_i)
	return fpr, tpr


def compute_prc(labels, pred_probs):

	from sklearn.metrics import precision_recall_curve
	
	precision, recall = [], []
	for i in range(labels.shape[1]):
		p_i, r_i, _ = precision_recall_curve(labels[:,i], 
											 pred_probs[:,i], pos_label=1)
		precision.append(p_i)
		recall.append(r_i)
	return precision, recall


def plot_curves(file_name, x, y, xlab="", ylab="", diag=True, alpha=0.5, color="gray", 
				fname="style/computer-modern/cmunss.ttf", print_fig=True):

	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams

	font = font_manager.FontProperties(fname=fname, size=22)
	rcParams["axes.linewidth"] = 1.5

	fig = plt.figure(figsize=[3.6, 3.2])
	ax = fig.add_axes([0,0,1,1])

	# Plot the curves
	for i in range(len(x)):
		plt.plot(x[i], y[i], alpha=alpha, 
				 c=color, linewidth=2)

	# Plot a diagonal line
	if diag:
		plt.plot([-1,2], [-1,2], linestyle="dashed", c="k", 
				 alpha=1, linewidth=2)

	plt.xlim([-0.05, 1])
	plt.ylim([-0.05, 1])
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	plt.xticks(fontproperties=font)
	plt.yticks(fontproperties=font)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)

	plt.xlabel(xlab, fontproperties=font)
	plt.ylabel(ylab, fontproperties=font)
	ax.xaxis.set_label_coords(0.5, -0.165)
	ax.yaxis.set_label_coords(-0.18, 0.5)

	plt.savefig("plots/{}.png".format(file_name), 
				bbox_inches="tight", dpi=250)
	if print_fig:
		plt.show()
	plt.close()


def report_macro(labels, pred_probs):

	from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

	n_structs = pred_probs.shape[1]
	predictions = 1.0 * (pred_probs > 0.5)
	output = ""
	output += "{:11s}{:4.4f}\n".format("F1", f1_score(labels, predictions, average="macro"))
	output += "{:11s}{:4.4f}\n".format("Precision", precision_score(labels, predictions, average="macro"))
	output += "{:11s}{:4.4f}\n".format("Recall", recall_score(labels, predictions, average="macro"))
	output += "{:11s}{:4.4f}\n".format("Accuracy", np.mean(np.sum(predictions == labels.detach().numpy(), axis=1)) / n_structs)
	output += "{:11s}{:4.4f}\n".format("ROC-AUC", roc_auc_score(labels, pred_probs, average="macro"))
	return output


def report_class(labels, pred_probs):

	from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

	predictions = 1.0 * (pred_probs > 0.5)
	output = ""
	output += "{:11s}{:4.4f}\n".format("F1", f1_score(labels, predictions, average="binary"))
	output += "{:11s}{:4.4f}\n".format("Precision", precision_score(labels, predictions, average="binary"))
	output += "{:11s}{:4.4f}\n".format("Recall", recall_score(labels, predictions, average="binary"))
	output += "{:11s}{:4.4f}\n".format("Accuracy", accuracy_score(labels, predictions))
	output += "{:11s}{:4.4f}\n".format("ROC-AUC", roc_auc_score(labels, pred_probs, average=None))
	return output

def report_metrics(file_name, data_set, net, classes):

	import torch
	from torch.autograd import Variable

	with torch.no_grad():
		inputs, labels = data_set[0]
		inputs = Variable(torch.from_numpy(inputs.T).float())
		labels = Variable(torch.from_numpy(labels.T).float())
		pred_probs = net(inputs).float().detach().numpy()
		output = ""
		output += ("-" * 50) + "\nMACRO-AVERAGED TOTAL\n" + "-" * 50 + "\n"
		output += report_macro(labels, pred_probs)
		output += "\n" + ("-" * 50) + "\n\n"
		for i in range(len(classes)):
			output += "\n" + ("-" * 50) + "\n" + classes[i].title().replace("_", " ") + "\n" + ("-" * 50) + "\n"
			output += report_class(labels[:,i], pred_probs[:,i]) + "\n"
		with open(file_name, "w+") as file:
			file.write(output)


def load_atlas(path=""):

	import numpy as np
	from nilearn import image

	cer = "{}data/brain/atlases/Cerebellum-MNIfnirt-maxprob-thr25-1mm.nii.gz".format(path)
	cor = "{}data/brain/atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz".format(path)
	sub = "{}data/brain/atlases/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz".format(path)

	sub_del_dic = {1:0, 2:0, 3:0, 12:0, 13:0, 14:0}
	sub_lab_dic_L = {4:1, 5:2, 6:3, 7:4, 9:5, 10:6, 11:7, 8:8}
	sub_lab_dic_R = {15:1, 16:2, 17:3, 18:4, 19:5, 20:6, 21:7, 7:8}

	sub_mat_L = image.load_img(sub).get_data()[91:,:,:]
	sub_mat_R = image.load_img(sub).get_data()[:91,:,:]

	for old, new in sub_del_dic.items():
		sub_mat_L[sub_mat_L == old] = new
	for old, new in sub_lab_dic_L.items():
		sub_mat_L[sub_mat_L == old] = new
	sub_mat_L = sub_mat_L + 48
	sub_mat_L[sub_mat_L == 48] = 0

	for old, new in sub_del_dic.items():
		sub_mat_R[sub_mat_R == old] = new
	for old, new in sub_lab_dic_R.items():
		sub_mat_R[sub_mat_R == old] = new
	sub_mat_R = sub_mat_R + 48
	sub_mat_R[sub_mat_R == 48] = 0

	cor_mat_L = image.load_img(cor).get_data()[91:,:,:]
	cor_mat_R = image.load_img(cor).get_data()[:91,:,:]

	mat_L = np.add(sub_mat_L, cor_mat_L)
	mat_L[mat_L > 56] = 0
	mat_R = np.add(sub_mat_R, cor_mat_R)
	mat_R[mat_R > 56] = 0

	mat_R = mat_R + 57
	mat_R[mat_R > 113] = 0
	mat_R[mat_R < 58] = 0

	cer_mat_L = image.load_img(cer).get_data()[91:,:,:]
	cer_mat_R = image.load_img(cer).get_data()[:91,:,:]
	cer_mat_L[cer_mat_L > 0] = 57
	cer_mat_R[cer_mat_R > 0] = 114

	mat_L = np.add(mat_L, cer_mat_L)
	mat_L[mat_L > 57] = 0
	mat_R = np.add(mat_R, cer_mat_R)
	mat_R[mat_R > 114] = 0

	mat = np.concatenate((mat_R, mat_L), axis=0)
	atlas_image = image.new_img_like(sub, mat)
	return atlas_image

		
def mni2vox(x, y, z):
	x = (float(x) * -1.0) + 90.0
	y = float(y) + 126.0
	z = float(z) * 2
	return (x, y, z)
	