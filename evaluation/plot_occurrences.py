#!/usr/bin/python3

import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import utilities
from classifiers import neural_network


##########################################
########## Plotting parameters ###########
##########################################

color = "#873434" # Color for plotting evaluation metrics and word clouds
cmap = "Reds" # Color map for plotting brain structures
prefix = "occ" # Prefix for plot file names
n_top = 15 # Number of terms (i.e., inputs) to plot per brain structure (i.e., class)
inputs = ["texts", "titles"] # Whether the inputs are from article full texts or titles
batch_size = 64
verbose = True
print_fig = False


##########################################
########### Inputs and labels ############
##########################################

print("Loading inputs and labels")

# Vector space model of GloVe embeddings trained on article full texts
vsm = pd.read_csv("../data/text/glove_gen_n100_win15_min5_iter500_190428.txt", sep = " ", index_col=0, header=0)
n_vocab, n_emb = vsm.shape

# Document-term matrix generated from article full texts or titles
dtm = {}
for inp in inputs:
	dtm_inp = utilities.load_doc_term_matrix(path="../", inputs=inp)
	dtm[inp] = dtm_inp[dtm_inp.columns.intersection(vsm.index)]
X = dtm

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
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(42)

criterion = F.binary_cross_entropy

loss, fit = {}, {}
params = {inp: {} for inp in inputs}
for inp in inputs:

	loss[inp] = pd.read_csv("../classifiers/data/loss_{}_{}_3h.csv".format(prefix, inp), index_col=0, header=0)
	
	p = pd.read_csv("../classifiers/data/params_{}_{}_3h.csv".format(prefix, inp), index_col=0, header=None)
	params[inp]["n_hid"] = int(p.loc["n_hid"].values[0])
	params[inp]["p_dropout"] = p.loc["p_dropout"].values[0]
	params[inp]["weight_decay"] = p.loc["weight_decay"].values[0]
	params[inp]["lr"] = p.loc["lr"].values[0]

	fit[inp] = neural_network.Net(n_input=X[inp].shape[1], n_output=n_structs, n_hid=params[inp]["n_hid"], p_dropout=params[inp]["p_dropout"])
	optimizer = optim.Adam(fit[inp].parameters(), lr=params[inp]["lr"], weight_decay=params[inp]["weight_decay"])
	net_file = "../classifiers/fits/classifier_{}_{}_3h.pt".format(prefix, inp)
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


##########################################
############## Feature maps ##############
##########################################

print("Plotting feature maps\n")

import os
from nilearn import image, plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
from wordcloud import WordCloud
from PIL import Image, ImageOps

for inp in inputs:

	terms = list(dtm[inp].columns)
	term_map = pd.DataFrame(index=Y.columns, columns=dtm[inp].columns)

	for i, term in enumerate(terms):

		fit[inp] = neural_network.Net(n_input=X[inp].shape[1], n_output=n_structs, n_hid=params[inp]["n_hid"], p_dropout=params[inp]["p_dropout"])
		optimizer = optim.Adam(fit[inp].parameters(), lr=params[inp]["lr"], weight_decay=params[inp]["weight_decay"])
		net_file = "../classifiers/fits/classifier_{}_{}_3h.pt".format(prefix, inp)
		fit[inp].load_state_dict(torch.load(net_file))
		
		inputs = np.zeros((batch_size, X[inp].shape[1]))
		inputs[:, i] = 1.0
		inputs = Variable(torch.from_numpy(inputs).float())
		pred_probs = fit[inp](inputs).float().detach().numpy()
		term_map[term] = pred_probs[0,:]

	c = pd.read_csv("../data/brain/labels.csv", index_col=None, header=0)
	struct2coord = {}
	for struct, x, y, z in zip(c["PREPROCESSED"], c["X"], c["Y"], c["Z"]):
		struct2coord[struct] = utilities.mni2vox(x, y, z)

	for i, struct in enumerate(Y.columns):
		struct = Y.columns[i]
		outfile = "maps/{}/{}/{}.png".format(prefix, inp, struct)
		
		if not os.path.exists(outfile):
			x, y, z = struct2coord[struct]
			if verbose:
				print("{} (z={})".format(struct.title().replace("_", " "), int(z)))
			if not print_fig:
				plt.ioff()

			fig, ax = plt.subplots(1,2, figsize=(6,6))
			gs1 = gridspec.GridSpec(1,2)
			gs1.update(wspace=-20, hspace=-10)
			fname ="style/computer-modern/cmunss.ttf"
			fig.suptitle(c["PRESENTABLE_TITLE"][i], y=0.79,
						 fontproperties=font_manager.FontProperties(fname=fname, size=24))

			bg_img = image.load_img("../data/brain/atlases/MNI152_T1_1mm_brain.nii.gz")
			bg_img = np.flip(np.rot90(bg_img.get_data()[:,:,int(z)]).astype(float), axis=1)
			bg_img[bg_img == 0] = np.nan
			bg_img = bg_img[10:198, 20:162]

			bilateral_atlas = utilities.load_atlas(path="../")
			struct_img = np.flip(np.rot90(bilateral_atlas.get_data()[:,:,int(z)]), axis=1)
			struct_img[struct_img != i+1] = np.nan
			struct_img[struct_img == i+1] = 1.0
			struct_img[struct_img == 0] = np.nan
			struct_img = struct_img[10:198, 20:162]
		
			ax[0].imshow(bg_img, cmap="Greys_r", alpha=0.7, vmin=1)
			ax[0].imshow(struct_img, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
			for side in ["left", "right", "top", "bottom"]:
				ax[0].spines[side].set_visible(False)
			ax[0].set_xticks([])
			ax[0].set_yticks([])

			def color_func(word, font_size, position, orientation, 
					   random_state=None, idx=0, **kwargs):
				return color

			top = term_map.loc[struct].sort_values(ascending=False)[:n_top]
			vals = top.values
			tkns = [t.replace("_", " ") for t in top.index]
			cloud = WordCloud(background_color="rgba(255, 255, 255, 0)", mode="RGB", 
							  max_font_size=180, min_font_size=10, 
							  prefer_horizontal=1, scale=20, margin=3,
							  width=1200, height=1400, font_path=fname, 
							  random_state=42).generate_from_frequencies(zip(tkns, vals))
			ax[1].imshow(cloud.recolor(color_func=color_func, random_state=42))
			ax[1].axis("off")

			plt.savefig(outfile, bbox_inches="tight", dpi=250)
			if print_fig:
				plt.show()
			plt.close()

	# Combine brain plots into a single figure
	images = ["maps/{}/{}/{}.png".format(prefix, inp, struct) for struct in Y.columns]
	img_w, img_h = Image.open(images[0]).size

	pad_w, pad_h = 60, 30
	img_w += pad_w * 2
	img_h += pad_h * 2
	n_row, n_col = 19, 6
	fig_w = n_col * img_w
	fig_h = n_row * img_h
	x_coords = list(range(0, fig_w, img_w)) * n_row
	y_coords = np.repeat(list(range(0, fig_h, img_h)), n_col)
	padding = (pad_w, pad_h, pad_w, pad_h)
	white = (255,255,255,0)

	figure = Image.new("RGB", (fig_w, fig_h), color=white)
	for i, img in enumerate(images):
		img = Image.open(img)
		img = ImageOps.expand(img, padding, fill=white)
		figure.paste(img, (x_coords[i], y_coords[i]))
	figure.save("maps/{}_{}.png".format(prefix, inp))