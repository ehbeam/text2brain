#!/usr/bin/python3

for data in ["texts", "titles"]:
	for state in ["hidden", "cell"]:
		for layer in [0, 1, 2]:

			encodings = "../lstm/encodings/lstm_{}_{}{}_100d".format(data, state[0], layer)
			suffix = "_con_{}_{}{}".format(data, state[0], layer)

			lines = ["#!/bin/bash\n",
					 "#SBATCH --job-name=op{}".format(suffix),
					 "#SBATCH --output=logs/op{}.%j.out".format(suffix),
					 "#SBATCH --error=logs/op{}.%j.err".format(suffix),
					 "#SBATCH --time=01-00:00:00",
					 "#SBATCH -p aetkin",
					 "#SBATCH --mail-type=FAIL",
					 "#SBATCH --mail-user=ebeam@stanford.edu\n",
					 "module load python/3.6 py-pytorch/1.0.0_py36",
					 "srun python3 classify_contexts.py --data '{}' --suffix '{}'".format(encodings, suffix)]

			outfile = "classify_contexts_{}_{}{}.sbatch".format(data, state[0], layer)
			with open(outfile, "w+") as file:
				file.write("\n".join(lines))