#!/bin/bash

#SBATCH --job-name=op_cnn_texts
#SBATCH --output=logs/op_cnn_texts.%j.out
#SBATCH --error=logs/op_cnn_texts.%j.err
#SBATCH --time=00-10:00:00
#SBATCH -p aetkin
#SBATCH --mem 20G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6 py-pytorch/1.0.0_py36
srun python3 classify_cnn.py --data "texts"
