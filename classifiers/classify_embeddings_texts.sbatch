#!/bin/bash

#SBATCH --job-name=op_emb_texts
#SBATCH --output=logs/op_emb_texts.%j.out
#SBATCH --error=logs/op_emb_texts.%j.err
#SBATCH --time=01-00:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6 py-pytorch/1.0.0_py36
srun python3 classify_embeddings_texts.py
