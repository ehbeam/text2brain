#!/bin/bash

#SBATCH --job-name=txt_c2
#SBATCH --output=logs/txt_c2.%j.out
#SBATCH --error=logs/txt_c2.%j.err
#SBATCH --time=00-12:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6 py-pytorch/1.0.0_py36
srun python3 encode.py --input '../data/text/corpus/texts' --data '../data/text/lstm/raw_5000train' --model 'fits/lstm_5000texts-raw.pt' --state 'cell' --layer 2 --save 'encodings/lstm_texts'
