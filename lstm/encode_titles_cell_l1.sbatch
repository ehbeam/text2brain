#!/bin/bash

#SBATCH --job-name=tit_c1
#SBATCH --output=logs/tit_c1.%j.out
#SBATCH --error=logs/tit_c1.%j.err
#SBATCH --time=00-12:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6 py-pytorch/1.0.0_py36
srun python3 encode.py --input '../data/text/corpus/titles' --data '../data/text/lstm/titles' --model 'fits/lstm_titles.pt' --state 'cell' --layer 1 --save 'encodings/lstm_titles'

