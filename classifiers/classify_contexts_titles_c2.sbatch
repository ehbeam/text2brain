#!/bin/bash

#SBATCH --job-name=op_con_titles_c2
#SBATCH --output=logs/op_con_titles_c2.%j.out
#SBATCH --error=logs/op_con_titles_c2.%j.err
#SBATCH --time=01-00:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6 py-pytorch/1.0.0_py36
srun python3 classify_contexts.py --data '../lstm/encodings/lstm_titles_c2_100d' --suffix '_con_titles_c2'