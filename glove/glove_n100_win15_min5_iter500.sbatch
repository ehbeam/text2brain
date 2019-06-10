#!/bin/bash

#SBATCH --job-name=gen_glove_n100_win15_min5_iter500
#SBATCH --output=logs/glove_n100_win15_min5_iter500.%j.out
#SBATCH --error=logs/glove_n100_win15_min5_iter500.%j.err
#SBATCH --time=07-00:00:00
#SBATCH -p normal
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --mem=8000
#SBATCH -c 2
#SBATCH --mail-type=FAIL # notifications for job failure only
#SBATCH --mail-user=ebeam@stanford.edu

srun sh run_n100_win15_min5_iter500.sh
