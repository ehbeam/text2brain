#!/bin/bash

#SBATCH --job-name=lstm_2000epochs_gpu_raw
#SBATCH --output=logs/lstm_2000epochs_gpu_raw.%j.out
#SBATCH --error=logs/lstm_2000epochs_gpu_raw.%j.err
#SBATCH --time=00-08:00:00
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH --gres gpu:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6 py-pytorch/1.0.0_py36
srun python3 main.py --cuda --epochs 2000 --save "fits/lstm_2000epochs_gpu_raw.pt" --data "../data/text/raw" --model "LSTM" --emsize 100 --nhid 100 --nlayers 3 --clip 1 --batch_size 64 --bptt 15 --dropout 0.5 --seed 42 --log-interval 100
