#!/bin/bash

# Set this to the IP address of the AWS instance
IP="52.33.122.220"

# 1. Set up the file structure
# scp -i ../../cs230.pem ../setup_dirs.sh  ubuntu@${IP}:~

# 2. Upload scripts and inputs to training
# scp -i ../../cs230.pem ../utilities.py  ubuntu@${IP}:~
# scp -i ../../cs230.pem neural_network.py  ubuntu@${IP}:~/classifiers
# scp -i ../../cs230.pem neural_network_lstm.py  ubuntu@${IP}:~/classifiers
# scp -i ../../cs230.pem classify_contexts_end2end.py  ubuntu@${IP}:~/classifiers
# scp -i ../../cs230.pem neural_network_cnn.py  ubuntu@${IP}:~/classifiers
# scp -i ../../cs230.pem classify_cnn.py  ubuntu@${IP}:~/classifiers
# scp -i ../../cs230.pem classify_embeddings_gpu.py  ubuntu@${IP}:~/classifiers
# scp -i ../../cs230.pem classify_occurrences_gpu.py  ubuntu@${IP}:~/classifiers
# scp -i ../../cs230.pem ../data/splits/train.txt  ubuntu@${IP}:~/data/splits
# scp -i ../../cs230.pem ../data/splits/dev.txt  ubuntu@${IP}:~/data/splits
# scp -i ../../cs230.pem ../data/splits/test.txt  ubuntu@${IP}:~/data/splits
# scp -i ../../cs230.pem ../data/brain/coordinates.csv  ubuntu@${IP}:~/data/brain
# scp -i ../../cs230.pem ../data/brain/labels.csv  ubuntu@${IP}:~/data/brain
# scp -i ../../cs230.pem ../data/text/lexicon_cogneuro.txt  ubuntu@${IP}:~/data/text
# scp -i ../../cs230.pem ../data/text/ttm_190325.csv.gz  ubuntu@${IP}:~/data/text
# scp -i ../../cs230.pem ../data/text/dtm_190325.csv.gz  ubuntu@${IP}:~/data/text
# scp -i ../../cs230.pem ../data/text/glove_gen_n100_win15_min5_iter500_190428.txt ubuntu@${IP}:~/data/text
# scp -i ../../cs230.pem -r ../data/text/corpus/titles ubuntu@${IP}:~/data/text/corpus/titles

# 3. Download outputs from training
# scp -i ../../cs230.pem ubuntu@${IP}:~/classifiers/fits/classifier_emb_gpu.pt aws
# scp -i ../../cs230.pem ubuntu@${IP}:~/classifiers/data/params_emb_gpu.pt aws
# scp -i ../../cs230.pem ubuntu@${IP}:~/classifiers/data/loss_emb_gpu.pt aws
# scp -i ../../cs230.pem ubuntu@${IP}:~/classifiers/fits/classifier_occ_gpu.pt aws
# scp -i ../../cs230.pem ubuntu@${IP}:~/classifiers/data/params_occ_gpu.pt aws
# scp -i ../../cs230.pem ubuntu@${IP}:~/classifiers/data/loss_occ_gpu.pt aws
scp -i ../../cs230.pem ubuntu@${IP}:~/classifiers/fits/classifier_cnn_texts_3h.pt fits
scp -i ../../cs230.pem ubuntu@${IP}:~/classifiers/data/loss_cnn_texts_3h.csv data
scp -i ../../cs230.pem ubuntu@${IP}:~/classifiers/data/params_cnn_texts_3h.csv data