#!/bin/bash

# Set this to the IP address of the AWS instance
IP="54.245.13.225"

# Upload inputs to training
# scp -i ../../cs230.pem ../setup_dirs.sh  ubuntu@${IP}:~
# scp -i ../../cs230.pem data.py  ubuntu@${IP}:~/lstm
# scp -i ../../cs230.pem main.py  ubuntu@${IP}:~/lstm
# scp -i ../../cs230.pem model.py  ubuntu@${IP}:~/lstm
# scp -i ../../cs230.pem ../data/text/lexicon.txt  ubuntu@${IP}:~/data/text
# scp -i ../../cs230.pem ../data/text/raw_orig_split/corpus_train.txt  ubuntu@${IP}:~/data/text/raw_orig_split
# scp -i ../../cs230.pem ../data/text/raw_orig_split/corpus_dev.txt  ubuntu@${IP}:~/data/text/raw_orig_split
# scp -i ../../cs230.pem ../data/text/raw_orig_split/corpus_test.txt  ubuntu@${IP}:~/data/text/raw_orig_split

# scp -i ../../cs230.pem ../data/text/lstm/raw_5000train/corpus_train.txt  ubuntu@${IP}:~/data/text/lstm/raw_5000train
# scp -i ../../cs230.pem ../data/text/lstm/raw_5000train/corpus_dev.txt  ubuntu@${IP}:~/data/text/lstm/raw_5000train
# scp -i ../../cs230.pem ../data/text/lstm/raw_5000train/corpus_test.txt  ubuntu@${IP}:~/data/text/lstm/raw_5000train

# scp -i ../../cs230.pem -r ../data/text/corpus/texts ubuntu@${IP}:~/data/text/corpus/texts
# scp -i ../../cs230.pem -r encode.py ubuntu@${IP}:~/lstm
# scp -i ../../cs230.pem -r ../utilities.py ubuntu@${IP}:~
# scp -i ../../cs230.pem -r fits/lstm_5000texts-raw.pt ubuntu@${IP}:~/lstm/fits

# scp -i ../../cs230.pem ../data/splits/train.txt  ubuntu@${IP}:~/data/splits
# scp -i ../../cs230.pem ../data/splits/dev.txt  ubuntu@${IP}:~/data/splits
# scp -i ../../cs230.pem ../data/splits/test.txt  ubuntu@${IP}:~/data/splits


# # Download output from training
# scp -i ../../cs230.pem ubuntu@${IP}:~/lstm/fits/lstm_3epochs_raw_5000train.pt fits

scp -i ../../cs230.pem ubuntu@${IP}:~/lstm/encodings/lstm_texts_h2_100d_train.csv encodings
scp -i ../../cs230.pem ubuntu@${IP}:~/lstm/encodings/lstm_texts_h2_100d_dev.csv encodings
scp -i ../../cs230.pem ubuntu@${IP}:~/lstm/encodings/lstm_texts_h2_100d_test.csv encodings