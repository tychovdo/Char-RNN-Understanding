#!/bin/bash
#SBATCH -n 1
#SBATCH -t 500
#SBATCH -p gpu
python main.py data/linux/train.txt data/linux/test.txt models/linux_3_3x512_0d5_lstm_50l_50000 --cuda --hidden_size 512 --dropout 0.2 --n_epochs 50000 --n_layers 3 --chunk_len 128 --rnn_class lstm --print_every 2000 >> log1.txt &
python main.py data/linux/train.txt data/linux/test.txt models/linux_3_3x512_0d5_gru_50l_50000 --cuda --hidden_size 512 --dropout 0.2 --n_epochs 50000 --n_layers 3 --chunk_len 128 --rnn_class gru --print_every 2000 >> log2.txt
