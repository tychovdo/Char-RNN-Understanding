python main.py data/linux/train.txt data/linux/test.txt models/linux_2_3x512_0d2_gru_50l_20000 --cuda --hidden_size 512 --dropout 0.2 --n_epochs 20000 --n_layers 3 --chunk_len 50 --rnn_class lstm
python main.py data/linux/train.txt data/linux/test.txt models/linux_2_3x512_0d2_lstm_50l_20000 --cuda --hidden_size 512 --dropout 0.2 --n_epochs 20000 --n_layers 3 --chunk_len 50 --rnn_class gru
