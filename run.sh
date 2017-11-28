python main.py data/linux/train.txt data/linux/test.txt linux_3x512_0d5_20000 --cuda --hidden_size 512 --dropout 0.5 --n_epochs 20000 --n_layers 3 --rnn_class lstm
python main.py data/linux/train.txt data/linux/test.txt linux_3x512_0d5_20000 --cuda --hidden_size 512 --dropout 0.5 --n_epochs 20000 --n_layers 3 --rnn_class gru
