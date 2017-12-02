import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import string

from helpers import *
from train import *
from model import *
from generate import *

# PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('train_file', type=str)
parser.add_argument('test_file', type=str)
parser.add_argument('save_file', type=str)
parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--print_every', type=int, default=500)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--chunk_len', type=int, default=200)
parser.add_argument('--rnn_class', type=str, default='gru')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    print("CUDA Enabled")

def save(model, filename):
    ''' Save a model '''
    torch.save(model, filename)
    print('Saved model as {}.'.format(filename))

def main():
    all_characters = string.printable
    n_characters = len(all_characters)

    # Init model
    model = CharRNN(n_characters, args.hidden_size, n_characters,
                    rnn_class=args.rnn_class, n_layers=args.n_layers,
                    dropout=args.dropout)

    for weights in model.parameters():
        nn.init.uniform(weights, -0.08, 0.08)
    print(model)

    # Init optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.cuda:
        model.cuda()

    start = time.time()

    # Load training file
    train_file = unidecode.unidecode(open(args.train_file, encoding='utf-8', errors='ignore').read())
    test_file = unidecode.unidecode(open(args.test_file, encoding='utf-8', errors='ignore').read())

    # Training procedure
    try:
        print("Training for %d epochs..." % args.n_epochs)
        train_losses = []
        for epoch in range(1, args.n_epochs + 1):
            x, t = get_batch(train_file, args.chunk_len, args.batch_size, args.cuda)
            train_losses.append(train(model, optim, x, t))

            if epoch % args.print_every == 0:
                test_count = 100
                test_losses = []
                for i in range(test_count):
                    x, t  = get_batch(test_file, args.chunk_len, args.batch_size, args.cuda)
                    test_losses.append(test(model, optim, x, t))
                test_loss = np.mean(test_losses)
                train_loss = np.mean(train_losses)

                print('[%s (%d %d%%) train:%.4f test:%.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, train_loss, test_loss))
                train_losses = []

                generated, _ = generate(model, 'a', cuda=args.cuda)
                print(generated, '\n')

                print("Saving...")
                save(model, '{}_E{}'.format(args.save_file, epoch))

        print("Saving final file: ", args.save_file)
        save(model, args.save_file)

    except KeyboardInterrupt:
        print("Saving before quit...")
        save(model, '{}_Q{}'.format(args.save_file, epoch))

if __name__=='__main__':
    main()

