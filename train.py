import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import string

from helpers import *
from model import *
from generate import *

# PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--save_file', type=str, default=None)
parser.add_argument('--model', type=str, default="gru")
parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--chunk_len', type=int, default=200)
parser.add_argument('--rnn_class', type=str, default='gru')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    print("CUDA Enabled")

if not args.save_file:
    args.save_file = '{}.model'.format(args.filename.split('/')[-1])

def train(model, optim, x, t):
    criterion = nn.CrossEntropyLoss()

    model.zero_grad()
    loss = 0

    hidden = None
    for c in range(args.chunk_len):
        output, hidden = model(x[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), t[:,c])

    loss.backward()
    optim.step()

    return loss.data[0] / args.chunk_len

def save(model):
    torch.save(model, args.save_file)
    print('Saved model as {}.'.format(args.save_file))

def get_train_chunk(train_file, chunk_len, batch_size):
    if args.cuda:
        x = torch.cuda.LongTensor(batch_size, chunk_len)
        t = torch.cuda.LongTensor(batch_size, chunk_len)
    else:
        x = torch.LongTensor(batch_size, chunk_len)
        t = torch.LongTensor(batch_size, chunk_len)

    for bi in range(batch_size):
        start_index = random.randint(0, len(train_file) - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = train_file[start_index:end_index]
        x[bi] = char_tensor(chunk[:-1])
        t[bi] = char_tensor(chunk[1:])

    return Variable(x), Variable(t)

def main():
    all_characters = string.printable
    n_characters = len(all_characters)

    # Init model
    model = CharRNN(n_characters, args.hidden_size, n_characters, rnn_class=args.rnn_class, n_layers=args.n_layers)
    print(model)

    # Init optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.cuda:
        model.cuda()

    start = time.time()

    # Load training file
    train_file = unidecode.unidecode(open(args.filename, encoding='utf-8', errors='ignore').read())

    # Training procedure
    try:
        print("Training for %d epochs..." % args.n_epochs)
        for epoch in range(1, args.n_epochs + 1):
            x, t = get_train_chunk(train_file, args.chunk_len, args.batch_size)
            train_loss = train(model, optim, x, t)

            if epoch % args.print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, train_loss))
                generated, _ = generate(model, 'a', cuda=args.cuda)
                print(generated, '\n')

        print("Saving...")
        save(model)

    except KeyboardInterrupt:
        print("Saving before quit...")
        save(model)

if __name__=='__main__':
    main()

