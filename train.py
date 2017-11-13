import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from helpers import *
from model import *
from generate import *

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--model', type=str, default="gru")
parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--chunk_len', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    print("CUDA Enabled")

file, file_len = read_file(args.filename)

def random_training_set(chunk_len, batch_size):
    x = torch.LongTensor(batch_size, chunk_len)
    t = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        x[bi] = char_tensor(chunk[:-1])
        t[bi] = char_tensor(chunk[1:])
    x, t = Variable(x), Variable(t)
    if args.cuda:
        return x.cuda(), t.cuda()
    else:
        return x, t

def train(model, optim, x, t):
    criterion = nn.CrossEntropyLoss()

    hidden = model.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    model.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = model(x[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), t[:,c])

    loss.backward()
    optim.step()

    return loss.data[0] / args.chunk_len

def save(model):
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)


def main():
    model = CharRNN(n_characters, args.hidden_size, n_characters, model=args.model, n_layers=args.n_layers)
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.cuda:
        model.cuda()

    start = time.time()
    all_losses = []
    loss_avg = 0

    try:
        print("Training for %d epochs..." % args.n_epochs)
        for epoch in range(1, args.n_epochs + 1):
            x, t = random_training_set(args.chunk_len, args.batch_size)
            loss = train(model, optim, x, t)
            loss_avg += loss

            if epoch % args.print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
                generated, _ = generate(model, 'a', cuda=args.cuda)
                print(generated, '\n')

        print("Saving...")
        save(model)

    except KeyboardInterrupt:
        print("Saving before quit...")
        save(model)

if __name__=='__main__':
    main()

