import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNN(nn.Module):
<<<<<<< HEAD
    def __init__(self, input_size, hidden_size, output_size, rnn_class='gru', n_layers=1, dropout=0.0):
=======
    def __init__(self, input_size, hidden_size, output_size, rnn_class, n_layers=1, act_fn='tanh'):
>>>>>>> 2badd524df6944f3ccfb539b64655d223e72ce6e
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_class = rnn_class
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)

        if rnn_class == 'lstm':
<<<<<<< HEAD
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        elif rnn_class == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, dropout=dropout)
=======
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, act_fn)
        elif rnn_class == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, act_fn)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, act_fn)
>>>>>>> 2badd524df6944f3ccfb539b64655d223e72ce6e
        
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        batch_size = x.size(0)

<<<<<<< HEAD
        encoded = self.encoder(x)
        y, h = self.rnn(encoded.view(1, batch_size, -1), h)
        y = self.decoder(y.view(batch_size, -1))

        return y, h

    def init_hidden(self, batch_size):
        if self.rnn_class == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        else:
            return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
=======
        if type(h) == type(None):
            if self.rnn_class == "lstm":
                h = (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                     Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
            else:
                h = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
                
        if next(self.parameters()).is_cuda:
            h = h.cuda()

        x = self.encoder(x)
        x, h = self.rnn(x.view(1, batch_size, -1), h)
        x = self.decoder(x.view(batch_size, -1))

        return x, h
>>>>>>> 2badd524df6944f3ccfb539b64655d223e72ce6e
