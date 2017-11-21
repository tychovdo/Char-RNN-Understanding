import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_class, n_layers=1, act_fn='tanh'):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_class = rnn_class
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)

        if rnn_class == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, act_fn)
        elif rnn_class == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, act_fn)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, act_fn)
        
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        batch_size = x.size(0)

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
