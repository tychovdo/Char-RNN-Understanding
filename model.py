import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_class='gru', n_layers=1, dropout=0.0):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_class = rnn_class
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)

        if rnn_class == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        elif rnn_class == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, dropout=dropout)
        
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        batch_size = x.size(0)
        encoded = self.encoder(x)
        y, h = self.rnn(encoded.view(1, batch_size, -1), h)
        y = self.decoder(y.view(batch_size, -1))

        return y, h

    def forward_gates(self, x, h):
        ''' Forward pass that also returns gate activations '''
        # TODO dropout?

        # Encode
        batch_size = x.size(0)
        encoded = self.encoder(x)

        gate_acts = []

        if self.rnn_class == 'gru':
            for layer in range(self.n_layers):
                # Get individual weights
                w_hh = getattr(self.rnn, 'weight_hh_l{}'.format(layer))
                w_ih = getattr(self.rnn, 'weight_ih_l{}'.format(layer))
                b_hh = getattr(self.rnn, 'bias_hh_l{}'.format(layer))
                b_ih = getattr(self.rnn, 'bias_ih_l{}'.format(layer))

                # Create gates functions
                if layer == 0:
                    gi = F.linear(encoded, w_ih, b_ih)
                else:
                    gi = F.linear(h[layer - 1], w_ih, b_ih)
                gh = F.linear(h[layer], w_hh, b_hh)
                i_r, i_i, i_n = gi.chunk(3, 1)
                h_r, h_i, h_n = gh.chunk(3, 1)

                # Calculate gate activations
                resetgate = F.sigmoid(i_r + h_r)
                inputgate = F.sigmoid(i_i + h_i)
                newgate = F.tanh(i_n + resetgate * h_n)

                # Store gate activations
                gate_acts.append([resetgate, inputgate, newgate])

                # Update hidden state
                h[layer] = newgate + inputgate * (h[layer] - newgate)

                # Decode output
                y = self.decoder(h[-1].view(batch_size, -1))

        elif self.rnn_class == 'lstm':
            hx, cx = h
            for layer in range(self.n_layers):
                # Get individual weights
                w_hh = getattr(self.rnn, 'weight_hh_l{}'.format(layer))
                w_ih = getattr(self.rnn, 'weight_ih_l{}'.format(layer))
                b_hh = getattr(self.rnn, 'bias_hh_l{}'.format(layer))
                b_ih = getattr(self.rnn, 'bias_ih_l{}'.format(layer))

                # Set right input
                if layer == 0:
                    gates = F.linear(encoded, w_ih, b_ih) + F.linear(hx[layer], w_hh, b_hh)
                else:
                    gates = F.linear(hx[layer - 1], w_ih, b_ih) + F.linear(hx[layer], w_hh, b_hh)

                # Create gate functions
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

                # Calculate gate activations
                ingate = F.sigmoid(ingate)
                forgetgate = F.sigmoid(forgetgate)
                cellgate = F.tanh(cellgate)
                outgate = F.sigmoid(outgate)

                # Store gate activations
                gate_acts.append([ingate, forgetgate, cellgate, outgate])

                # Update hidden state
                cx[layer] = (forgetgate * cx[layer]) + (ingate * cellgate)
                hx[layer] = outgate * F.tanh(cx[layer])

                # Decode output
                y = self.decoder(hx[-1].view(batch_size, -1))

            h = (cx, hx)

        # Return output (y), hidden (h) and gate activations (ga)
        return y, h, gate_acts

    def init_hidden(self, batch_size):
        if self.rnn_class == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        else:
            return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
