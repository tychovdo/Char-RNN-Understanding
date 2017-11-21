import torch
import os
import argparse
import numpy as np

from helpers import *
from model import *

def generate(model, prime_str='A', predict_len=100, temperature=0.8, cuda=False):
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))
    hidden = None
    
    if cuda:
        model.cuda()
        prime_input = prime_input.cuda()
    else:
        model.cpu()

    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[:,p], hidden)
        
    x = prime_input[:,-1]
    
    hiddens = []
    for p in range(predict_len):
        output, hidden = model(x, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        all_characters = string.printable
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        hiddens.append(hidden.data.cpu().numpy())
        x = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            x = x.cuda()

    return predicted, np.array(hiddens)

# Run as standalone script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('-p', '--prime_str', type=str, default='A')
    parser.add_argument('-l', '--predict_len', type=int, default=100)
    parser.add_argument('-t', '--temperature', type=float, default=0.8)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    model = torch.load(args.filename)
    del args.filename
    generated, _ = generate(model, **vars(args))
    print(generated)
