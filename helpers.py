import unidecode
import string
import random
import time
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import Variable

# Turning a string into a tensor
def char_tensor(text):
    tensor = torch.zeros(len(text)).long()
    for c in range(len(text)):
        try:
            tensor[c] = string.printable.index(text[c])
        except:
            continue
    return tensor

# Readable time elapsed
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Plotting helper
def wrap_colored_text(text, colors, W, tabsize=4):
    ''' Wrap text with hidden units to certain width '''
    new_text, new_colors = [], []

    # Wrap lines to width
    for char, color in zip(text, colors):
        if char == '\t':
            for i in range(tabsize):
                new_text.append(' ')
                new_colors.append(color)
        else:
            new_text.append(char)
            new_colors.append(color)
            if char == '\n':
                for i in range(W - len(new_text) % W):
                    new_text.append(' ')
                    new_colors.append(np.zeros_like(colors[0]))

    # Add final space to obtain (rectangular) matrix
    for i in range(W - len(new_text) % W):
        new_text.append(' ')
        new_colors.append(np.zeros_like(colors[0]))

    return np.array(new_text).reshape(-1, W), np.array(new_colors).reshape(-1, W)

def plot_colored_text(text, colors, W=60, title=None, save_file=None):
    wrapped_text, wrapped_colors = wrap_colored_text(text, colors, W, tabsize=4)

    H = len(wrapped_text)
    plt.figure(figsize=(W/5, H/3))

    color_palette = sns.color_palette("RdBu_r", 255, desat=.9)
    color_palette[127] = (1.,1.,1.) # set zero to white color
    sns.heatmap(wrapped_colors, annot=wrapped_text,
                annot_kws={'color':'black',
                           'family':'monospace',
                           'horizontalalignment':'center',
                           'fontweight':'light'},
                fmt='s', cbar=False, cmap=color_palette, vmin=-2, vmax=2)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    if save_file:
        plt.savefig(save_file)
    plt.show()

# Test model
def test_model(model, test_file, batch_size=128, chunk_len=200, cuda=True):
    model.eval()

    f = unidecode.unidecode(open(test_file, encoding='utf-8', errors='ignore').read())

    if cuda:
        x = torch.cuda.LongTensor(batch_size, chunk_len)
        t = torch.cuda.LongTensor(batch_size, chunk_len)
    else:
        x = torch.LongTensor(batch_size, chunk_len)
        t = torch.LongTensor(batch_size, chunk_len)

    for bi in range(batch_size):
        start_index = random.randint(0, len(f) - chunk_len - 1)
        end_index = start_index + chunk_len + 1
        chunk = f[start_index:end_index]

        x[bi] = char_tensor(chunk[:-1])
        t[bi] = char_tensor(chunk[1:])
    x, t = Variable(x), Variable(t)

    h = model.init_hidden(batch_size)
    if cuda:
        if isinstance(h, tuple):
            h = (h[0].cuda(), h[1].cuda())
        else:
            h = h.cuda()

    criterion = nn.CrossEntropyLoss()

    loss = 0
    for c in range(chunk_len):
        y, h = model(x[:,c], h)
        loss += criterion(y.view(batch_size, -1), t[:, c])

    return loss.data[0] / chunk_len
