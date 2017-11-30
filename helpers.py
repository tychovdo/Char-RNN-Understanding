import unidecode
import string
import random
import time
import math
import torch
import numpy as np

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

def wrap_colored_text(text, colors, W=20):
    ''' Wrap text with hidden units to certain width '''
    new_text, new_colors = [], []
    for char, color in zip(text, colors):
        new_text.append(char)
        new_colors.append(color)
        if char == '\n':
            for i in range(W - len(new_text) % W):
                new_text.append(' ')
                new_colors.append(np.zeros_like(colors[0]))
    for i in range(W - len(new_text) % W):
        new_text.append(' ')
        new_colors.append(np.zeros_like(colors[0]))
    
    return np.array(new_text).reshape(-1, W), np.array(new_colors).reshape(-1, W)

def plot_colored_text(text, colors, W=60, title=None, save_file=None):
    wrapped_text, wrapped_colors = wrap_colored_text(text, colors, W=W)
    
    H = len(wrapped_text)
    plt.figure(figsize=(W/5, H/3))
    sns.heatmap(wrapped_colors, annot=wrapped_text, fmt='s',
                cbar=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    if save_file:
        plt.savefig(save_file)
    plt.show()