import unidecode
import string
import random
import time
import math
import torch

# Reading and un-unicode-encoding data

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

