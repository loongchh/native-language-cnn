import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np


def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, out_channel, n_language):
        super(CNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.cnn3 = nn.Conv1d(1, out_channel, 3)
        self.cnn4 = nn.Conv1d(1, out_channel, 4)
        self.cnn5 = nn.Conv1d(1, out_channel, 5)

        self.linear = nn.Linear(out_channel * 3, n_language)

        self.weight_init()

    def forward(self, x):
        embedding = self.embed(x)

        h3 = self.cnn3(embedding).max(dim=2)
        h4 = self.cnn4(embedding).max(dim=2)
        h5 = self.cnn5(embedding).max(dim=2)

        h_cnn = torch.cat((h3, h4, h5), dim=1)
        out = self.linear(h_cnn)
        return out
