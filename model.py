import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np


class NativeLanguageCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout, out_channel, n_language):
        super(NativeLanguageCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size)
        self.dropout = nn.Dropout(dropout, inplace=True)

        self.cnn3 = nn.Conv2d(1, out_channel, (3, embed_dim))
        self.cnn4 = nn.Conv2d(1, out_channel, (4, embed_dim))
        self.cnn5 = nn.Conv2d(1, out_channel, (5, embed_dim))
        self.cnn6 = nn.Conv2d(1, out_channel, (6, embed_dim))

        self.linear = nn.Linear(out_channel * 4, n_language)

    def forward(self, x):
        embedding = self.dropout(self.embed(x)).unsqueeze(1)

        h3 = self.cnn3(embedding).max(dim=2)[0].squeeze(-1)
        h4 = self.cnn4(embedding).max(dim=2)[0].squeeze(-1)
        h5 = self.cnn5(embedding).max(dim=2)[0].squeeze(-1)
        h6 = self.cnn6(embedding).max(dim=2)[0].squeeze(-1)

        h_cnn = torch.cat((h3, h4, h5, h6), dim=1).squeeze(-1)
        out = self.linear(h_cnn)
        return out
