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

        self.cnn2 = nn.Conv2d(1, out_channel, (2, embed_dim))
        self.cnn3 = nn.Conv2d(1, out_channel, (3, embed_dim))
        self.cnn4 = nn.Conv2d(1, out_channel, (4, embed_dim))
        # self.cnn5 = nn.Conv2d(1, out_channel, (5, embed_dim))

        self.batch_norm2 = nn.BatchNorm2d(out_channel)
        self.batch_norm3 = nn.BatchNorm2d(out_channel)
        self.batch_norm4 = nn.BatchNorm2d(out_channel)
        # self.batch_norm5 = nn.BatchNorm2d(out_channel)

        self.linear = nn.Linear(out_channel * 3, n_language)

    def forward(self, x):
        embedding = self.dropout(self.embed(x)).unsqueeze(1)

        h2 = self.batch_norm2(self.cnn2(embedding)).max(dim=2)[0].squeeze(-1)
        h3 = self.batch_norm3(self.cnn3(embedding)).max(dim=2)[0].squeeze(-1)
        h4 = self.batch_norm4(self.cnn4(embedding)).max(dim=2)[0].squeeze(-1)
        # h5 = self.batch_norm5(self.cnn5(embedding)).max(dim=2)[0].squeeze(-1)

        h_cnn = torch.cat((h2, h3, h4), dim=1).squeeze(-1)
        out = self.linear(h_cnn)
        return out
