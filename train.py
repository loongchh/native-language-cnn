import argparse
from os import listdir
from os.path import join
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

from model import NativeLanguageCNN


def read_mat(file_dir, max_length):
    file_list = os.listdir(file_dir)
    mat = -np.ones((len(file_list), max_length))

    for (i, fl) in enumerate(file_list):
        tokens = open(join(file_dir, fl)).read().split()
        length = min(len(tokens), max_length)
        mat[i, :length] = tokens[:length]

    return mat


def train(args):
    train_mat = read_mat(join(args.feature_dir, 'train'), args.max_length)
    dev_mat = read_mat(join(args.feature_dir, 'dev'), args.max_length)

    lang_list = sorted(list(set(dev_label)))
    lang_dict = {i: l for (i, l) in enumerate(language_list)}
    lang_rev_dict = {l: i for (i, l) in lang_dict.items()}

    train_lang = pd.read_csv(args.train_label)['L1'].values.tolist()
    dev_lang = pd.read_csv(args.dev_label)['L1'].values.tolist()
    train_label = [lang_rev_dict[la] for la in train_lang]
    dev_label = [lang_rev_dict[la] for la in dev_lang]

    with open(join(args.feature_dir, 'dict_rev_dict.pkl'), 'rb') as fpkl:
        (feature_dict, feature_rev_dict) = pickle.load(fpkl)
    n_features = len(feature_dict)

    nl_cnn_model = NativeLanguageCNN(n_features, args.embed_dim, args.channel, len(lang_list))
    if args.gpu:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.regularization)
    criterion = nn.CrossEntropyLoss()

    #TODO: split out validation set
    train_dataset = TensorDataset(torch.from_numpy(train_mat), torch.IntTensor(train_label))
    train_data_loader = DataLoader(train_dataset)

    for ep in tqdm(range(args.num_epochs)):
        for t, (x, y) in enumerate(train_data_loader):
            if args.gpu:
                x = x.cuda()
                y = y.cuda()

            x_var = Variable(x)
            y_var = Variable(y.long())
            score = nl_cnn_model(x_var)
            loss = criterion(score, y_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #TODO: print out losses every epoch
    #TODO: evaluate on val set


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--regularization', type=float, default=1e-4,
                        help='regularization coefficient')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='size of mini-batch')
    parser.add_argument('--max-length', type=int, default=500,
                        help='maximum feature length for each document')
    parser.add_argument('--embed-dim', type=int, default=500,
                        help='dimension of the feature embeddings')
    parser.add_argument('--channel', type=int, default=500,
                        help='number of channel output for each CNN layer')
    parser.add_argument('--feature-dir', type=str, default='data/features/speech_transcriptions/ngrams/2/',
                        help='directory containing features, including train/dev directories and \
                              pickle file of (dict, rev_dict) mapping indices to feature labels')
    parser.add_argument('--train-label', type=str, default='data/labels/train/labels.train.csv',
                        help='CSV of the train set labels')
    parser.add_argument('--dev-label', type=str, default='data/labels/dev/labels.dev.csv',
                        help='CSV of the dev set labels')
    parser.add_argument('--gpu', action='store_true',
                        help='using GPU-enabled CUDA Variables')
    args = parser.parse_args()
    train(args)
