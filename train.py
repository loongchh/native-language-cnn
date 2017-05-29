import argparse
from os import listdir
from os.path import join
import logging
import pickle
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model import NativeLanguageCNN


def read_mat(file_dir, max_length, vocab_size):
    file_list = listdir(file_dir)

    # vocab_size indices stands for padding
    mat = np.ones((len(file_list), max_length), dtype=np.int64) * vocab_size

    for (i, fl) in enumerate(file_list):
        tokens = open(join(file_dir, fl)).read().split()
        length = min(len(tokens), max_length)
        mat[i, :length] = tokens[:length]

    return mat


def train(args, logger=None):
    with open(join(args.feature_dir, 'dict.pkl'), 'rb') as fpkl:
        (feature_dict, feature_rev_dict) = pickle.load(fpkl)
    n_features = len(feature_dict)
    logger.debug("number of features = {}".format(n_features))

    logger.info("Reading train dataset...")
    train_mat = read_mat(join(args.feature_dir, 'train'),
                         args.max_length, n_features)

    logger.info("Reading train labels...")
    train_lang = pd.read_csv(args.label)['L1'].values.tolist()
    lang_list = sorted(list(set(train_lang)))
    logger.debug("list of L1: {}".format(lang_list))
    lang_dict = {i: l for (i, l) in enumerate(lang_list)}
    lang_rev_dict = {l: i for (i, l) in lang_dict.items()}
    train_label = [lang_rev_dict[la] for la in train_lang]

    (train_mat, val_mat, train_label, val_label) = \
        train_test_split(train_mat, train_label, test_size=args.val_split)
    logger.debug("created train set of size {}, val set of size {}".format(
        train_mat.shape[0], val_mat.shape[0]))

    logger.info("Constructing CNN model...")
    nl_cnn = NativeLanguageCNN(n_features, args.embed_dim, args.dropout,
                               args.channel, len(lang_list))
    if args.gpu:
        logger.info("Enabling GPU computation...")
        nl_cnn.cuda()

    logger.info("Creating optimizer...")
    optimizer = optim.Adam(nl_cnn.parameters(), lr=args.lr,
                           weight_decay=args.regularization)
    logger.debug("list of parameters: {}".format(list(zip(*nl_cnn.named_parameters()))[0]))
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(torch.from_numpy(train_mat), torch.LongTensor(train_label))
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_mat_var = Variable(torch.from_numpy(val_mat))

    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    num_steps = args.num_epochs * int(np.ceil(train_mat.shape[0] / args.batch_size))

    with trange(num_steps) as progbar:
        for ep in range(args.num_epochs):
            print("Epoch #{:d} of {:d}".format(ep + 1, args.num_epochs))

            for (t, (x, y)) in enumerate(train_data_loader):
                if args.gpu:
                    x = x.cuda()
                    y = y.cuda()

                x_var = Variable(x)
                y_var = Variable(y)
                nl_cnn.train()
                score = nl_cnn(x_var)
                pred = np.argmax(score.data.cpu().numpy(), axis=1)
                train_acc.append(np.mean(pred == train_label))

                loss = criterion(score, y_var)
                train_loss.append(loss.data.cpu().numpy())
                # logger.debug("train loss = {:4f}, acc = {:4f}".format(train_loss[-1], train_acc[-1]))

                optimizer.zero_grad()
                loss.backward()
                if args.clip_norm:
                    norm = nn.utils.clip_grad_norm(nl_cnn.parameters, args.clip_norm)
                    progbar.set_postfix(loss=train_loss[-1], accuracy=train_acc[-1], norm=norm)
                    # logger.debug("grad norm = {:4f}".format(norm))
                else:
                    progbar.set_postfix(loss=train_loss[-1], accuracy=train_acc[-1])

                progbar.update(1)
                optimizer.step()

            if args.debug or (ep > 0 and (ep % args.evaluate_every) == 0):
                logger.info("Evaluating...")
                val_score = nl_cnn(val_mat_var)
                val_pred = np.argmax(val_score.data.cpu().numpy(), axis=1)
                val_acc.append(np.mean(val_pred == val_label))
                logger.info("Epoch #{:d} val accuracy = {:4f}".format(val_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=224,
                        help='seed for random initialization')
    parser.add_argument('--regularization', type=float, default=1e-4,
                        help='regularization coefficient')
    parser.add_argument('--clip-norm', type=float, default=None,
                        help='clip by total norm')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='dropout strength')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='size of mini-batch')
    parser.add_argument('--val-split', type=float, default=0.0909,
                        help='fraction of train set to use as val set')
    parser.add_argument('--max-length', type=int, default=600,
                        help='maximum feature length for each document')
    parser.add_argument('--evaluate-every', type=int, default=10,
                        help='frequency of which evaluation is done with the val set')
    parser.add_argument('--embed-dim', type=int, default=500,
                        help='dimension of the feature embeddings')
    parser.add_argument('--channel', type=int, default=500,
                        help='number of channel output for each CNN layer')
    parser.add_argument('--feature-dir', type=str, default='data/features/speech_transcriptions/ngrams/2/',
                        help='directory containing features, including train/dev directories and \
                              pickle file of (dict, rev_dict) mapping indices to feature labels')
    parser.add_argument('--label', type=str, default='data/labels/train/labels.train.csv',
                        help='CSV of the train set labels')
    parser.add_argument('--debug', action='store_true',
                        help='enable debugging messages')
    parser.add_argument('--gpu', action='store_true',
                        help='using GPU-enabled CUDA Variables')

    args = parser.parse_args()

    logging.basicConfig(level=(logging.DEBUG if args.debug else logging.INFO))
    logger = logging.getLogger("train")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args, logger)
