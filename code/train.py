import argparse
from os import listdir, makedirs
from os.path import join
import logging
from time import strftime
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from model import NativeLanguageCNN


def read_data(file_dir, label_file, val_split, vocab_size, max_len, sen_len=None, logger=None):
    """
    """
    lang = pd.read_csv(label_file)['L1'].values.tolist()
    lang_list = sorted(list(set(lang)))
    if logger:
        logger.debug("list of L1: {}".format(lang_list))
    lang_dict = {i: l for (i, l) in enumerate(lang_list)}
    lang_rev_dict = {l: i for (i, l) in lang_dict.items()}
    label = [lang_rev_dict[la] for la in lang]
    pad = [vocab_size]  # vocab_size indices stands for padding

    file_list = sorted(listdir(file_dir))
    (train_file, val_file, train_label, val_label) = \
        train_test_split(file_list, label, test_size=val_split)

    sample = []
    line_label = []
    train_mat = vocab_size * np.ones((len(train_file), max_len), dtype=np.int64)
    for (i, fl) in enumerate(train_file):
        if sen_len:
            lines = open(join(file_dir, fl)).readlines()
            for ln in lines:
                tokens = ln.split()
                sample.append(tokens[:sen_len] + pad * (sen_len - len(tokens)))
            line_label += [label[i]] * len(lines)
        else:
            tokens = open(join(file_dir, fl)).read().split()
            train_mat[i, :] = tokens[:max_len] + pad * (max_len - len(tokens))

    if sen_len:
        train_mat = np.array(sample, dtype=np.int64)
        train_label = line_label

    val_mat = vocab_size * np.ones((len(val_file), max_len), dtype=np.int64)
    for (i, fl) in enumerate(val_file):
        tokens = open(join(file_dir, fl)).read().split()
        val_mat[i, :] = tokens[:max_len] + pad * (max_len - len(tokens))

    return (train_mat, train_label, val_mat, val_label, lang_dict)


def train(args, save_dir=None, logger=None, progbar=True):
    with open(join(args.feature_dir, 'dict.pkl'), 'rb') as fpkl:
        (feature_dict, feature_rev_dict) = pickle.load(fpkl)
    n_features = len(feature_dict)

    if logger:
        logger.info("Read train dataset")
        logger.debug("feature dir = {:s}, label file = {:s}".format(
            args.feature_dir, args.label))
        logger.debug("max len = {:d}, num of features = {:d}".format(
            args.max_len, n_features))
    (train_mat, train_label, val_mat, val_label, lang_dict) = \
        read_data(join(args.feature_dir, 'train'), args.label, args.val_split,
        n_features, args.max_len, logger=logger)
    if logger:
        logger.debug("created train set of size {} x {}, val set of size {} x {}".format(
            train_mat.shape[0], train_mat.shape[1], val_mat.shape[0], val_mat.shape[1]))

        logger.info("Construct CNN model")
    nlcnn_model = NativeLanguageCNN(n_features, args.embed_dim, args.dropout,
                                    args.channel, len(lang_dict))
    if logger:
        logger.debug("embed dim={:d}, dropout={:.2f}, channels={:d}".format(
            args.embed_dim, args.dropout, args.channel))
    if args.cuda is not None:
        if logger:
            logger.info("Enable CUDA Device (ID #{:d})".format(args.cuda))
        nlcnn_model.cuda(args.cuda)

    if logger:
        logger.info("Create optimizer")
        logger.debug("list of parameters: {}".format(list(zip(*nlcnn_model.named_parameters()))[0]))
        logger.debug("lr={:.2e}, regularization={:.2e}".format(args.lr, args.regularization))
    optimizer = optim.Adam(nlcnn_model.parameters(), lr=args.lr,
                           weight_decay=args.regularization)
    criterion = nn.CrossEntropyLoss()

    train_mat_tensor = torch.from_numpy(train_mat)
    train_label_tensor = torch.LongTensor(train_label)

    train_dataset = TensorDataset(train_mat_tensor, train_label_tensor)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_mat_var = Variable(torch.from_numpy(val_mat).cuda(args.cuda) if args.cuda is not None
                           else torch.from_numpy(val_mat))

    train_loss = []
    train_f1 = []
    val_f1 = []

    for ep in range(args.num_epochs):
        if logger:
            logger.info("========================================")
            logger.info("Epoch #{:d} of {:d}".format(ep + 1, args.num_epochs))

        train_pred = []
        train_y = []
        loader = tqdm(train_data_loader) if progbar else train_data_loader



        for (x, y) in loader:
            if args.cuda is not None:
                x = x.cuda(args.cuda)
                y = y.cuda(args.cuda)

            x_var = Variable(x)
            y_var = Variable(y)

            nlcnn_model.train()
            score = nlcnn_model(x_var)
            pred = np.argmax(score.data.cpu().numpy(), axis=1)
            train_pred += pred.tolist()
            train_y += y.cpu().numpy().tolist()

            loss = criterion(score, y_var)
            optimizer.zero_grad()
            loss.backward()
            if args.clip_norm:  # clip by gradient norm
                norm = nn.utils.clip_grad_norm(nlcnn_model.parameters, args.clip_norm)
                if progbar:
                    loader.set_postfix(loss=loss.data.cpu().numpy()[0], norm=norm)
            else:
                if progbar:
                    loader.set_postfix(loss=loss.data.cpu().numpy()[0])

            optimizer.step()

        if logger:
            logger.info("Evaluating...")
        train_loss.append(loss.data.cpu().numpy()[0])
        train_f1.append(f1_score(train_y, train_pred, average='weighted'))

        nlcnn_model.eval()  # eval mode: no dropout
        val_score = nlcnn_model(val_mat_var)
        val_pred = np.argmax(val_score.data.cpu().numpy(), axis=1)
        val_f1.append(f1_score(val_label, val_pred, average='weighted'))
        if logger:
            logger.info("Epoch #{:d}: loss = {:.3f}, train F1 = {:.2%}, val F1 = {:.2%}".format(
                ep + 1, train_loss[-1], train_f1[-1], val_f1[-1]))

        if save_dir:  # save model state
            if (ep + 1) % args.save_every == 0 or ep == args.num_epochs - 1:
                if logger:
                    logger.info("Save model-state-{:04d}.pkl".format(ep + 1))
                save_path = join(save_dir, "model-state-{:04d}.pkl".format(ep + 1))
                torch.save(nlcnn_model.state_dict(), save_path)

    if save_dir:  # save loss/f1 histroy
        save_path = join(save_dir, "model-loss-f1.pkl")
        pickle.dump((train_loss, train_f1, val_f1), open(save_path, 'wb'))

    return (nlcnn_model, train_loss, train_f1, val_f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLCNN')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=224,
                        help='seed for random initialization')
    parser.add_argument('--regularization', type=float, default=0,
                        help='regularization coefficient')
    parser.add_argument('--clip-norm', type=float, default=None,
                        help='clip by total norm')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='dropout strength')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=25,
                        help='size of mini-batch')
    parser.add_argument('--val-split', type=float, default=0.0909,
                        help='fraction of train set to use as val set')
    parser.add_argument('--max-len', type=int, default=600,
                        help='maximum feature length for each document')
    parser.add_argument('--embed-dim', type=int, default=500,
                        help='dimension of the feature embeddings')
    parser.add_argument('--channel', type=int, default=500,
                        help='number of channel output for each CNN layer')
    parser.add_argument('--feature-dir', type=str, default='data/features/speech_transcriptions/ngrams/2',
                        help='directory containing features, including train/dev directories and \
                              pickle file of (dict, rev_dict) mapping indices to feature labels')
    parser.add_argument('--label', type=str, default='data/labels/train/labels.train.csv',
                        help='CSV of the train set labels')
    parser.add_argument('--log-dir', type=str, default='model',
                        help='directory in which model states are to be saved')
    parser.add_argument('--save-every', type=int, default=10,
                        help='epoch frequncy of saving model state to directory')
    parser.add_argument('--cuda', type=int,
                        help='CUDA device to use')
    args = parser.parse_args()

    # Create log directory + file
    timestamp = strftime("%Y-%m-%d-%H%M%S")
    log_dir = join(args.log_dir, timestamp)
    makedirs(log_dir)

    # Setup logger
    logging.basicConfig(filename=join(log_dir, timestamp + ".log"),
                        format='[%(asctime)s] {%(pathname)s:%(lineno)3d} %(levelname)6s - %(message)s',
                        level=logging.DEBUG, datefmt='%H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger("TRAIN")

    # Set random seed
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    train(args, log_dir, logger)
