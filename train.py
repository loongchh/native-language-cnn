import argparse
from os import listdir, makedirs
from os.path import join
import logging
from time import strftime
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


def read_data(file_dir, label_file, max_length, vocab_size, logger=None, line=True):
    lang = pd.read_csv(label_file)['L1'].values.tolist()
    lang_list = sorted(list(set(lang)))
    if logger:
        logger.debug("list of L1: {}".format(lang_list))
    lang_dict = {i: l for (i, l) in enumerate(lang_list)}
    lang_rev_dict = {l: i for (i, l) in lang_dict.items()}
    label = [lang_rev_dict[la] for la in lang]

    samples = []
    label_line = []
    pad = [vocab_size]  # vocab_size indices stands for padding

    for (i, fl) in enumerate(listdir(file_dir)):
        if line:
            lines = open(join(file_dir, fl)).readlines()
            for ln in lines:
                tokens = ln.split()
                samples.append(tokens[:max_length] + pad * (max_length - len(tokens)))
            label_line += [label[i]] * len(lines)
        else:
            tokens = open(join(file_dir, fl)).read().split()
            samples += tokens[:max_length] + pad * (max_length - len(tokens))

    if line:
        label = label_line
    mat = np.array(samples, dtype=np.int64)

    return (mat, label, lang_dict)


def train(args, logger, log_dir):
    with open(join(args.feature_dir, 'dict.pkl'), 'rb') as fpkl:
        (feature_dict, feature_rev_dict) = pickle.load(fpkl)
    n_features = len(feature_dict)

    logger.info("Read train dataset")
    logger.debug("feature dir = {:s}, label file = {:s}".format(
        args.feature_dir, args.label))
    logger.debug("max len = {:d}, num of features = {:d}".format(
        args.max_length, n_features))
    (train_mat, train_label, lang_dict) = read_data(join(args.feature_dir, 'train'),
                                                    args.label, args.max_length,
                                                    n_features, logger)

    # Split into train/val set
    (train_mat, val_mat, train_label, val_label) = \
        train_test_split(train_mat, train_label, test_size=args.val_split)
    logger.debug("created train set of size {}, val set of size {}".format(
        train_mat.shape[0], val_mat.shape[0]))

    logger.info("Construct CNN model")
    nlcnn_model = NativeLanguageCNN(n_features, args.embed_dim, args.dropout,
                                    args.channel, len(lang_dict))
    logger.debug("embed dim={:d}, dropout={:.2f}, channels={:d}".format(
        args.embed_dim, args.dropout, args.channel))
    if args.gpu:
        logger.info("Enable GPU computation")
        nlcnn_model.cuda()

    logger.info("Create optimizer")
    optimizer = optim.Adam(nlcnn_model.parameters(), lr=args.lr,
                           weight_decay=args.regularization)
    logger.debug("list of parameters: {}".format(list(zip(*nlcnn_model.named_parameters()))[0]))
    logger.debug("lr={:.2e}, regularization={:.2e}".format(args.lr, args.regularization))
    criterion = nn.CrossEntropyLoss()

    train_mat_tensor = torch.from_numpy(train_mat)
    train_label_tensor = torch.LongTensor(train_label)

    train_dataset = TensorDataset(train_mat_tensor, train_label_tensor)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_mat_var = Variable(torch.from_numpy(val_mat).cuda() if args.gpu \
                           else torch.from_numpy(val_mat))

    train_loss = []
    train_acc = []
    val_acc = []
    steps_per_epoch = int(np.ceil(train_mat.shape[0] / args.batch_size))

    for ep in range(args.num_epochs):
        logger.info("========================================")
        logger.info("Epoch #{:d} of {:d}".format(ep + 1, args.num_epochs))
        batch_acc = []

        with trange(steps_per_epoch) as progbar:
            for (x, y) in train_data_loader:
                if args.gpu:
                    x = x.cuda()
                    y = y.cuda()
                x_var = Variable(x)
                y_var = Variable(y)

                nlcnn_model.train()
                score = nlcnn_model(x_var)
                pred = np.argmax(score.data.cpu().numpy(), axis=1)
                batch_acc.append(np.mean(pred == y.cpu().numpy()))
                loss = criterion(score, y_var)

                optimizer.zero_grad()
                loss.backward()
                if args.clip_norm:  # clip by gradient norm
                    norm = nn.utils.clip_grad_norm(nlcnn_model.parameters, args.clip_norm)
                    progbar.set_postfix(loss=loss.data.cpu().numpy()[0], norm=norm)
                else:
                    progbar.set_postfix(loss=loss.data.cpu().numpy()[0])

                progbar.update(1)
                optimizer.step()

        train_loss.append(loss.data.cpu().numpy()[0])
        train_acc.append(np.mean(batch_acc))

        logger.info("Evaluating...")
        nlcnn_model.eval()  # eval mode: no dropout
        val_score = nlcnn_model(val_mat_var)
        val_pred = np.argmax(val_score.data.cpu().numpy(), axis=1)
        val_acc.append(np.mean(val_pred == val_label))
        logger.info("Epoch #{:d}: loss = {:.4f}, train acc = {:.4f}, val acc = {:.4f}".format(
            ep + 1, train_loss[-1], train_acc[-1], val_acc[-1]))

        # Save model state
        if (ep + 1) % args.save_every == 0 or ep == args.num_epochs - 1:
            logger.info("Saving model-state-{:04d}.pkl...".format(ep + 1))
            save_path = join(log_dir, "model-state-{:04d}.pkl".format(ep + 1))
            torch.save(nlcnn_model.state_dict(), save_path)


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
    parser.add_argument('--max-length', type=int, default=200,
                        help='maximum feature length for each document')
    parser.add_argument('--embed-dim', type=int, default=500,
                        help='dimension of the feature embeddings')
    parser.add_argument('--channel', type=int, default=500,
                        help='number of channel output for each CNN layer')
    parser.add_argument('--feature-dir', type=str, default='data/features/speech_transcriptions/ngrams/2',
                        help='directory containing features, including train/dev directories and \
                              pickle file of (dict, rev_dict) mapping indices to feature labels')
    parser.add_argument('--log-dir', type=str, default='model',
                        help='directory in which model states are to be saved')
    parser.add_argument('--save-every', type=int, default=10,
                        help='epoch frequncy of saving model state to directory')
    parser.add_argument('--label', type=str, default='data/labels/train/labels.train.csv',
                        help='CSV of the train set labels')
    parser.add_argument('--gpu', action='store_true',
                        help='using GPU-enabled CUDA Variables')
    # parser.add_argument('--line', action='store_true',
    #                     help='create train set split by lines')
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

    train(args, logger, log_dir)
