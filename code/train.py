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


def read_data(file_dir, label_file, val_split, vocab_size, max_len, sen_len=None, logger=None):
    """ Read train matrices and labels from specified directory.

    Given directory containing training files, produce training matrix and
    training labels read form the files. The files are lines of sentences
    containing vocabulary indices separated by spaces.

    Arguments:
        file_dir: (str) directory containing train text files, which contain
            space-separated files of vocabulary indices
        label_file: (str) file path of label CSV file
        val_split: (float) fraction of training data to use as validation set
        vocab_size: (int) size of vocabulary
        max_len: (int) trim document to length, and pad shorter documents to
            this length
        sen_len: (int) if given, break document by line to produce more training
            data (defaults: None)
        logger: (Logger) logger object to which logging descriptions are written

    Returns:
        train_mat: (numpy array) 2D training matrix of samples x document indices
        train_label: (list) labels of each training samples
        val_mat (numpy array) 2D validation matrix of samples x document indices
        val_label: (list) labels of each validation samples
        lang_dict: (dict) dictionary mapping indices to the L1 language
    """
    lang = pd.read_csv(label_file)['L1'].values.tolist()
    lang_list = sorted(list(set(lang)))  # sorted list of L1
    if logger:
        logger.debug("list of L1: {}".format(lang_list))
    lang_dict = {i: l for (i, l) in enumerate(lang_list)}   # index to L1
    lang_rev_dict = {l: i for (i, l) in lang_dict.items()}  # L1 to index
    label = [lang_rev_dict[la] for la in lang]
    pad = [vocab_size]  # vocab_size indices stands for padding

    # Split file list to train/dev by val_split
    file_list = sorted(listdir(file_dir))
    (train_file, val_file, train_label, val_label) = \
        train_test_split(file_list, label, test_size=val_split)

    sample = []
    line_label = []

    # Padding is [vocab_size]
    train_mat = vocab_size * np.ones((len(train_file), max_len), dtype=np.int64)
    for (i, fl) in enumerate(train_file):
        if sen_len:  # split train samples by line
            lines = open(join(file_dir, fl)).readlines()
            for ln in lines:
                tokens = ln.split()
                sample.append(tokens[:sen_len] + pad * (sen_len - len(tokens)))
            line_label += [label[i]] * len(lines)  # duplicate labels for all lines in file
        else:
            tokens = open(join(file_dir, fl)).read().split()
            train_mat[i, :] = tokens[:max_len] + pad * (max_len - len(tokens))

    if sen_len:
        train_mat = np.array(sample, dtype=np.int64)
        train_label = line_label

    val_mat = vocab_size * np.ones((len(val_file), max_len), dtype=np.int64)
    for (i, fl) in enumerate(val_file):
        # Validation matrices never split by line
        tokens = open(join(file_dir, fl)).read().split()
        val_mat[i, :] = tokens[:max_len] + pad * (max_len - len(tokens))

    return (train_mat, train_label, val_mat, val_label, lang_dict)


def train(args, save_dir=None, logger=None, progbar=True):
    """ Train a NativeLanguageCNN model and save model states.

    Given args specifying training hyperparameters, train a NativeLanguageCNN
    model and save model states to specified directory

    Arguments:
        args: (ArgumentParser) argument parser containing training parameters
        save_dir: (str) file directory in which model training logs and model
            states are saved
        logger: (Logger) logger object to which logging descriptions are written
        progbar: (bool) whether to show a tqdm progress bar

    Returns:
        nlcnn_model: (NativeLanguageCNN) final model after training
        train_loss: (float) final training loss
        train_f1: (float) final train set F1 score
        val_f1: (float) final dev set F1 score
    """
    # Load preprocessing pickle files mapping indices to feature (like bigrams)
    with open(join(args.feature_dir, 'dict.pkl'), 'rb') as fpkl:
        (feature_dict, feature_rev_dict) = pickle.load(fpkl)
    n_features = len(feature_dict)

    if logger:
        logger.info("Read train dataset")
        logger.debug("feature dir = {:s}, label file = {:s}".format(
            args.feature_dir, args.label))
        logger.debug("max len = {:d}, num of features = {:d}".format(
            args.max_len, n_features))

    # Read data from directory and split to train/val set
    (train_mat, train_label, val_mat, val_label, lang_dict) = \
        read_data(join(args.feature_dir, 'train'), args.label, args.val_split,
        n_features, args.max_len, logger=logger)

    if logger:
        logger.debug("created train set of size {} x {}, val set of size {} x {}".format(
            train_mat.shape[0], train_mat.shape[1], val_mat.shape[0], val_mat.shape[1]))

    # Construct NativeLanguageCNN model
        logger.info("Construct CNN model")
    nlcnn_model = NativeLanguageCNN(n_features, args.embed_dim, args.dropout,
                                    args.channel, len(lang_dict))
    if logger:
        logger.debug("embed dim={:d}, dropout={:.2f}, channels={:d}".format(
            args.embed_dim, args.dropout, args.channel))
    if args.cuda is not None:  # Enable GPU computation
        if logger:
            logger.info("Enable CUDA Device (ID #{:d})".format(args.cuda))
        nlcnn_model.cuda(args.cuda)  # place at CUDA Device with specified ID

    if logger:
        logger.info("Create optimizer")
        logger.debug("list of parameters: {}".format(list(zip(*nlcnn_model.named_parameters()))[0]))
        logger.debug("lr={:.2e}, reg={:.2e}".format(args.lr, args.reg))
    optimizer = optim.Adam(nlcnn_model.parameters(), lr=args.lr,
                           weight_decay=args.reg)  # Adam optimizer
    criterion = nn.CrossEntropyLoss()  # cross-entropy loss function

    # Create train data loader
    train_mat_tensor = torch.from_numpy(train_mat)
    train_label_tensor = torch.LongTensor(train_label)
    train_dataset = TensorDataset(train_mat_tensor, train_label_tensor)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create val data loader
    val_mat_tensor = torch.from_numpy(val_mat)
    val_label_tensor = torch.LongTensor(val_label)
    val_dataset = TensorDataset(val_mat_tensor, val_label_tensor)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Record loss/F1 at end of each epoch
    train_loss = []
    train_f1 = []
    val_f1 = []

    for ep in range(args.num_epochs):
        if logger:
            logger.info("========================================")
            logger.info("Epoch #{:d} of {:d}".format(ep + 1, args.num_epochs))

        train_pred = []  # record predictions for all batches
        train_y = []  # record ground truth labels for all batches
        loader = tqdm(train_data_loader) if progbar else train_data_loader

        for (x, y) in loader:
            if args.cuda is not None:  # GPU
                x = x.cuda(args.cuda)
                y = y.cuda(args.cuda)

            nlcnn_model.train()  # set model to train mode (influences behavior of dropout functions)
            score = nlcnn_model(Variable(x))
            pred = np.argmax(score.data.cpu().numpy(), axis=1)
            train_pred += pred.tolist()  # append all predictions from batch
            train_y += y.cpu().numpy().tolist()  # append all ground truth label from batch

            loss = criterion(score, Variable(y))  # cross-entropy loss
            optimizer.zero_grad()  # set Variables' gradient to zero
            loss.backward()  # backward pass calculates gradients

            if args.clip_norm:  # clip by gradient norm
                norm = nn.utils.clip_grad_norm(nlcnn_model.parameters(), args.clip_norm)
                if progbar:
                    loader.set_postfix(loss=loss.data.cpu().numpy()[0], norm=norm)
            else:
                if progbar:
                    loader.set_postfix(loss=loss.data.cpu().numpy()[0])

            optimizer.step()  # update model parameters according to gradients

        # Evaluate at end of each epoch
        if logger:
            logger.info("Evaluating...")
        train_loss.append(loss.data.cpu().numpy()[0])
        train_f1.append(f1_score(train_y, train_pred, average='weighted'))

        val_pred = []
        val_y = []
        for (x, y) in val_data_loader:
            if args.cuda is not None:  # GPU
                x = x.cuda(args.cuda)
                y = y.cuda(args.cuda)

            nlcnn_model.eval()  # model eval mode: no dropout
            score = nlcnn_model(Variable(x))
            pred = np.argmax(score.data.cpu().numpy(), axis=1)
            val_pred += pred.tolist()  # append all predictions from batch
            val_y += y.cpu().numpy().tolist()  # append all ground truth label from batch

        val_f1.append(f1_score(val_y, val_pred, average='weighted'))
        if logger:
            if args.clip_norm:
                logger.info("Epoch #{:d}: loss = {:.3f}, train F1 = {:.2%}, val F1 = {:.2%}, norm = {:.2f}".format(
                    ep + 1, train_loss[-1], train_f1[-1], val_f1[-1], norm))
            else:
                logger.info("Epoch #{:d}: loss = {:.3f}, train F1 = {:.2%}, val F1 = {:.2%}".format(
                    ep + 1, train_loss[-1], train_f1[-1], val_f1[-1]))

        if save_dir:  # save model state
            # Save at end of training or by frequency args.save_every
            if (ep + 1) % args.save_every == 0 or ep == args.num_epochs - 1:
                if logger:
                    logger.info("Save model-state-{:04d}.pkl".format(ep + 1))
                save_path = join(save_dir, "model-state-{:04d}.pkl".format(ep + 1))
                torch.save(nlcnn_model.state_dict(), save_path)

            save_path = join(save_dir, "model-loss-f1.pkl")
            pickle.dump((train_loss, train_f1, val_f1), open(save_path, 'wb'))

    return (nlcnn_model, train_loss, train_f1, val_f1)


if __name__ == '__main__':
    from model import NativeLanguageCNN

    parser = argparse.ArgumentParser(description='NLCNN')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=224,
                        help='seed for random initialization')
    parser.add_argument('--reg', type=float, default=0,
                        help='regularization coefficient')
    parser.add_argument('--clip-norm', type=float, default=None,
                        help='clip by total norm')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='dropout strength')
    parser.add_argument('--num-epochs', type=int, default=50,
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
    logging.basicConfig(filename=join(args.log_dir, timestamp + ".log"),
                        format='[%(asctime)s] {%(pathname)s:%(lineno)3d} %(levelname)6s - %(message)s',
                        level=logging.DEBUG, datefmt='%H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger("TRAIN")
    logger.info("Timestamp: {}".format(timestamp))

    # Set random seed
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    train(args, log_dir, logger)
