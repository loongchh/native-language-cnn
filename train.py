import argparse
import os

import torch.utils.data as data
import pandas as pd


def train(args):
    vectorizer = CountVectorizer(input='file', decode_error='ignore',
                                 analyzer='char', ngram_range=(2, 2),
                                 max_features=None)
    train_file = [os.path.join(args.train, p) for p in os.listdir(args.train)]
    dev_file = [os.path.join(args.train, p) for p in os.listdir(args.dev)]
    train_labels = pd.read_csv(args.train_label)
    dev_labels = pd.read_csv(args.dev_label)


    vocab_size =


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--train', type=str, default='data/speech_transcriptions/train/tokenized', metavar='TRN',
                        help='directory containing tokenized speech transcriptions for the train set',)
    parser.add_argument('--dev', type=str, default='data/speech_transcriptions/dev/tokenized', metavar='DEV',
                        help='directory containing tokenized speech transcriptions for the dev set')
    parser.add_argument('--train-label', type=str, default='data/labels/train/labels.train.csv', metavar='TRL',
                        help='CSV of the train set labels',)
    parser.add_argument('--dev-label', type=str, default='data/labels/dev/labels.dev.csv', metavar='DEL',
                        help='CSV of the dev set labels')
    parser.add_argument('--gpu', action='store_true',
                        help='using GPU-enabled CUDA Variables. (default: False)')
    args = parser.parse_args()
    train(args)
